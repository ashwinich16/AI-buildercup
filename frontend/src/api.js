// Talks to the Scout backend.
// Every request carries the current shopper's credentials. How they look depends on how
// the shopper signed in, so api.js doesn't need to know:
//   demo:     "X-Scout-User: <persona id>"
//   firebase: "Authorization: Bearer <Google sign-in token>" (Firebase refreshes it automatically)

let credentials = null;

const SESSION_TIMEOUT_MS = 25000; // opening a conversation normally takes a second or two

export function setCredentials(newCredentials) {
  credentials = newCredentials;
}

export const demoCredentials = (personaId) => ({
  headers: async () => ({ "X-Scout-User": personaId }),
});

export const firebaseCredentials = (firebaseUser) => ({
  headers: async () => ({ Authorization: `Bearer ${await firebaseUser.getIdToken()}` }),
});

// `timeoutMs`: give up after this long with a clear message, instead of waiting forever.
async function request(path, { method = "GET", body, timeoutMs } = {}) {
  const controller = new AbortController();
  const timer = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : null;
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: {
        "Content-Type": "application/json",
        ...(credentials ? await credentials.headers() : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Scout is taking too long to answer. Please try again.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return response;
}

export const api = {
  authConfig: async () => (await request("/api/auth-config")).json(),
  // Carries on the shopper's last conversation (returns its `history`), unless `fresh` asks for a new one.
  startSession: async (preferredName = "", fresh = false) =>
    (
      await request("/api/sessions", {
        method: "POST",
        body: { preferred_name: preferredName, fresh },
        timeoutMs: SESSION_TIMEOUT_MS,
      })
    ).json(),
  endSession: async (sessionId) =>
    (await request(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })).json(),
  // Scout's reply as it is written: onEvent gets {type: "delta" | "screen" | "done"} items.
  // Resolves with the final "done" item.
  chatStream: async (sessionId, message, onEvent) => {
    const response = await request("/api/chat/stream", { method: "POST", body: { session_id: sessionId, message } });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let done = null;
    for (;;) {
      const { value, done: finished } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !finished });
      const lines = buffer.split("\n");
      buffer = finished ? "" : lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const item = JSON.parse(line);
        if (item.type === "error") throw new Error(item.detail);
        if (item.type === "done") done = item;
        onEvent(item);
      }
      if (finished) break;
    }
    if (!done) throw new Error("The reply was cut off.");
    return done;
  },
  // The shopper's photo ({ data: base64, mimeType }) wearing the products: resolves with a PNG blob.
  // `request`: { productIds, wardrobeIds } — products from the shop, and pieces they own (for a planned look).
  productDetails: async (sessionId, productId) =>
    (await request("/api/product", { method: "POST", body: { session_id: sessionId, product_id: productId } })).json(),
  tryOn: async (sessionId, what, photo, wish = "") =>
    (
      await request("/api/try-on", {
        method: "POST",
        body: {
          session_id: sessionId,
          product_ids: what.productIds || [],
          wardrobe_ids: what.wardrobeIds || [],
          photo: photo.data,
          mime_type: photo.mimeType,
          wish,
        },
      })
    ).blob(),
  // Clothes they own, from a photo file: made small here (quick to send), then read by Scout.
  // Resolves with { wardrobe, added }.
  addToWardrobe: async (sessionId, file) => {
    const image = await loadImage(file);
    const photo = resized(image, PHOTO_SIZE).split(",")[1];
    const body = { session_id: sessionId, photo, mime_type: "image/jpeg", thumbnail: resized(image, THUMBNAIL_SIZE) };
    return (await request("/api/wardrobe", { method: "POST", body })).json();
  },
  removeFromWardrobe: async (sessionId, itemId) =>
    (
      await request(`/api/wardrobe/${encodeURIComponent(itemId)}?session_id=${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      })
    ).json(),
  speech: async (text, rate) =>
    (await request("/api/speech", { method: "POST", body: { text, rate } })).blob(),
};

const PHOTO_SIZE = 768;     // longest side of a wardrobe photo sent to be read: plenty for Gemini, quick to upload
const THUMBNAIL_SIZE = 512; // longest side of the picture kept of the piece: shown on the board and tried on

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("That photo couldn't be opened."));
    };
    image.src = url;
  });
}

// The image as a JPEG data URL, no larger than `size` on its longest side.
function resized(image, size) {
  const scale = Math.min(1, size / Math.max(image.width, image.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);
  canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.85);
}
