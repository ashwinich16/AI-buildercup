// Talks to the Scout backend.
// Every request carries the current shopper's credentials. How they look depends on how
// the shopper signed in, so api.js doesn't need to know:
//   demo:     "X-Scout-User: <persona id>"
//   firebase: "Authorization: Bearer <Google sign-in token>" (Firebase refreshes it automatically)

let credentials = null;

export function setCredentials(newCredentials) {
  credentials = newCredentials;
}

export const demoCredentials = (personaId) => ({
  headers: async () => ({ "X-Scout-User": personaId }),
});

export const firebaseCredentials = (firebaseUser) => ({
  headers: async () => ({ Authorization: `Bearer ${await firebaseUser.getIdToken()}` }),
});

async function request(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(credentials ? await credentials.headers() : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
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
  startSession: async (preferredName = "") =>
    (await request("/api/sessions", { method: "POST", body: { preferred_name: preferredName } })).json(),
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
  tryOn: async (sessionId, productIds, photo) =>
    (
      await request("/api/try-on", {
        method: "POST",
        body: { session_id: sessionId, product_ids: productIds, photo: photo.data, mime_type: photo.mimeType },
      })
    ).blob(),
  speech: async (text, rate) =>
    (await request("/api/speech", { method: "POST", body: { text, rate } })).blob(),
};
