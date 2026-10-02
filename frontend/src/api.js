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
  chat: async (sessionId, message) =>
    (await request("/api/chat", { method: "POST", body: { session_id: sessionId, message } })).json(),
  speech: async (text, rate) =>
    (await request("/api/speech", { method: "POST", body: { text, rate } })).blob(),
};
