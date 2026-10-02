import { useEffect, useRef, useState } from "react";
import { api, demoCredentials, firebaseCredentials } from "../api.js";

// The start screen. The backend says how shoppers sign in (/api/auth-config):
//   demo     -> pick one of the demo shoppers
//   firebase -> Sign in with Google
// Either way it hands App a shopper: { name, preferredName, credentials, leave: { label, action } }.
export default function SignIn({ onSignedIn }) {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState("");
  const headingRef = useRef(null);

  useEffect(() => {
    headingRef.current?.focus();
    api.authConfig().then(setConfig).catch(() => setError("I couldn't reach Scout. Is the server running?"));
  }, []);

  return (
    <main className="picker" aria-labelledby="picker-title">
      <div className="picker__intro">
        <p className="brand brand--large" aria-hidden="true">
          Scout
        </p>
        <h1 id="picker-title" tabIndex={-1} ref={headingRef}>
          {config?.mode === "demo" ? "Who's shopping today?" : "Welcome to Scout"}
        </h1>
        <p className="muted">
          Scout is a friendly shopping companion you can talk to. It keeps your cart and remembers your taste, just
          for you.
        </p>
      </div>

      {!config && !error && <p className="muted">Getting ready…</p>}
      {error && (
        <p role="alert" className="notice notice--error">
          {error}
        </p>
      )}

      {config?.mode === "firebase" && (
        <GoogleSignIn settings={config.firebase} onSignedIn={onSignedIn} onError={setError} />
      )}
      {config?.mode === "demo" && <DemoShoppers personas={config.personas} onSignedIn={onSignedIn} />}
    </main>
  );
}

function DemoShoppers({ personas, onSignedIn }) {
  const pick = (persona) =>
    onSignedIn({
      name: persona.name,
      credentials: demoCredentials(persona.id),
      leave: { label: "Switch shopper", action: async () => {} },
    });

  return (
    <ul className="picker__list" role="list">
      {personas.map((persona) => (
        <li key={persona.id}>
          <button className="persona" onClick={() => pick(persona)}>
            <span className="persona__avatar" aria-hidden="true">
              {persona.name.charAt(0)}
            </span>
            <span className="persona__text">
              <span className="persona__name">{persona.name}</span>
              <span className="persona__description">{persona.description}</span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

// Firebase is loaded only in firebase mode, so demo mode stays light.
async function connectFirebase(settings) {
  const [{ initializeApp, getApps }, auth] = await Promise.all([import("firebase/app"), import("firebase/auth")]);
  const firebaseAuth = auth.getAuth(getApps()[0] || initializeApp(settings));
  return {
    onUserChanged: (callback) => auth.onAuthStateChanged(firebaseAuth, callback),
    signIn: () => auth.signInWithPopup(firebaseAuth, new auth.GoogleAuthProvider()),
    signOut: () => auth.signOut(firebaseAuth),
  };
}

function GoogleSignIn({ settings, onSignedIn, onError }) {
  const [firebase, setFirebase] = useState(null);
  const [callMe, setCallMe] = useState(""); // optional; Scout remembers it, empty = name from Google
  const [user, setUser] = useState(undefined); // undefined = still checking, null = signed out

  useEffect(() => {
    let unsubscribe = () => {};
    connectFirebase(settings)
      .then((connected) => {
        setFirebase(connected);
        unsubscribe = connected.onUserChanged(setUser);
      })
      .catch(() => onError("Sign-in isn't available right now. Please try again later."));
    return () => unsubscribe();
  }, [settings, onError]);

  const start = (signedInUser) =>
    onSignedIn({
      name: callMe.trim() || signedInUser.displayName || "",
      preferredName: callMe.trim(),
      credentials: firebaseCredentials(signedInUser),
      leave: { label: "Sign out", action: firebase.signOut },
    });

  const signIn = async () => {
    try {
      start((await firebase.signIn()).user);
    } catch (error) {
      if (error?.code === "auth/popup-closed-by-user") return;
      console.error("Google sign-in failed", error);
      onError(`Sign-in didn't work (${error?.code || "unknown error"}). Please try again.`);
    }
  };

  if (!firebase || user === undefined) return <p className="muted">Getting ready…</p>;

  // Pressing Enter in the name box does the main action (continue, or sign in).
  const submit = (event) => {
    event.preventDefault();
    if (user) start(user);
    else signIn();
  };

  const nameBox = (
    <label className="name-field">
      <span className="name-field__label">What should Scout call you? (optional)</span>
      <input
        type="text"
        autoComplete="nickname"
        maxLength={40}
        value={callMe}
        onChange={(event) => setCallMe(event.target.value)}
        aria-describedby="name-field-hint"
      />
      <span id="name-field-hint" className="muted">
        Leave it empty to use your Google name, or the name you gave last time.
      </span>
    </label>
  );

  // Already signed in on this browser: one press to continue, no need to sign in again.
  if (user) {
    return (
      <form className="picker__list" onSubmit={submit}>
        {nameBox}
        <button type="submit" className="persona">
          <span className="persona__avatar" aria-hidden="true">
            {(user.displayName || "?").charAt(0)}
          </span>
          <span className="persona__text">
            <span className="persona__name">Continue as {user.displayName || user.email}</span>
            <span className="persona__description">Start talking with Scout</span>
          </span>
        </button>
        <button type="button" className="button button--quiet" onClick={firebase.signOut}>
          Use a different account
        </button>
      </form>
    );
  }

  return (
    <form className="picker__list" onSubmit={submit}>
      {nameBox}
      <button type="submit" className="persona">
        <span className="persona__avatar" aria-hidden="true">
          G
        </span>
        <span className="persona__text">
          <span className="persona__name">Sign in with Google</span>
          <span className="persona__description">Opens a Google sign-in window</span>
        </span>
      </button>
    </form>
  );
}
