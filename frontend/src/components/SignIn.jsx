import { useEffect, useRef, useState } from "react";
import { api, demoCredentials, firebaseCredentials } from "../api.js";
import Icon from "./Icons.jsx";

// The start screen. The backend says how shoppers sign in (/api/auth-config):
//   demo     -> pick one of the demo shoppers
//   firebase -> Sign in with Google
// Either way it hands App a shopper: { name, preferredName, credentials, leave: { label, action }, resumed }.
// After a page refresh the shopper is still signed in (`resumed`), so the start screen is skipped:
//   firebase -> Firebase keeps the Google sign-in itself
//   demo     -> the chosen shopper is in the page address (#shopper=<id>), so a refresh keeps it
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
          <span className="brand__mark" />
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

const SHOPPER_IN_ADDRESS = /^#shopper=(.+)$/;

function DemoShoppers({ personas, onSignedIn }) {
  const pick = (persona, resumed = false) => {
    window.history.replaceState(null, "", `#shopper=${encodeURIComponent(persona.id)}`);
    onSignedIn({
      name: persona.name,
      credentials: demoCredentials(persona.id),
      leave: {
        label: "Switch shopper",
        action: async () => window.history.replaceState(null, "", window.location.pathname + window.location.search),
      },
      resumed,
    });
  };

  // A refresh keeps the shopper chosen before it (once: React may run this effect twice in development).
  const resumedRef = useRef(false);
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.match(SHOPPER_IN_ADDRESS)?.[1] || "");
    const persona = personas.find((p) => p.id === id);
    if (persona && !resumedRef.current) {
      resumedRef.current = true;
      pick(persona, true);
    }
  }, [personas]);

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
            <Icon name="arrow" className="icon persona__arrow" />
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

  const start = (signedInUser, connected = firebase, resumed = false) =>
    onSignedIn({
      name: callMe.trim() || signedInUser.displayName || "",
      preferredName: callMe.trim(),
      credentials: firebaseCredentials(signedInUser),
      leave: { label: "Sign out", action: connected.signOut },
      resumed,
    });

  useEffect(() => {
    let unsubscribe = () => {};
    let cancelled = false;   // React may run this effect twice in development: only the last one listens
    let firstCheck = true;   // Firebase says right away whether this browser is still signed in
    connectFirebase(settings)
      .then((connected) => {
        if (cancelled) return;
        setFirebase(connected);
        unsubscribe = connected.onUserChanged((signedInUser) => {
          setUser(signedInUser);
          if (firstCheck && signedInUser) start(signedInUser, connected, true);   // a refresh: carry on
          firstCheck = false;
        });
      })
      .catch(() => onError("Sign-in isn't available right now. Please try again later."));
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [settings, onError]);

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
