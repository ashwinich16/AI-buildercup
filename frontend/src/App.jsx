import { useCallback, useEffect, useRef, useState } from "react";
import { api, setCredentials } from "./api.js";
import { earcons } from "./audio/earcons.js";
import { useSpeaker, useSpeechInput } from "./hooks/useVoice.js";
import SignIn from "./components/SignIn.jsx";
import Header from "./components/Header.jsx";
import Conversation from "./components/Conversation.jsx";
import Composer from "./components/Composer.jsx";
import ProductList from "./components/ProductList.jsx";
import CartPanel from "./components/CartPanel.jsx";

const EMPTY_SCREEN = { products: [], cart: { items: [], total: 0 }, focus_product_id: null };
const DEFAULT_SETTINGS = { voice: true, rate: 1, language: "en-IN", highContrast: false, largeText: false };

let nextMessageId = 1;
const message = (role, text) => ({ id: nextMessageId++, role, text });

// The conversation loop, hands-free:
//   Scout speaks -> "your turn" chime -> shopper speaks -> soft "thinking" tick -> Scout speaks ...
// The microphone stays open throughout. Speaking over Scout interrupts it immediately.
export default function App() {
  const [shopper, setShopper] = useState(null); // { name, credentials, leave }
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [screen, setScreen] = useState(EMPTY_SCREEN);
  const [busy, setBusy] = useState(false);
  const [paused, setPaused] = useState(false);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [announcement, setAnnouncement] = useState("");

  const speaker = useSpeaker();
  const { speak, stop: stopSpeaking, isEcho } = speaker;

  // Refs let the always-on microphone callbacks see the latest values.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const busyRef = useRef(false);
  const sessionRef = useRef(null);
  const listeningRef = useRef(false);

  useEffect(() => {
    document.documentElement.dataset.contrast = settings.highContrast ? "high" : "normal";
    document.documentElement.dataset.text = settings.largeText ? "large" : "normal";
  }, [settings.highContrast, settings.largeText]);

  const yourTurn = useCallback(() => {
    if (listeningRef.current) earcons.yourTurn();
  }, []);

  // Show and say something; when finished, signal that it's the shopper's turn.
  const scoutSays = useCallback(
    (text, afterSpeaking = yourTurn) => {
      setMessages((all) => [...all, message("scout", text)]);
      setAnnouncement(text);
      if (settingsRef.current.voice) speak(text, settingsRef.current.rate, afterSpeaking);
      else afterSpeaking();
    },
    [speak, yourTurn]
  );

  const send = useCallback(
    async (text) => {
      if (!text || !sessionRef.current || busyRef.current) return;
      stopSpeaking();
      setPaused(false);
      setMessages((all) => [...all, message("shopper", text)]);
      busyRef.current = true;
      setBusy(true);
      earcons.thinking();
      try {
        const turn = await api.chat(sessionRef.current, text);
        setScreen({ products: turn.products, cart: turn.cart, focus_product_id: turn.focus_product_id });
        if (turn.action === "pause") {
          setPaused(true); // quiet, but still listening
          earcons.paused();
        } else if (turn.action === "end") {
          scoutSays(turn.reply, () => {
            voiceInputRef.current.stop();
            earcons.goodbye();
          });
        } else {
          scoutSays(turn.reply);
        }
      } catch (error) {
        scoutSays(`Sorry, something went wrong on my side. ${error.message || ""} Could you try again?`);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [stopSpeaking, scoutSays]
  );

  const voiceInput = useSpeechInput({
    language: settings.language,
    // Barge-in: the shopper starts talking over Scout, so Scout stops at once.
    onInterim: (text) => {
      if (speaker.speaking && !isEcho(text)) stopSpeaking();
    },
    onFinal: (text) => {
      if (isEcho(text) || busyRef.current) return; // Scout's own voice, or still answering
      send(text);
    },
  });
  const voiceInputRef = useRef(voiceInput);
  voiceInputRef.current = voiceInput;
  listeningRef.current = voiceInput.active;

  const startShopping = async (signedIn) => {
    earcons.unlock(); // signing in is the click that allows the chimes to play
    setCredentials(signedIn.credentials);
    setShopper(signedIn);
    setMessages([]);
    setScreen(EMPTY_SCREEN);
    setPaused(false);
    try {
      const started = await api.startSession(signedIn.preferredName);
      sessionRef.current = started.session_id;
      setShopper({ ...signedIn, name: started.shopper_name || signedIn.name });
      setSessionId(started.session_id);
      setScreen({ products: started.products, cart: started.cart, focus_product_id: started.focus_product_id });
      voiceInput.start(); // signing in is the click browsers need before using the microphone
      scoutSays(started.greeting);
    } catch (error) {
      scoutSays(`I couldn't start our conversation: ${error.message}`);
    }
  };

  const leave = async () => {
    stopSpeaking();
    voiceInput.stop();
    await shopper.leave.action(); // "Sign out" with Google; nothing to do for a demo shopper
    setCredentials(null);
    sessionRef.current = null;
    setShopper(null);
    setSessionId(null);
  };

  const toggleListening = () => {
    earcons.unlock();
    if (voiceInput.active) {
      voiceInput.stop();
      stopSpeaking();
      earcons.paused();
    } else {
      voiceInput.start();
      setPaused(false);
      earcons.yourTurn();
    }
  };

  // Escape always silences Scout, wherever focus is.
  useEffect(() => {
    const onKey = (event) => event.key === "Escape" && stopSpeaking();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stopSpeaking]);

  if (!shopper) return <SignIn onSignedIn={startShopping} />;

  return (
    <div className="app">
      <a className="skip-link" href="#talk">
        Skip to talking with Scout
      </a>
      <Header shopper={shopper} settings={settings} onSettingsChange={setSettings} onLeave={leave} />

      {/* Screen readers hear Scout's replies here, unless Scout is already speaking them aloud. */}
      <div className="visually-hidden" aria-live={settings.voice ? "off" : "polite"}>
        {announcement}
      </div>

      <main className="layout">
        <div className="layout__chat" id="talk">
          <Conversation messages={messages} busy={busy} />
          <Composer
            voiceInput={voiceInput}
            speaking={speaker.speaking}
            busy={busy}
            paused={paused}
            ready={Boolean(sessionId)}
            onToggleListening={toggleListening}
            onSend={send}
            onStopSpeaking={stopSpeaking}
          />
        </div>
        <aside className="layout__side" aria-label="Options and cart">
          <ProductList products={screen.products} focusId={screen.focus_product_id} busy={busy} onAsk={send} />
          <CartPanel cart={screen.cart} busy={busy} onAsk={send} />
        </aside>
      </main>
    </div>
  );
}
