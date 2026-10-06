import { useCallback, useEffect, useRef, useState } from "react";
import { api, setCredentials } from "./api.js";
import { earcons } from "./audio/earcons.js";
import { useSpeaker, useSpeechInput, words } from "./hooks/useVoice.js";
import SignIn from "./components/SignIn.jsx";
import Header from "./components/Header.jsx";
import Conversation from "./components/Conversation.jsx";
import Composer from "./components/Composer.jsx";
import ProductList from "./components/ProductList.jsx";
import CartPanel from "./components/CartPanel.jsx";

const EMPTY_SCREEN = { products: [], cart: { items: [], total: 0 }, focus_product_id: null };
// What to show, from any API response that carries it (session start, streamed screen, end of turn).
const screenOf = ({ products, cart, focus_product_id }) => ({ products, cart, focus_product_id });
const DEFAULT_SETTINGS = {
  voice: true, rate: 1, language: "en-IN", highContrast: false, largeText: false, headphones: false,
};

// Through speakers the microphone also hears Scout, so interrupting Scout takes a clear phrase
// of the shopper's own words. With headphones there is no echo, so any word interrupts.
const MIN_INTERRUPT_WORDS = 3;

let nextMessageId = 1;
const message = (role, text) => ({ id: nextMessageId++, role, text });

// The conversation loop, hands-free:
//   Scout speaks -> "your turn" chime -> shopper speaks -> soft "thinking" tick -> Scout speaks ...
// The microphone stays open throughout. Speaking over Scout interrupts it (with speakers, it takes
// a short phrase, so Scout's own voice coming back through the microphone never counts as the shopper).
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
  const { speak, startStream, stop: stopSpeaking, isEcho, audibleAt } = speaker;

  // Refs let the always-on microphone callbacks see the latest values.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const busyRef = useRef(false);
  const queuedRef = useRef(null); // what the shopper said while Scout was still finishing a turn
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
      if (!text || !sessionRef.current) return;
      if (busyRef.current) {
        queuedRef.current = text; // Scout is still finishing (e.g. learning in the background): say it next
        return;
      }
      stopSpeaking();
      setPaused(false);
      setMessages((all) => [...all, message("shopper", text)]);
      busyRef.current = true;
      setBusy(true);
      earcons.thinking();

      // Scout's words are shown and spoken while they are still being written.
      const reply = message("scout", "");
      let shown = false;
      let afterSpeaking = yourTurn; // replaced below for "pause" and "end"
      const voice = settingsRef.current.voice
        ? startStream(settingsRef.current.rate, () => afterSpeaking())
        : null;
      const showReply = (fullText) => {
        if (!shown) {
          shown = true;
          setMessages((all) => [...all, { ...reply, text: fullText }]);
        } else {
          setMessages((all) => all.map((m) => (m.id === reply.id ? { ...m, text: fullText } : m)));
        }
      };

      try {
        let soFar = "";
        const turn = await api.chatStream(sessionRef.current, text, (item) => {
          if (item.type === "delta") {
            soFar += item.text;
            showReply(soFar);
            voice?.push(item.text);
          } else if (item.type === "screen" || item.type === "done") {
            setScreen(screenOf(item));
          }
        });
        if (turn.action === "pause") {
          afterSpeaking = () => {};
          setPaused(true); // quiet, but still listening
          earcons.paused();
        } else if (turn.action === "end") {
          afterSpeaking = () => {
            voiceInputRef.current.stop();
            earcons.goodbye();
          };
        }
        setAnnouncement(turn.reply);
        if (voice) voice.end();
        else afterSpeaking();
      } catch (error) {
        stopSpeaking();
        scoutSays(`Sorry, something went wrong on my side. ${error.message || ""} Could you try again?`);
      } finally {
        busyRef.current = false;
        setBusy(false);
        const queued = queuedRef.current;
        queuedRef.current = null;
        if (queued) send(queued);
      }
    },
    [stopSpeaking, startStream, scoutSays, yourTurn]
  );

  // Is this the shopper talking over Scout, rather than Scout's own voice coming back?
  const isInterruption = (text) =>
    !isEcho(text) && (settingsRef.current.headphones || words(text).length >= MIN_INTERRUPT_WORDS);

  // Should a finished phrase be sent? Judged by when it STARTED: while Scout was playing it must be a
  // real interruption; just after Scout stopped it must not be Scout's words; after that it is the shopper.
  const isShopperSpeech = (text, startedAt) => {
    const audible = audibleAt(startedAt);
    if (audible === "playing") return isInterruption(text);
    if (audible === "tail") return !isEcho(text);
    return true;
  };

  const voiceInput = useSpeechInput({
    language: settings.language,
    // Barge-in: the shopper starts talking over Scout, so Scout stops at once.
    onInterim: (text) => {
      if (speaker.speaking && isInterruption(text)) stopSpeaking();
    },
    onFinal: (text, startedAt) => {
      if (!isShopperSpeech(text, startedAt)) return; // Scout's own voice
      send(text);                                     // queued if Scout is still finishing its turn
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
      setScreen(screenOf(started));
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
