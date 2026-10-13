import { useCallback, useEffect, useRef, useState } from "react";
import { api, setCredentials } from "./api.js";
import { earcons } from "./audio/earcons.js";
import {
  END_OF_SENTENCE_MS, END_OF_SENTENCE_SLOW_MS, stripCues, useSpeaker, useSpeechInput, words,
} from "./hooks/useVoice.js";
import SignIn from "./components/SignIn.jsx";
import Header from "./components/Header.jsx";
import ScoutStage from "./components/ScoutStage.jsx";
import Composer from "./components/Composer.jsx";
import ProductList from "./components/ProductList.jsx";
import Spotlight from "./components/Spotlight.jsx";
import SuggestionsPopup from "./components/SuggestionsPopup.jsx";
import CapsuleBoard from "./components/CapsuleBoard.jsx";
import WardrobeSheet from "./components/WardrobeSheet.jsx";
import TryOnSheet from "./components/TryOnSheet.jsx";
import CartSheet from "./components/CartSheet.jsx";
import WishlistSheet from "./components/WishlistSheet.jsx";
import SettingsSheet from "./components/SettingsSheet.jsx";

const EMPTY_SCREEN = {
  products: [], cart: { items: [], total: 0 }, wishlist: { items: [] }, focus_product_id: null, suggestion: null, capsule: null,
  planning_outfit: false, wardrobe: [], suggestions: [], spotlight: null,
};
// What to show, from any API response that carries it (session start, streamed screen, end of turn).
// The wardrobe is left out of mid-turn updates (it carries photos), so the last one shown is kept then.
const suggestionsKeyOf = (suggestions) => (suggestions || []).map((item) => item.id).join(",");
const screenOf = (
  { products, cart, wishlist, focus_product_id, suggestion, suggestions, capsule, planning_outfit, wardrobe, spotlight },
  current = EMPTY_SCREEN,
) => ({
  products, cart, wishlist: wishlist ?? current.wishlist, focus_product_id, suggestion, capsule, planning_outfit, wardrobe: wardrobe ?? current.wardrobe,
  suggestions: suggestions || [], spotlight,
});
const DEFAULT_SETTINGS = {
  voice: true, rate: 1, language: "en-IN", highContrast: false, largeText: false, headphones: false, moreTime: false,
};

// The Scout panel beside the products can be dragged wider or narrower, within these limits (pixels).
const DEFAULT_ASSISTANT_WIDTH = 400;
const MIN_ASSISTANT_WIDTH = 320;
const MAX_ASSISTANT_SHARE = 0.6;   // never more than this share of the window: the products need room too

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
  const [openSheet, setOpenSheet] = useState(null); // "cart", "wishlist", "wardrobe", "settings" or null
  const [spokenOption, setSpokenOption] = useState(null); // position of the option Scout is describing right now
  const [assistantWidth, setAssistantWidth] = useState(DEFAULT_ASSISTANT_WIDTH); // the Scout panel, in pixels
  const [categories, setCategories] = useState([]); // what the shop sells, offered as first things to ask for
  const [tryOnAdjustable, setTryOnAdjustable] = useState(false); // the try-on follows "tuck the shirt in"
  const [tryOn, setTryOn] = useState(null); // what to try on: { productIds, wardrobeIds, label }, or null
  const [photo, setPhoto] = useState(null); // the shopper's photo for try-on, kept only while the page is open
  const [attachments, setAttachments] = useState([]); // clothes photos attached in the chat: { id, preview, status, note }
  const [choices, setChoices] = useState([]); // short answers to Scout's last question, to tap
  const [chooseOne, setChooseOne] = useState(false); // one tap answers it (a size)
  const [resumedChat, setResumedChat] = useState(false); // carrying on the last conversation: show what was said
  // The "You might also like" pop-up opens once for each new set of suggestions (e.g. after adding to cart).
  const [dismissedSuggestions, setDismissedSuggestions] = useState("");

  const speaker = useSpeaker();
  const { speak, startStream, stop: stopSpeaking, isEcho, audibleAt } = speaker;

  // Refs let the always-on microphone callbacks see the latest values.
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const busyRef = useRef(false);
  const queuedRef = useRef(null); // what the shopper said while Scout was still finishing a turn
  const sessionRef = useRef(null);
  const listeningRef = useRef(false);
  const photosRef = useRef({ reading: [], count: 0 }); // attached photos still being read, and how many are attached

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
      text = stripCues(text);
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
        queuedRef.current = `${queuedRef.current || ""} ${text}`.trim(); // Scout is still finishing: say it next
        return;
      }
      stopSpeaking();
      setSpokenOption(null);
      setPaused(false);
      setChoices([]);
      const photos = photosRef.current;
      photosRef.current = { reading: [], count: 0 };
      const said = photos.count ? `${text} (${photos.count} ${photos.count === 1 ? "photo" : "photos"})` : text;
      setMessages((all) => [...all, message("shopper", said)]);
      busyRef.current = true;
      setBusy(true);
      earcons.thinking();
      await Promise.allSettled(photos.reading); // photos are read as they are attached; usually already done
      setAttachments([]);

      // Scout's words are shown and spoken while they are still being written.
      const reply = message("scout", "");
      let shown = false;
      let afterSpeaking = yourTurn; // replaced below for "pause" and "end"
      // While Scout describes an option, that card lights up ([[n]] cues in its words).
      const voice = settingsRef.current.voice
        ? startStream(
            settingsRef.current.rate,
            () => {
              setSpokenOption(null);
              afterSpeaking();
            },
            setSpokenOption
          )
        : null;
      const showReply = (rawText) => {
        const fullText = stripCues(rawText);
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
            if (voice) voice.push(item.text);
            else {
              const cue = item.text.match(/\[\[(\d+)\]\]/);   // no voice: follow the text as it appears
              if (cue) setSpokenOption(Number(cue[1]));
            }
          } else if (item.type === "screen" || item.type === "done") {
            setScreen((current) => screenOf(item, current));
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
        setChoices(turn.choices || []);
        setChooseOne(Boolean(turn.choose_one));
        setAnnouncement(stripCues(turn.reply));
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

  // Photos of their clothes, attached in the chat: each is read straight away (while they type), so Scout
  // already knows the pieces when the message is sent.
  const attachPhotos = (files) => {
    for (const file of files) {
      const id = `${Date.now()}-${Math.random()}`;
      const update = (change) => setAttachments((all) => all.map((p) => (p.id === id ? { ...p, ...change } : p)));
      setAttachments((all) => [...all, { id, preview: URL.createObjectURL(file), status: "reading", note: "Reading…" }]);
      const reading = api
        .addToWardrobe(sessionRef.current, file)
        .then(({ wardrobe, added }) => {
          update({ status: "done", note: added.map((item) => item.description).join(", ") });
          setScreen((current) => ({ ...current, wardrobe }));
        })
        .catch((error) => update({ status: "failed", note: error.message }));
      photosRef.current = { reading: [...photosRef.current.reading, reading], count: photosRef.current.count + 1 };
    }
  };

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
    pauseMs: settings.moreTime ? END_OF_SENTENCE_SLOW_MS : END_OF_SENTENCE_MS,
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

  // What to show as "you said" while listening: the same judgement as for sending, so Scout's own voice
  // coming back through the speakers is never displayed as the shopper's words.
  const heard = voiceInput.interim;
  const audibleNow = speaker.speaking ? "playing" : audibleAt(Date.now());
  const shownHeard =
    !heard ? "" : audibleNow === "playing" ? (isInterruption(heard) ? heard : "") : audibleNow === "tail" && isEcho(heard) ? "" : heard;
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
      setDismissedSuggestions(suggestionsKeyOf(started.suggestions)); // pop up only after adding, not on resume
      setCategories(started.categories || []);
      setTryOnAdjustable(Boolean(started.try_on_adjustable));
      // Carrying on their last conversation: what was said is shown again, as they left it.
      setMessages((started.history || []).map((m) => message(m.role, m.text)));
      setResumedChat(Boolean(started.resumed));
      if (signedIn.resumed) {
        // After a refresh there has been no click yet, and browsers only allow sound and the microphone
        // after one: show the greeting, and the orb ("Tap to talk") starts the conversation.
        setMessages((all) => [...all, message("scout", started.greeting)]);
      } else {
        voiceInput.start(); // signing in is the click browsers need before using the microphone
        scoutSays(started.greeting);
      }
    } catch (error) {
      scoutSays(`I couldn't start our conversation: ${error.message} Refreshing the page tries again.`);
    }
  };

  // Clear the conversation and start a fresh one. What Scout remembers (name, preferences, cart, wardrobe) stays.
  const clearConversation = async () => {
    if (busyRef.current) return;                 // let Scout finish the turn it is on first
    stopSpeaking();
    setSpokenOption(null);
    setChoices([]);
    setAttachments([]);
    photosRef.current = { reading: [], count: 0 };
    queuedRef.current = null;
    const previous = sessionRef.current;
    sessionRef.current = null;
    setSessionId(null);
    setMessages([]);
    setScreen(EMPTY_SCREEN);
    if (previous) api.endSession(previous).catch(() => {});   // the server forgets the old one
    try {
      setResumedChat(false);
      const started = await api.startSession("", true);   // a fresh conversation
      sessionRef.current = started.session_id;
      setSessionId(started.session_id);
      setScreen(screenOf(started));
      setDismissedSuggestions(suggestionsKeyOf(started.suggestions)); // pop up only after adding, not on resume
      scoutSays(started.greeting);
    } catch (error) {
      scoutSays(`I couldn't start a new conversation: ${error.message}`);
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
    const onKey = (event) => {
      if (event.key !== "Escape") return;
      stopSpeaking();
      setSpokenOption(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stopSpeaking]);

  if (!shopper) return <SignIn onSignedIn={startShopping} />;

  const suggestionsKey = suggestionsKeyOf(screen.suggestions);
  // 1-based number of the suggestion being described, or 0 when the spotlight is something else
  const spotlightSuggestion = screen.spotlight
    ? (screen.suggestions || []).findIndex((item) => item.id === screen.spotlight.id) + 1
    : 0;
  const hasProducts = screen.products.length > 0 || Boolean(screen.capsule) || Boolean(screen.spotlight);   // something to show beside Scout

  return (
    <div className="app">
      <a className="skip-link" href="#talk">
        Skip to talking with Scout
      </a>
      <Header
        shopper={shopper}
        cartCount={screen.cart.items?.length || 0}
        onOpenCart={() => setOpenSheet("cart")}
        wishlistCount={screen.wishlist?.items?.length || 0}
        onOpenWishlist={() => setOpenSheet("wishlist")}
        onOpenWardrobe={() => setOpenSheet("wardrobe")}
        onClearChat={clearConversation}
        onOpenSettings={() => setOpenSheet("settings")}
        onLeave={leave}
      />

      {/* Screen readers hear Scout's replies here, unless Scout is already speaking them aloud. */}
      <div className="visually-hidden" aria-live={settings.voice ? "off" : "polite"}>
        {announcement}
      </div>

      {/* Until there are products, Scout fills the page; then the products take the main area. */}
      <main
        className={`layout ${hasProducts ? "" : "layout--talk"}`}
        style={{ "--assistant-width": `${assistantWidth}px` }}
      >
        {hasProducts && (
          <section className="layout__shop" aria-label="Options">
            <Spotlight
              product={screen.spotlight}
              speaking={speaker.speaking}
              busy={busy}
              addPhrase={spotlightSuggestion > 0 ? `Add suggestion ${spotlightSuggestion} to my cart` : null}
              onAsk={send}
            />
            <CapsuleBoard
              capsule={screen.capsule}
              busy={busy}
              onAsk={send}
              onTryOn={setTryOn}
              onOpenWardrobe={() => setOpenSheet("wardrobe")}
            />
            <ProductList
              products={screen.products}
              focusId={screen.focus_product_id}
              spokenPosition={spokenOption}
              busy={busy || !sessionId}
              choosingFor={screen.capsule?.changing}
              savedIds={(screen.wishlist?.items || []).map((item) => item.product_id)}
              onAsk={send}
              onTryOn={setTryOn}
            />
          </section>
        )}
        {hasProducts && <PanelResizer width={assistantWidth} onResize={setAssistantWidth} />}

        <aside className="layout__assistant" id="talk" aria-label="Talk with Scout">
          <ScoutStage
            messages={messages}
            busy={busy}
            speaking={speaker.speaking}
            paused={paused}
            ready={Boolean(sessionId)}
            voiceInput={voiceInput}
            heard={shownHeard}
            empty={!hasProducts}
            starters={categories}
            choices={busy ? [] : choices}
            chooseOne={chooseOne}
            showHistory={resumedChat}
            planningOutfit={screen.planning_outfit}
            onToggleListening={toggleListening}
            onAsk={send}
          />
          <Composer
            speaking={speaker.speaking}
            busy={busy}
            paused={paused}
            ready={Boolean(sessionId)}
            voiceInput={voiceInput}
            heard={shownHeard}
            attachments={attachments}
            onAttach={attachPhotos}
            onToggleListening={toggleListening}
            onSend={send}
            onStopSpeaking={() => {
              stopSpeaking();
              setSpokenOption(null);
            }}
          />
        </aside>
      </main>

      <SuggestionsPopup
        suggestions={screen.suggestions}
        open={suggestionsKey !== "" && suggestionsKey !== dismissedSuggestions && openSheet === null}
        onClose={() => setDismissedSuggestions(suggestionsKey)}
        busy={busy}
        onAsk={send}
      />
      <CartSheet
        open={openSheet === "cart"}
        onClose={() => setOpenSheet(null)}
        cart={screen.cart}
        suggestion={screen.suggestion}
        busy={busy}
        onAsk={send}
      />
      <WishlistSheet
        open={openSheet === "wishlist"}
        onClose={() => setOpenSheet(null)}
        wishlist={screen.wishlist}
        busy={busy}
        onAsk={send}
      />
      <WardrobeSheet
        open={openSheet === "wardrobe"}
        onClose={() => setOpenSheet(null)}
        sessionId={sessionId}
        wardrobe={screen.wardrobe || []}
        onWardrobe={(wardrobe) => setScreen((current) => ({ ...current, wardrobe }))}
      />
      <TryOnSheet
        adjustable={tryOnAdjustable}
        request={tryOn}
        sessionId={sessionId}
        photo={photo}
        onPhoto={setPhoto}
        onClose={() => setTryOn(null)}
      />
      <SettingsSheet
        open={openSheet === "settings"}
        onClose={() => setOpenSheet(null)}
        settings={settings}
        onSettingsChange={setSettings}
      />
    </div>
  );
}

// The bar between the products and Scout: drag it, or focus it and use the arrow keys, to resize Scout's panel.
function PanelResizer({ width, onResize }) {
  const clamp = (value) =>
    Math.round(Math.min(Math.max(value, MIN_ASSISTANT_WIDTH), window.innerWidth * MAX_ASSISTANT_SHARE));

  const startDrag = (event) => {
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const drag = (event) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) onResize(clamp(window.innerWidth - event.clientX));
  };
  const onKey = (event) => {
    const step = { ArrowLeft: 32, ArrowRight: -32 }[event.key];   // left makes Scout's panel wider
    if (!step) return;
    event.preventDefault();
    onResize(clamp(width + step));
  };

  return (
    <div
      className="resizer"
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize the conversation panel"
      aria-valuenow={width}
      aria-valuemin={MIN_ASSISTANT_WIDTH}
      tabIndex={0}
      onPointerDown={startDrag}
      onPointerMove={drag}
      onKeyDown={onKey}
      onDoubleClick={() => onResize(DEFAULT_ASSISTANT_WIDTH)}
    />
  );
}
