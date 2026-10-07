import { useEffect, useRef, useState } from "react";
import Conversation from "./Conversation.jsx";
import Icon from "./Icons.jsx";

// What Scout is doing right now. Drives the orb's look and the status line.
export function scoutState({ ready, voiceInput, busy, speaking, paused }) {
  if (!ready) return { name: "idle", status: "Connecting to Scout…" };
  if (busy) return { name: "thinking", status: "Thinking…" };
  if (speaking) return { name: "speaking", status: "Scout is speaking" };
  if (voiceInput.blocked) return { name: "idle", status: "Microphone blocked. Allow it in the browser, or type below." };
  if (!voiceInput.active) return { name: "idle", status: "Tap to talk" };
  if (paused) return { name: "listening", status: "Paused. I'm still listening" };
  if (!voiceInput.hearing) return { name: "idle", status: "Starting the microphone…" };
  return { name: "listening", status: "Listening…" };
}

// The orb is Scout: press it to talk or pause. Its look shows what Scout is doing.
export function Orb({ state, active, ready, supported, onToggle, small = false }) {
  return (
    <button
      className={`orb orb--${state} ${small ? "orb--small" : ""}`}
      onClick={onToggle}
      disabled={!ready || !supported}
      aria-pressed={active}
      aria-label={active ? "Pause listening" : "Start talking"}
    >
      <span className="orb__ring" aria-hidden="true" />
      <span className="orb__ring orb__ring--late" aria-hidden="true" />
      <span className="orb__core" aria-hidden="true">
        {state === "speaking" ? (
          <span className="orb__bars">
            <span />
            <span />
            <span />
            <span />
            <span />
          </span>
        ) : (
          <Icon name="mic" />
        )}
      </span>
    </button>
  );
}

// `starters`: the shop's categories, so a new shopper can start with one press (they can also just talk).
export default function ScoutStage({ messages, busy, speaking, paused, ready, voiceInput, empty, starters = [], onToggleListening, onAsk }) {
  const state = scoutState({ ready, voiceInput, busy, speaking, paused });
  const [historyOpen, setHistoryOpen] = useState(false);

  const last = messages[messages.length - 1];
  const lastScout = [...messages].reverse().find((m) => m.role === "scout");
  const lastShopper = [...messages].reverse().find((m) => m.role === "shopper");
  const waiting = busy && last?.role === "shopper";        // Scout hasn't started its reply yet
  const youSaid = voiceInput.interim || lastShopper?.text;

  // Keep the newest words of a long reply in view while it streams in.
  const captionRef = useRef(null);
  useEffect(() => {
    const box = captionRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [lastScout?.text, waiting]);

  return (
    <div className={`stage ${empty ? "stage--hero" : ""}`}>
      <div className="stage__top">
        <Orb
          state={state.name}
          active={voiceInput.active}
          ready={ready}
          supported={voiceInput.supported}
          onToggle={onToggleListening}
        />
        <p className="stage__status">{state.status}</p>
        {!voiceInput.supported && (
          <p className="notice">Voice conversation needs Chrome, Edge or Safari. You can type below.</p>
        )}
      </div>

      <div className="stage__caption" ref={captionRef}>
        {youSaid && (
          <p className={`stage__you ${voiceInput.interim ? "is-live" : ""}`}>
            <span className="visually-hidden">You said: </span>“{youSaid}”
          </p>
        )}
        {waiting ? (
          <p className="stage__scout" aria-hidden="true">
            <span className="dots">
              <span />
              <span />
              <span />
            </span>
          </p>
        ) : (
          lastScout && <p className="stage__scout">{lastScout.text}</p>
        )}

        {empty && starters.length > 0 && (
          <ul className="chips stage__starters" role="list" aria-label="Try saying">
            {starters.map((starter) => (
              <li key={starter}>
                <button className="chip" disabled={busy || !ready} onClick={() => onAsk(starter)}>
                  {starter}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {messages.length > 1 && (
        <div className="history">
          <button className="history__toggle" aria-expanded={historyOpen} onClick={() => setHistoryOpen((open) => !open)}>
            {historyOpen ? "Hide conversation" : `Show conversation (${messages.length})`}
          </button>
          {historyOpen && <Conversation messages={messages} busy={busy} />}
        </div>
      )}
    </div>
  );
}
