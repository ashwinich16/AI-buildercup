import { useEffect, useRef, useState } from "react";
import Conversation from "./Conversation.jsx";

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

// Scout's face: an original little character. The orb's state (idle, listening, thinking, speaking) sets how
// it looks, all in CSS: it blinks, looks up while thinking, and its mouth moves while it speaks.
function ScoutFace() {
  return (
    <svg className="face" viewBox="0 0 100 100" focusable="false">
      <g className="face__eyes">
        <ellipse className="face__eye" cx="36" cy="44" rx="7" ry="9" />
        <ellipse className="face__eye" cx="64" cy="44" rx="7" ry="9" />
      </g>
      <ellipse className="face__cheek" cx="25" cy="60" rx="6" ry="3.5" />
      <ellipse className="face__cheek" cx="75" cy="60" rx="6" ry="3.5" />
      <path className="face__smile" d="M39 63 Q50 73 61 63" />
      <ellipse className="face__talk" cx="50" cy="66" rx="8" ry="6" />
    </svg>
  );
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
        <ScoutFace />
      </span>
    </button>
  );
}

// `starters`: the shop's categories, so a new shopper can start with one press (they can also just talk).
// `choices`: short answers to Scout's question; tap one or more, then send (or just say them).
export default function ScoutStage({
  messages, busy, speaking, paused, ready, voiceInput, empty, starters = [], choices = [], onToggleListening, onAsk,
}) {
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
        {choices.length > 0 && <Choices key={choices.join("|")} choices={choices} busy={busy} onAsk={onAsk} />}

        {empty && (
          <div className="stage__ways" role="group" aria-label="How would you like to shop?">
            <button className="button" disabled={busy || !ready} onClick={() => onAsk("I'd like to plan my outfit")}>
              Plan my outfit
            </button>
            <button
              className="button button--secondary"
              disabled={busy || !ready}
              onClick={() => onAsk("I'm shopping for something in particular")}
            >
              Regular shopping
            </button>
          </div>
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

function Choices({ choices, busy, onAsk }) {
  const [picked, setPicked] = useState([]);
  const toggle = (choice) =>
    setPicked((all) => (all.includes(choice) ? all.filter((c) => c !== choice) : [...all, choice]));

  return (
    <div className="stage__choices" role="group" aria-label="Quick answers">
      <ul className="chips" role="list">
        {choices.map((choice) => (
          <li key={choice}>
            <button className="chip" aria-pressed={picked.includes(choice)} disabled={busy} onClick={() => toggle(choice)}>
              {choice}
            </button>
          </li>
        ))}
      </ul>
      {picked.length > 0 && (
        <button className="button button--small" disabled={busy} onClick={() => onAsk(picked.join(", "))}>
          Send
        </button>
      )}
    </div>
  );
}
