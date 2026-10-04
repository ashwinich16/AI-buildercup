import { useState } from "react";

// What the conversation is doing right now, in plain words.
function statusText({ voiceInput, speaking, busy, paused }) {
  if (voiceInput.blocked) return "Microphone access is blocked. Allow it in the browser to talk, or type below.";
  if (!voiceInput.active) return "Not listening. Press Start talking when you're ready.";
  if (busy) return "Thinking…";
  if (speaking) return "Scout is speaking. Talk over it to interrupt, or press Escape.";
  if (paused) return "Paused. I'm still listening whenever you're ready.";
  if (!voiceInput.hearing) return "Starting the microphone…";
  return "Listening. Go ahead.";
}

export default function Composer({ voiceInput, speaking, busy, paused, ready, onToggleListening, onSend, onStopSpeaking }) {
  const [text, setText] = useState("");

  const submit = (event) => {
    event.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setText("");
  };

  const listeningNow = voiceInput.active && voiceInput.hearing && !speaking && !busy;

  return (
    <div className="composer">
      {voiceInput.supported ? (
        <div className="talk">
          <button
            className={`talk__button ${voiceInput.active ? "is-active" : ""} ${listeningNow ? "is-listening" : ""}`}
            onClick={onToggleListening}
            aria-pressed={voiceInput.active}
            disabled={!ready}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true" className="talk__icon">
              <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.08A7 7 0 0 0 19 12h-2Z" />
            </svg>
            <span className="talk__label">{voiceInput.active ? "Pause listening" : "Start talking"}</span>
          </button>
          <p className="talk__status">{statusText({ voiceInput, speaking, busy, paused })}</p>
          <p className="talk__interim">{voiceInput.interim}</p>
        </div>
      ) : (
        <p className="notice">Voice conversation needs Chrome, Edge or Safari. You can type below.</p>
      )}

      <form className="type" onSubmit={submit}>
        <label htmlFor="message" className="visually-hidden">
          Type a message to Scout
        </label>
        <input
          id="message"
          className="type__input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Or type here…"
          autoComplete="off"
        />
        <button className="button" type="submit" disabled={busy || !text.trim()}>
          Send
        </button>
        {speaking && (
          <button className="button button--quiet" type="button" onClick={onStopSpeaking}>
            Stop speaking
          </button>
        )}
      </form>
    </div>
  );
}
