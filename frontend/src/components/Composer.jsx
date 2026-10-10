import { useState } from "react";
import Icon from "./Icons.jsx";
import { Orb, scoutState } from "./ScoutStage.jsx";

// Typing is the alternative to talking (the orb above is the microphone). On a phone the big orb
// scrolls away below the products, so this bar, pinned to the bottom, carries a small orb too.
export default function Composer({ speaking, busy, paused, ready, voiceInput, heard = "", onToggleListening, onSend, onStopSpeaking }) {
  const state = scoutState({ ready, voiceInput, busy, speaking, paused });
  const [text, setText] = useState("");

  const submit = (event) => {
    event.preventDefault();
    if (!text.trim() || busy || !ready) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div className="composer">
      <div className="composer__voice">
        <Orb
          state={state.name}
          active={voiceInput.active}
          ready={ready}
          supported={voiceInput.supported}
          onToggle={onToggleListening}
          small
        />
        <p className="composer__status">{heard ? `“${heard}”` : state.status}</p>
      </div>
      {speaking && (
        <button className="button button--secondary composer__stop" type="button" onClick={onStopSpeaking}>
          <Icon name="volumeOff" />
          Stop speaking
        </button>
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
          placeholder={ready ? "Or type a message…" : "Connecting to Scout…"}
          autoComplete="off"
          disabled={!ready}
        />
        <button className="type__send" type="submit" disabled={!ready || busy || !text.trim()} aria-label="Send">
          <Icon name="send" />
        </button>
      </form>
    </div>
  );
}
