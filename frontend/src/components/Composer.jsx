import { useRef, useState } from "react";
import Icon from "./Icons.jsx";
import { Orb, scoutState } from "./ScoutStage.jsx";

// Typing is the alternative to talking (the orb above is the microphone). On a phone the big orb
// scrolls away below the products, so this bar, pinned to the bottom, carries a small orb too.
// Photos of their clothes can be attached: each is read the moment it is attached, while they type.
export default function Composer({
  speaking, busy, paused, ready, voiceInput, heard = "", attachments, onAttach, onToggleListening, onSend, onStopSpeaking,
}) {
  const state = scoutState({ ready, voiceInput, busy, speaking, paused });
  const [text, setText] = useState("");
  const fileRef = useRef(null);
  const canSend = ready && !busy && (text.trim() || attachments.length > 0);

  const submit = (event) => {
    event.preventDefault();
    if (!canSend) return;
    onSend(text.trim() || "Here are some clothes I own.");
    setText("");
  };

  const attach = (event) => {
    const files = [...(event.target.files || [])];
    event.target.value = "";                     // choosing the same file again still counts
    if (files.length) onAttach(files);
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
      {attachments.length > 0 && (
        <ul className="attachments" role="list" aria-label="Attached photos">
          {attachments.map((photo) => (
            <li key={photo.id} className={`attachment is-${photo.status}`} title={photo.note}>
              <img src={photo.preview} alt="" />
              <span className="attachment__note">{photo.note}</span>
            </li>
          ))}
        </ul>
      )}
      <form className="type" onSubmit={submit}>
        <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={attach} />
        <button
          className="type__attach"
          type="button"
          disabled={!ready}
          onClick={() => fileRef.current?.click()}
          aria-label="Attach photos of your clothes"
        >
          <Icon name="camera" />
        </button>
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
        <button className="type__send" type="submit" disabled={!canSend} aria-label="Send">
          <Icon name="send" />
        </button>
      </form>
    </div>
  );
}
