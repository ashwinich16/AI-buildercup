import { useEffect, useRef } from "react";

export default function Conversation({ messages, busy }) {
  const boxRef = useRef(null);

  // Keep the newest message in view by scrolling this box only (scrollIntoView would scroll the page too).
  useEffect(() => {
    const box = boxRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [messages, busy]);

  return (
    <section className="conversation" aria-label="Conversation with Scout" ref={boxRef}>
      <ol className="messages" role="list">
        {messages.map((message) => (
          <li key={message.id} className={`message message--${message.role}`}>
            <span className="message__who">{message.role === "scout" ? "Scout" : "You"}</span>
            <p className="message__text">{message.text}</p>
          </li>
        ))}
        {busy && (
          <li className="message message--scout message--thinking" role="status">
            <span className="message__who">Scout</span>
            <p className="message__text">
              <span className="dots" aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
              <span className="visually-hidden">Scout is thinking</span>
            </p>
          </li>
        )}
      </ol>
    </section>
  );
}
