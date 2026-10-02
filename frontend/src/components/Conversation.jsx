import { useEffect, useRef } from "react";

export default function Conversation({ messages, busy }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  return (
    <section className="conversation" aria-label="Conversation with Scout">
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
      <div ref={endRef} />
    </section>
  );
}
