import { useRef, useState } from "react";
import { api } from "../api.js";
import Sheet from "./Sheet.jsx";

// The clothes the shopper already owns, so Scout can plan looks around them. A piece is added from a
// photo (Scout recognises it) or simply by telling Scout about it in the conversation.
export default function WardrobeSheet({ open, onClose, sessionId, wardrobe, onWardrobe }) {
  const [status, setStatus] = useState("");
  const [adding, setAdding] = useState(false);
  const fileRef = useRef(null);

  const addPhotos = async (event) => {
    const files = [...(event.target.files || [])];
    event.target.value = "";                     // choosing the same file again still counts
    setAdding(true);
    for (const [index, file] of files.entries()) {
      setStatus(`Looking at photo ${index + 1} of ${files.length}…`);
      try {
        onWardrobe((await api.addToWardrobe(sessionId, file)).wardrobe);
      } catch (error) {
        setStatus(error.message);
        setAdding(false);
        return;
      }
    }
    setStatus(files.length ? "Added. Scout will use it when planning your looks." : "");
    setAdding(false);
  };

  const remove = async (item) => {
    try {
      onWardrobe(await api.removeFromWardrobe(sessionId, item.id));
      setStatus(`Removed the ${item.description}.`);
    } catch (error) {
      setStatus(error.message);
    }
  };

  return (
    <Sheet open={open} onClose={onClose} title="Your wardrobe">
      <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={addPhotos} />
      <div className="wardrobe">
        <p className="muted">
          Add photos of clothes you own (one piece or several per photo), attach them in the chat, or just tell
          Scout about them. Scout plans your looks
          around them and suggests only what's missing.
        </p>
        <button className="button" disabled={adding || !sessionId} onClick={() => fileRef.current?.click()}>
          {adding ? "Adding…" : "Add photos"}
        </button>
        <p role="status" className="muted">
          {status}
        </p>
        {wardrobe.length === 0 ? (
          <p className="muted">Nothing here yet.</p>
        ) : (
          <ul className="wardrobe__items" role="list">
            {wardrobe.map((item) => (
              <li key={item.id} className="wardrobe__item">
                <div className="capsule__image">
                  {item.thumbnail ? (
                    <img src={item.thumbnail} alt="" />
                  ) : (
                    <span className="capsule__placeholder muted">
                      {item.category}
                      <small>No photo</small>
                    </span>
                  )}
                </div>
                <p className="capsule__name">{item.description}</p>
                <button
                  className="button button--ghost button--small"
                  onClick={() => remove(item)}
                  aria-label={`Remove ${item.description}`}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Sheet>
  );
}
