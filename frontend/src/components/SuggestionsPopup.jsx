import { useEffect, useId, useRef } from "react";
import { formatPrice } from "../format.js";
import Icon from "./Icons.jsx";

// "You might also like": a few items that go with what the shopper just added, shown as a pop-up so
// they are seen at once (not at the bottom of the page). Built on the native <dialog>: Escape or the
// close button dismisses it, and focus returns where it was. Scout names the first one out loud.
export default function SuggestionsPopup({ suggestions, open, onClose, busy, onAsk }) {
  const ref = useRef(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  const act = (text) => {
    onClose();
    onAsk(text);
  };
  const goesWith = suggestions[0]?.goes_with;

  return (
    <dialog
      ref={ref}
      className="popup"
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}   // a click on the dimmed backdrop
    >
      <div className="popup__head">
        <div>
          <p className="popup__eyebrow">{goesWith ? `Goes well with your ${goesWith}` : "Goes well with your cart"}</p>
          <h2 id={titleId} className="popup__title">
            You might also like
          </h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="Close suggestions">
          <Icon name="close" />
        </button>
      </div>

      <ul className="popup__grid" role="list">
        {suggestions.map((item, index) => {
          const reason = item.reasons?.[0];
          return (
            <li key={item.id} className="popup__item">
              <div className="popup__image">
                {item.image_url ? <img src={item.image_url} alt="" loading="lazy" /> : null}
              </div>
              <p className="popup__name">{item.name}</p>
              <p className="popup__meta">
                <span className="popup__price">{formatPrice(item.price)}</span>
                {reason && <span className="muted"> · {reason.charAt(0).toUpperCase() + reason.slice(1)}</span>}
              </p>
              <div className="popup__actions">
                <button
                  className="button button--secondary"
                  disabled={busy}
                  onClick={() => act(`Tell me about suggestion ${index + 1}`)}
                  aria-label={`Details about ${item.name}`}
                >
                  Details
                </button>
                <button
                  className="button"
                  disabled={busy}
                  onClick={() => act(`Add suggestion ${index + 1} to my cart`)}
                  aria-label={`Add ${item.name} to cart`}
                >
                  Add
                </button>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="popup__foot">
        <button className="button button--ghost" onClick={onClose}>
          Not now
        </button>
      </div>
    </dialog>
  );
}
