import { useEffect, useState } from "react";
import { formatPrice } from "../format.js";
import Icon from "./Icons.jsx";
import Sheet from "./Sheet.jsx";

// Products the shopper saved to come back to. Tick two or more to compare them, show them all on screen (to
// hear about one, try it on or add it to the cart), or take one off. Every button goes through the
// conversation, like anything the shopper says, so Scout answers out loud.
export default function WishlistSheet({ open, onClose, wishlist, busy, onAsk }) {
  const items = wishlist?.items || [];
  const [picked, setPicked] = useState([]);       // positions (1 = the first) ticked to compare

  useEffect(() => {
    if (!open) setPicked([]);                      // a fresh choice each time it opens
  }, [open]);
  useEffect(() => {
    setPicked((current) => current.filter((n) => n <= items.length));   // an item was taken off
  }, [items.length]);

  const toggle = (position) =>
    setPicked((current) =>
      current.includes(position) ? current.filter((n) => n !== position) : [...current, position].sort((a, b) => a - b),
    );
  const ask = (text, { close = false } = {}) => {
    if (close) onClose();
    onAsk(text);
  };
  const together = (numbers) =>
    numbers.length > 1 ? `${numbers.slice(0, -1).join(", ")} and ${numbers[numbers.length - 1]}` : `${numbers[0]}`;

  const footer = items.length > 0 && (
    <>
      <button
        className="button button--wide"
        disabled={busy || picked.length < 2}
        onClick={() => ask(`Compare wishlist items ${together(picked)}`, { close: true })}
      >
        {picked.length < 2 ? "Tick two or more to compare" : `Compare ${picked.length} items`}
      </button>
      <button className="button button--ghost button--wide" disabled={busy} onClick={() => ask("Show my wishlist", { close: true })}>
        Show them all on screen
      </button>
    </>
  );

  return (
    <Sheet open={open} onClose={onClose} title="Wishlist" footer={footer}>
      {items.length === 0 ? (
        <div className="cart-empty">
          <Icon name="heart" />
          <p>
            <strong>Nothing saved yet</strong>
          </p>
          <p className="muted">When you like something but aren't ready to buy, press Save or say "save it for later".</p>
        </div>
      ) : (
        <ol className="cart-items" role="list" aria-label={`${items.length} saved`}>
          {items.map((item, index) => {
            const position = index + 1;
            return (
              <li key={item.product_id} className="cart-item wishlist-item">
                <input
                  type="checkbox"
                  className="wishlist-item__pick"
                  checked={picked.includes(position)}
                  onChange={() => toggle(position)}
                  aria-label={`Compare ${item.name}`}
                />
                <span className="cart-item__thumb" aria-hidden="true">
                  {item.image_url ? <img src={item.image_url} alt="" loading="lazy" /> : null}
                </span>
                <span className="cart-item__name">{item.name}</span>
                <span className="cart-item__price">{formatPrice(item.price)}</span>
                <button
                  className="button button--ghost cart-item__remove"
                  disabled={busy}
                  onClick={() => ask(`Remove wishlist item ${position}`)}
                  aria-label={`Remove ${item.name} from your wishlist`}
                >
                  Remove
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </Sheet>
  );
}
