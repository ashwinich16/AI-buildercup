import { formatPrice } from "../format.js";
import Icon from "./Icons.jsx";
import PairsWith from "./PairsWith.jsx";
import Sheet from "./Sheet.jsx";

export default function CartSheet({ open, onClose, cart, suggestion, busy, onAsk }) {
  const items = cart.items || [];
  // Checking out is a conversation, so the cart closes; removing items keeps it open to see the change.
  const ask = (text, { close = false } = {}) => {
    if (close) onClose();
    onAsk(text);
  };

  // Items without a listed price are not counted as $0: the total says so instead.
  const unpriced = items.filter((item) => item.price == null).length;
  const priced = items.length - unpriced;

  const footer = items.length > 0 && (
    <>
      <p className="cart-total">
        <span>Total</span>
        <strong>{priced ? formatPrice(cart.total || 0) : "Not listed"}</strong>
      </p>
      {unpriced > 0 && (
        <p className="muted cart-note">
          {priced ? "Plus " : ""}
          {unpriced} {unpriced === 1 ? "item" : "items"} without a listed price.
        </p>
      )}
      <button className="button button--wide" disabled={busy} onClick={() => ask("I'd like to check out", { close: true })}>
        Check out
      </button>
      <button className="button button--ghost button--wide" disabled={busy} onClick={() => ask("Please empty my cart")}>
        Empty cart
      </button>
    </>
  );

  return (
    <Sheet open={open} onClose={onClose} title="Cart" footer={footer}>
      {items.length === 0 ? (
        <div className="cart-empty">
          <Icon name="bag" />
          <p>
            <strong>Your cart is empty</strong>
          </p>
          <p className="muted">When you find something you love, say "add it to my cart" or press Add to cart.</p>
        </div>
      ) : (
        <ol className="cart-items" role="list" aria-label={`${items.length} ${items.length === 1 ? "item" : "items"}`}>
          {items.map((item, index) => (
            <li key={item.product_id} className="cart-item">
              <span className="cart-item__thumb" aria-hidden="true">
                {item.image_url ? <img src={item.image_url} alt="" loading="lazy" /> : null}
              </span>
              <span className="cart-item__name">{item.name}</span>
              <span className="cart-item__price">{formatPrice(item.price)}</span>
              <button
                className="button button--ghost cart-item__remove"
                disabled={busy}
                onClick={() => ask(`Remove item ${index + 1} from my cart`)}
                aria-label={`Remove ${item.name}`}
              >
                Remove
              </button>
            </li>
          ))}
        </ol>
      )}
      {items.length > 0 && (
        <PairsWith suggestion={suggestion} busy={busy} onAsk={(text) => ask(text, { close: true })} compact />
      )}
    </Sheet>
  );
}
