import { formatPrice } from "../format.js";

export default function CartPanel({ cart, busy, onAsk }) {
  const items = cart.items || [];

  return (
    <section className="panel cart" aria-labelledby="cart-title">
      <h2 id="cart-title" className="panel__title">
        Your cart ({items.length} {items.length === 1 ? "item" : "items"})
      </h2>

      {items.length === 0 ? (
        <p className="muted">Your cart is empty. Say "add it to my cart" when you find something you love.</p>
      ) : (
        <>
          <ol className="cart__items" role="list">
            {items.map((item, index) => (
              <li key={item.product_id} className="cart__item">
                <span className="cart__name">{item.name}</span>
                <span className="cart__price">{formatPrice(item.price)}</span>
                <button
                  className="button button--quiet button--small"
                  disabled={busy}
                  onClick={() => onAsk(`Remove item ${index + 1} from my cart`)}
                  aria-label={`Remove ${item.name}`}
                >
                  Remove
                </button>
              </li>
            ))}
          </ol>
          <p className="cart__total">
            Total <strong>${(cart.total || 0).toFixed(2)}</strong>
          </p>
          <button className="button button--wide" disabled={busy} onClick={() => onAsk("I'd like to check out")}>
            Check out
          </button>
          <button className="button button--quiet button--wide" disabled={busy} onClick={() => onAsk("Please empty my cart")}>
            Empty cart
          </button>
        </>
      )}
    </section>
  );
}
