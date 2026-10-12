import { formatPrice } from "../format.js";

// The product Scout is describing when it is not one of the options on screen (for example the item it
// suggested to go with the cart). Shown large at the top of the page while Scout talks about it.
export default function Spotlight({ product, speaking, busy, addPhrase, onAsk }) {
  if (!product) return null;
  const evidence = product.evidence || [];

  return (
    <section
      className={`spotlight ${speaking ? "is-speaking" : ""}`}
      aria-labelledby="spotlight-title"
      aria-current={speaking ? "true" : undefined}
    >
      <div className="spotlight__media">
        {product.image_url ? <img src={product.image_url} alt="" /> : <span className="muted">No photo</span>}
      </div>
      <div className="spotlight__body">
        <p className="spotlight__eyebrow">{speaking ? "Scout is describing this" : "You asked about this"}</p>
        <h2 id="spotlight-title" className="spotlight__name">
          {product.name}
        </h2>
        <p className="card__meta">
          <span className="card__price">{formatPrice(product.price)}</span>
          {product.rating != null && (
            <span className="rating" aria-label={`Rated ${product.rating.toFixed(1)} out of 5`}>
              <span className="rating__star" aria-hidden="true">
                ★
              </span>{" "}
              <span aria-hidden="true">{product.rating.toFixed(1)}</span>
              {product.rating_count ? <span aria-hidden="true"> ({product.rating_count.toLocaleString()})</span> : null}
            </span>
          )}
        </p>
        {product.summary && <p className="spotlight__summary">{product.summary}</p>}
        {evidence.length > 0 && (
          <ul className="evidence" aria-label="What reviewers say">
            {evidence.map((item) => (
              <li key={item.quote} className={`evidence__item ${item.heads_up ? "is-heads-up" : ""}`}>
                <span className="evidence__quote">
                  {item.heads_up && <span className="evidence__flag">Heads-up: </span>}
                  <q>{item.quote}</q>
                </span>
                <span className="evidence__count">{item.count}</span>
              </li>
            ))}
          </ul>
        )}
        {addPhrase && (
          <div className="spotlight__actions">
            <button className="button" disabled={busy} onClick={() => onAsk(addPhrase)}>
              Add to cart
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
