import { useEffect } from "react";
import { formatPrice } from "../format.js";

const ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"];

// Next steps once options are showing.
const QUICK_ACTIONS = [
  "Show me more options",
  "Compare these for me",
  "Something cheaper",
  "Describe the first one",
];

// `choosingFor`: while a piece of an outfit is being changed, what these options are for ("shoe for look 1").
// `savedIds`: products already on the wishlist (their Save button says so).
export default function ProductList({ products, focusId, spokenPosition, busy, choosingFor, savedIds = [], onAsk, onTryOn }) {
  // Keep the option Scout is talking about in view: the one being described right now, else the focus.
  const spokenId = products.find((p) => p.position === spokenPosition)?.id;
  const inView = spokenId || focusId;
  useEffect(() => {
    if (inView) bringIntoView(document.getElementById(`card-${inView}`));
  }, [inView]);

  if (!products.length) return null;

  return (
    <section className="results" aria-labelledby="results-title">
      <div className="results__head">
        <div>
          <h2 id="results-title" className="results__title">
            {products.length} {products.length === 1 ? "option" : "options"} {choosingFor ? `for your ${choosingFor}` : "for you"}.{" "}
            <span className="muted">{choosingFor ? "Choose one for the outfit." : "Picked from what you asked."}</span>
          </h2>
          <p className="results__hint muted">Say "tell me about the first one", or use the buttons.</p>
        </div>
      </div>

      <ol className="products" role="list">
        {products.map((product) => (
          <li key={product.id}>
            <ProductCard
              product={product}
              focused={product.id === focusId}
              speaking={product.id === spokenId}
              busy={busy}
              choosing={Boolean(choosingFor)}
              saved={savedIds.includes(product.id)}
              onAsk={onAsk}
              onTryOn={onTryOn}
            />
          </li>
        ))}
      </ol>

      <div className="quick">
        <p className="quick__label" id="quick-label">
          Next, you could ask:
        </p>
        <ul className="chips" role="list" aria-labelledby="quick-label">
          {QUICK_ACTIONS.filter((action) => products.length > 1 || !action.startsWith("Compare")).map((action) => (
            <li key={action}>
              <button className="chip" disabled={busy} onClick={() => onAsk(action)}>
                {action}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function ProductCard({ product, focused, speaking, busy, choosing, saved, onAsk, onTryOn }) {
  const position = ORDINALS[product.position - 1] || `Option ${product.position}`;
  const describe = () => onAsk(`Describe option ${product.position} in detail`);
  // The strongest quote, plus a heads-up if reviewers flag one: enough to judge at a glance and keep
  // the card short. The full review picture is one press away (Describe).
  const evidence = product.evidence || [];
  const shownEvidence = [evidence.find((e) => !e.heads_up), evidence.find((e) => e.heads_up)].filter(Boolean);

  return (
    <article
      id={`card-${product.id}`}
      className={`card ${focused ? "is-focus" : ""} ${speaking ? "is-speaking" : ""}`}
      aria-current={speaking ? "true" : undefined}
      aria-labelledby={`name-${product.id}`}
    >
      <button className="card__media" onClick={describe} disabled={busy} aria-label={`Describe ${position.toLowerCase()} option`}>
        {product.image_url ? (
          <img src={product.image_url} alt="" loading="lazy" />
        ) : (
          <span className="card__noimage">No photo</span>
        )}
        <span className="card__index" aria-hidden="true">
          {product.position}
        </span>
        {!product.exact_match && (
          <span className="card__tag" aria-hidden="true">
            Close match
          </span>
        )}
        {(speaking || focused) && (
          <span className="card__talking" aria-hidden="true">
            {speaking ? "Scout is describing this" : "Talking about this"}
          </span>
        )}
      </button>

      <div className="card__body">
        <h3 id={`name-${product.id}`} className="card__name">
          <span className="visually-hidden">
            {position} option{product.exact_match ? "" : ", a close match"}:{" "}
          </span>
          {product.name}
        </h3>

        <p className="card__meta">
          <span className={`card__price ${product.price == null ? "card__price--unknown" : ""}`}>
            {formatPrice(product.price)}
          </span>
          {product.rating != null && (
            <span
              className="rating"
              aria-label={`Rated ${product.rating.toFixed(1)} out of 5${
                product.rating_count ? ` from ${product.rating_count.toLocaleString()} ratings` : ""
              }`}
            >
              <span className="rating__star" aria-hidden="true">
                ★
              </span>
              <span className="rating__value" aria-hidden="true">
                {product.rating.toFixed(1)}
              </span>
              {product.rating_count ? <span aria-hidden="true">({product.rating_count.toLocaleString()})</span> : null}
            </span>
          )}
        </p>

        {shownEvidence.length > 0 && (
          <ul className="evidence" aria-label="What reviewers say">
            {shownEvidence.map((item) => (
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

        <div className="card__actions">
          <button className="button button--secondary" disabled={busy} onClick={describe}>
            Describe
          </button>
          {choosing ? (
            <button className="button" disabled={busy} onClick={() => onAsk(`I'll choose option ${product.position} for the outfit`)}>
              Choose this
            </button>
          ) : (
            <button className="button" disabled={busy} onClick={() => onAsk(`Add option ${product.position} to my cart`)}>
              Add to cart
            </button>
          )}
          <button
            className="button button--ghost card__try"
            onClick={() => onTryOn({ productIds: [product.id], label: product.name })}
          >
            Try it on
          </button>
          <button
            className="button button--ghost"
            disabled={busy || saved}
            aria-pressed={saved}
            onClick={() => onAsk(`Save option ${product.position} to my wishlist`)}
          >
            {saved ? "Saved" : "Save"}
          </button>
        </div>
      </div>
    </article>
  );
}


// Scroll the products panel (not the whole page) so the card is fully visible. On a phone the panel does
// not scroll by itself, so there the page scrolls instead.
function bringIntoView(card) {
  const panel = card?.closest(".layout__shop");
  if (!card || !panel) return;
  if (panel.scrollHeight <= panel.clientHeight) {
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  const cardBox = card.getBoundingClientRect();
  const panelBox = panel.getBoundingClientRect();
  if (cardBox.top < panelBox.top || cardBox.bottom > panelBox.bottom) {
    panel.scrollBy({ top: cardBox.top - panelBox.top - 16, behavior: "smooth" });
  }
}
