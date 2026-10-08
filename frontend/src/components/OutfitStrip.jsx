import { formatPrice } from "../format.js";

// The outfit being put together: every piece in a row, filled in as the shopper picks them, so the whole
// look is visible at once. Scout also says it out loud; the button is a one-press way to add it all.
export default function OutfitStrip({ outfit, busy, onAsk, onTryOn }) {
  if (!outfit) return null;
  const picked = outfit.pieces.filter((piece) => piece.picked).map((piece) => piece.picked);

  return (
    <section className="outfit" aria-labelledby="outfit-title">
      <div className="outfit__head">
        <h2 id="outfit-title" className="outfit__title">
          Your outfit <span className="muted">for {outfit.occasion}</span>
        </h2>
        {picked.length > 0 && <span className="outfit__total">{formatPrice(outfit.total)}</span>}
      </div>
      <ol className="outfit__pieces" role="list">
        {outfit.pieces.map((piece, index) => (
          <li key={`${piece.kind}-${index}`} className={`outfit__piece ${piece.picked ? "is-picked" : ""}`}>
            <div className="outfit__image">
              {piece.picked?.image_url ? <img src={piece.picked.image_url} alt="" loading="lazy" /> : null}
            </div>
            <p className="outfit__kind">{piece.kind}</p>
            <p className="outfit__name muted">{piece.picked ? piece.picked.name : "Still to choose"}</p>
          </li>
        ))}
      </ol>
      <div className="outfit__actions">
        <button
          className="button"
          disabled={busy || picked.length === 0}
          onClick={() => onAsk("Put the whole outfit in my cart")}
        >
          Add whole outfit to cart
        </button>
        <button
          className="button button--secondary"
          disabled={picked.length === 0}
          onClick={() => onTryOn({ productIds: picked.map((p) => p.id), label: "outfit" })}
        >
          Try the outfit on
        </button>
        <button className="button button--ghost" disabled={busy} onClick={() => onAsk("Let's leave the outfit and just shop")}>
          Back to regular shopping
        </button>
      </div>
    </section>
  );
}
