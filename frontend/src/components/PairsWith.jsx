import { useId } from "react";
import { formatPrice } from "../format.js";

// "Goes well with your cart": the one item Scout suggested to go with what the shopper chose.
// Scout also says it out loud once; this keeps it on screen with one-press ways to act on it.
export default function PairsWith({ suggestion, busy, onAsk, compact = false }) {
  const titleId = useId();   // shown twice at once (page and cart sheet), so each needs its own id
  if (!suggestion) return null;
  const reason = suggestion.reasons?.[0];
  const kind = suggestion.kind || "item";

  return (
    <section className={`pairs ${compact ? "pairs--compact" : ""}`} aria-labelledby={titleId}>
      <p className="pairs__eyebrow" id={titleId}>
        Goes well with your {suggestion.goes_with || "cart"}
      </p>
      <div className="pairs__body">
        <div className="pairs__image">
          {suggestion.image_url ? <img src={suggestion.image_url} alt="" loading="lazy" /> : null}
        </div>
        <div className="pairs__text">
          <p className="pairs__name">{suggestion.name}</p>
          <p className="pairs__meta muted">
            {formatPrice(suggestion.price)}
            {reason ? ` · ${reason.charAt(0).toUpperCase()}${reason.slice(1)}` : ""}
          </p>
        </div>
        <div className="pairs__actions">
          <button
            className="button button--secondary"
            disabled={busy}
            onClick={() => onAsk(`Show me other options like the suggested ${kind}`)}
          >
            See similar
          </button>
          <button className="button" disabled={busy} onClick={() => onAsk(`Add the suggested ${kind} to my cart`)}>
            Add to cart
          </button>
        </div>
      </div>
    </section>
  );
}
