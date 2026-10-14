import { useEffect, useId, useRef, useState } from "react";
import { api } from "../api.js";
import { formatPrice } from "../format.js";

// A product from the outfit board, to read for yourself: photos, price, the size it will be added in, why it
// suits the occasion, its details and what reviewers say. Built on the native <dialog>: Escape or Close
// dismisses it and focus returns to the piece that opened it. `request`: { productId, look, category } or null.
export default function ProductDetail({ request, sessionId, busy, onAsk, onTryOn, onClose }) {
  const ref = useRef(null);
  const titleId = useId();
  const [product, setProduct] = useState(null);
  const [error, setError] = useState("");
  const [photo, setPhoto] = useState(0);

  useEffect(() => {
    const dialog = ref.current;
    if (request && !dialog.open) dialog.showModal();
    if (!request && dialog.open) dialog.close();
  }, [request]);

  useEffect(() => {
    if (!request) return undefined;
    let cancelled = false;
    setProduct(null);
    setError("");
    setPhoto(0);
    api
      .productDetails(sessionId, request.productId)
      .then((found) => !cancelled && setProduct(found))
      .catch((failure) => !cancelled && setError(failure.message));
    return () => {
      cancelled = true;
    };
  }, [request, sessionId]);

  const act = (text) => {
    onClose();
    onAsk(text);
  };
  const images = product?.images || [];

  return (
    <dialog
      ref={ref}
      className="detail"
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}   // a click on the dimmed backdrop
    >
      <div className="detail__head">
        <p className="detail__eyebrow">
          {request?.category}
          {request?.look ? ` · in look ${request.look}` : ""}
        </p>
        <button className="button button--ghost button--small" onClick={onClose}>
          Close
        </button>
      </div>

      {!product && !error && <p className="detail__status muted" role="status">Getting the details…</p>}
      {error && <p className="detail__status" role="alert">{error}</p>}

      {product && (
        <div className="detail__body">
          <div className="detail__media">
            <div className="detail__photo">
              {images[photo] ? <img src={images[photo]} alt={`Photo ${photo + 1} of ${product.name}`} /> : null}
            </div>
            {images.length > 1 && (
              <ul className="detail__thumbs" role="list" aria-label="More photos">
                {images.map((src, index) => (
                  <li key={src}>
                    <button
                      className={`detail__thumb ${index === photo ? "is-current" : ""}`}
                      aria-label={`Show photo ${index + 1}`}
                      aria-pressed={index === photo}
                      onClick={() => setPhoto(index)}
                    >
                      <img
                        src={src}
                        alt=""
                        loading="lazy"
                        onError={(event) => (event.currentTarget.closest("li").hidden = true)}   // a photo that won't load
                      />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="detail__info">
            <h2 id={titleId} className="detail__name">
              {product.name}
            </h2>
            <p className="detail__meta">
              <span className="detail__price">{formatPrice(product.price)}</span>
              {product.rating != null && (
                <span className="rating" aria-label={`Rated ${product.rating.toFixed(1)} out of 5 by ${product.rating_count || 0} people`}>
                  <span aria-hidden="true">
                    ★ {product.rating.toFixed(1)}
                    {product.rating_count ? ` (${product.rating_count.toLocaleString()} ratings)` : ""}
                  </span>
                </span>
              )}
            </p>

            {product.sizes.length > 0 && (
              <div className="detail__size">
                <p>
                  <strong>{product.size ? `Size ${product.size}` : "Size not chosen yet"}</strong>
                  <span className="muted">
                    {" "}
                    · comes in {product.sizes_spoken}.{" "}
                    {product.size ? "You can change it before it goes in your cart." : "Scout asks before it goes in your cart."}
                  </span>
                </p>
              </div>
            )}

            {product.occasion_note && (
              <p className="detail__why">
                <strong>Why it suits {product.occasion}:</strong> {cleanNote(product.occasion_note)}
              </p>
            )}

            {product.summary && <p className="detail__summary">{product.summary}</p>}

            {product.facts.length > 0 && (
              <section aria-label="Details">
                <h3 className="detail__label">Details</h3>
                <dl className="detail__facts">
                  {product.facts.map((fact) => (
                    <div key={`${fact.name}-${fact.value}`}>
                      <dt>{fact.name}</dt>
                      <dd>{fact.value}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            )}

            {(product.evidence.length > 0 || product.review_points.length > 0) && (
              <section aria-label="What reviewers say">
                <h3 className="detail__label">
                  What reviewers say
                  {product.reviews_read ? (
                    <span className="muted"> · from {product.reviews_read} written {product.reviews_read === 1 ? "review" : "reviews"}</span>
                  ) : null}
                </h3>
                {product.evidence.length > 0 && (
                  <ul className="evidence" role="list">
                    {product.evidence.map((item) => (
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
                {product.review_points.length > 0 && (
                  <ul className="detail__points" role="list">
                    {product.review_points.map((point) => (
                      <li key={`${point.aspect}-${point.finding}`} className={point.heads_up ? "is-heads-up" : ""}>
                        <strong>{point.aspect}:</strong> {point.finding}
                        {point.count && <span className="muted"> ({point.count})</span>}
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            )}

            <div className="detail__actions">
              <button
                className="button"
                disabled={busy}
                onClick={() => {
                  onClose();
                  onTryOn({ productIds: [product.id], label: product.name, items: [{ name: product.name, price: product.price }] });
                }}
              >
                Try it on
              </button>
              {request?.look && (
                <button
                  className="button button--secondary"
                  disabled={busy}
                  onClick={() => act(`Change the ${request.category} in look ${request.look}`)}
                >
                  Change this piece
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </dialog>
  );
}

// Occasion notes for near misses start "a close match:": shown as a plain sentence.
const cleanNote = (note) => {
  const text = note.replace(/^a close match:\s*/i, "A close match: ");
  return text.charAt(0).toUpperCase() + text.slice(1);
};
