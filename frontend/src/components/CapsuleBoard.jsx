import { formatPrice } from "../format.js";

// The outfit as a board: one row per look, each with its own pieces (the shopper's own and new ones), each
// look ready to try on. Every piece can be changed or removed in its own look. Scout says it out loud too;
// the buttons are one-press shortcuts that go through the conversation like anything the shopper says.
export default function CapsuleBoard({ capsule, busy, onAsk, onTryOn, onOpenWardrobe }) {
  if (!capsule) return null;
  const toBuy = capsule.to_buy.length;
  const ownedCount = capsule.looks.flatMap((look) => look.tiles).filter((tile) => tile.owned).length;

  return (
    <section className="capsule" aria-labelledby="capsule-title">
      <div className="capsule__head">
        <h2 id="capsule-title" className="capsule__title">
          Your looks <span className="muted">for {capsule.occasion}</span>
        </h2>
        <span className="capsule__total">
          {formatPrice(capsule.total)} new
          {capsule.budget ? <span className="muted"> of {formatPrice(capsule.budget)}</span> : null}
        </span>
      </div>
      {ownedCount === 0 && (
        <p className="capsule__hint muted">
          Every piece here is new.{" "}
          <button className="link-button" onClick={onOpenWardrobe}>
            Add clothes you own
          </button>{" "}
          and Scout will plan around them.
        </p>
      )}

      <ol className="capsule__looks" role="list">
        {capsule.looks.map((look, index) => {
          const number = index + 1;
          return (
            <li key={`${look.title}-${index}`} className="capsule__look">
              <div className="capsule__look-head">
                <h3 className="capsule__look-title">{look.title}</h3>
                {look.product_ids.length + look.wardrobe_ids.length > 0 && (
                  <button
                    className="button button--ghost button--small"
                    onClick={() =>
                      onTryOn({
                        productIds: look.product_ids,
                        wardrobeIds: look.wardrobe_ids,
                        label: `${look.title} look`,
                        items: look.tiles
                          .filter((tile) => !tile.missing)
                          .map((tile) => ({ name: tile.description, price: tile.owned ? null : tile.price, owned: tile.owned })),
                      })
                    }
                  >
                    Try this look on
                  </button>
                )}
              </div>
              <ul className="capsule__tiles" role="list">
                {look.tiles.map((tile, tileIndex) => (
                  <li
                    key={tileIndex}
                    className={`capsule__tile ${tile.owned ? "is-owned" : "is-new"} ${tile.missing ? "is-pending" : ""} ${tile.changing ? "is-current" : ""}`}
                  >
                    <div className="capsule__image">
                      {tile.image_url ? (
                        <img src={tile.image_url} alt="" loading="lazy" />
                      ) : (
                        <span className="capsule__placeholder muted">
                          {tile.category}
                          {tile.owned && <small>No photo</small>}
                        </span>
                      )}
                    </div>
                    <p className="capsule__badge">
                      {tile.changing ? "Changing" : tile.owned ? "Yours" : tile.missing ? "Not found" : "New"}
                    </p>
                    <p className="capsule__name muted">{tile.description}</p>
                    {tile.price != null && <p className="capsule__price">{formatPrice(tile.price)}</p>}
                    <div className="capsule__tile-actions">
                      <button
                        className="link-button"
                        disabled={busy}
                        aria-label={`Change the ${tile.category} in look ${number}`}
                        onClick={() => onAsk(`Change the ${tile.category} in look ${number}`)}
                      >
                        Change
                      </button>
                      <button
                        className="link-button"
                        disabled={busy}
                        aria-label={`Remove the ${tile.category} from look ${number}`}
                        onClick={() => onAsk(`Remove the ${tile.category} from look ${number}`)}
                      >
                        Remove
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
              {look.why && <p className="capsule__why muted">{look.why}</p>}
            </li>
          );
        })}
      </ol>

      <div className="capsule__actions">
        <button
          className="button"
          disabled={busy || toBuy === 0}
          onClick={() => onAsk("Add the new pieces from the plan to my cart")}
        >
          Add {toBuy} new {toBuy === 1 ? "piece" : "pieces"} to cart
        </button>
        <button className="button button--ghost" disabled={busy} onClick={() => onAsk("Let's leave the plan and just shop")}>
          Back to regular shopping
        </button>
      </div>
    </section>
  );
}
