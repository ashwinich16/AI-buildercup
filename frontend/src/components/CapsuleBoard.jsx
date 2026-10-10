import { formatPrice } from "../format.js";

// The outfit plan as a board: one row per look (day or occasion), the shopper's own pieces next to the few
// new ones; then what to pack, what to leave at home (and why), and the new pieces to buy. Scout says it out loud too; the buttons are one-press shortcuts
// that go through the conversation like anything the shopper says.
export default function CapsuleBoard({ capsule, busy, onAsk, onTryOn, onOpenWardrobe }) {
  if (!capsule) return null;
  const found = capsule.new_pieces.filter((piece) => piece.product);
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
        {capsule.looks.map((look, index) => (
          <li key={`${look.title}-${index}`} className="capsule__look">
            <div className="capsule__look-head">
              <h3 className="capsule__look-title">{look.title}</h3>
              {look.product_ids.length > 0 && (
                <button
                  className="button button--ghost button--small"
                  onClick={() => onTryOn({ productIds: look.product_ids, label: `${look.title} look` })}
                >
                  Try this look on
                </button>
              )}
            </div>
            <ul className="capsule__tiles" role="list">
              {look.tiles.map((tile, tileIndex) => (
                <li
                  key={tileIndex}
                  className={`capsule__tile ${tile.owned ? "is-owned" : "is-new"} ${tile.pending ? "is-pending" : ""} ${tile.current ? "is-current" : ""}`}
                >
                  <div className="capsule__image">
                    {tile.image_url ? <img src={tile.image_url} alt="" loading="lazy" /> : null}
                  </div>
                  <p className="capsule__badge">
                    {tile.owned ? "Yours" : tile.current ? "Choosing now" : tile.pending ? "To choose" : "New"}
                  </p>
                  <p className="capsule__name muted">{tile.description}</p>
                </li>
              ))}
            </ul>
            {look.why && <p className="capsule__why muted">{look.why}</p>}
          </li>
        ))}
      </ol>

      {capsule.packed.length > 0 && <OwnPieces title="Pack these" pieces={capsule.packed} />}
      {capsule.leave_at_home.length > 0 && <OwnPieces title="Better left at home" pieces={capsule.leave_at_home} />}

      {capsule.new_pieces.length > 0 && (
        <div className="capsule__buy">
          <h3 className="capsule__look-title">New pieces to buy</h3>
          <ul className="capsule__new" role="list">
            {capsule.new_pieces.map((piece) => (
              <li key={piece.key} className="capsule__new-piece">
                <p className="capsule__kind">
                  {piece.category}{" "}
                  <span className="muted">
                    · in {piece.looks} {piece.looks === 1 ? "look" : "looks"}
                  </span>
                </p>
                <p className="capsule__name">{pieceStatus(piece, capsule.guided)}</p>
                {!piece.current && !piece.skipped && (piece.product || !capsule.guided) && (
                  <button
                    className="button button--ghost button--small"
                    disabled={busy}
                    onClick={() => onAsk(`Show me a different ${piece.category} for the plan`)}
                  >
                    {capsule.guided ? "Change" : "Swap"}
                  </button>
                )}
                {piece.current && (
                  <button
                    className="button button--ghost button--small"
                    disabled={busy}
                    onClick={() => onAsk(`Skip the ${piece.category}`)}
                  >
                    Skip
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="capsule__actions">
        <button
          className="button"
          disabled={busy || found.length === 0}
          onClick={() => onAsk("Add the new pieces from the plan to my cart")}
        >
          Add {found.length} new {found.length === 1 ? "piece" : "pieces"} to cart
        </button>
        <button className="button button--ghost" disabled={busy} onClick={() => onAsk("Let's leave the plan and just shop")}>
          Back to regular shopping
        </button>
      </div>
    </section>
  );
}

// The shopper's own pieces in a row; for ones better left at home, the reason under each.
function OwnPieces({ title, pieces }) {
  return (
    <div className="capsule__buy">
      <h3 className="capsule__look-title">{title}</h3>
      <ul className="capsule__tiles" role="list">
        {pieces.map((piece, index) => (
          <li key={index} className="capsule__tile is-owned">
            <div className="capsule__image">{piece.image_url ? <img src={piece.image_url} alt="" loading="lazy" /> : null}</div>
            <p className="capsule__name">{piece.description}</p>
            {piece.why && <p className="capsule__why muted">{piece.why}</p>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function pieceStatus(piece, guided) {
  if (piece.product) return `${piece.product.name} · ${formatPrice(piece.product.price)}`;
  if (piece.skipped) return "Skipped";
  if (piece.current) return "Choosing now: pick one of the options below";
  return guided ? "Still to choose" : "Nothing suitable in the shop yet";
}
