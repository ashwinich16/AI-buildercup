const ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth"];

function price(value) {
  return value == null ? "Price not listed" : `$${value.toFixed(2)}`;
}

export default function ProductList({ products, focusId, busy, onAsk }) {
  if (!products.length) {
    return (
      <section className="panel" aria-labelledby="options-title">
        <h2 id="options-title" className="panel__title">
          Options
        </h2>
        <p className="muted">Tell Scout what you're looking for, and the options will appear here.</p>
      </section>
    );
  }

  return (
    <section className="panel" aria-labelledby="options-title">
      <div className="panel__head">
        <h2 id="options-title" className="panel__title">
          Options ({products.length})
        </h2>
        {products.length > 1 && (
          <button className="button button--quiet" disabled={busy} onClick={() => onAsk("Compare these options for me")}>
            Compare
          </button>
        )}
      </div>

      <ol className="products" role="list">
        {products.map((product) => {
          const label = ORDINALS[product.position - 1] || `Option ${product.position}`;
          return (
            <li key={product.id}>
              <article
                className={`product ${product.id === focusId ? "is-focus" : ""}`}
                aria-labelledby={`product-${product.id}`}
              >
                <div className="product__image">
                  {product.image_url ? (
                    <img src={product.image_url} alt={`Photo: ${product.summary}`} loading="lazy" />
                  ) : (
                    <span className="muted">No photo</span>
                  )}
                </div>
                <div className="product__body">
                  <p className="product__position">
                    {label}
                    {!product.exact_match && <span className="badge">Close match</span>}
                  </p>
                  <h3 id={`product-${product.id}`} className="product__name">
                    {product.name}
                  </h3>
                  <p className="product__meta">
                    <strong>{price(product.price)}</strong>
                    {product.rating != null && (
                      <span>
                        {" "}
                        · Rated {product.rating.toFixed(1)} of 5
                        {product.rating_count ? ` (${product.rating_count} ratings)` : ""}
                      </span>
                    )}
                  </p>
                  <div className="product__actions">
                    <button
                      className="button button--quiet"
                      disabled={busy}
                      onClick={() => onAsk(`Describe option ${product.position} in detail`)}
                      aria-describedby={`product-${product.id}`}
                    >
                      Describe
                    </button>
                    <button
                      className="button"
                      disabled={busy}
                      onClick={() => onAsk(`Add option ${product.position} to my cart`)}
                      aria-describedby={`product-${product.id}`}
                    >
                      Add to cart
                    </button>
                  </div>
                </div>
              </article>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
