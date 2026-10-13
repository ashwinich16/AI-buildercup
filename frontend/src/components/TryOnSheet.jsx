import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { formatPrice } from "../format.js";
import Sheet from "./Sheet.jsx";

// Virtual try-on: the shopper's photo wearing the chosen product (or the whole outfit). When the try-on can
// follow instructions (`adjustable`: Gemini's image model), they can say how they'd like it worn ("tuck the
// shirt in") and try again; the other model cannot, so the box is not offered there.
// The photo stays in this page only (App keeps it while the page is open), so the shopper can try on
// several things without choosing it again; the server uses it for one request and never stores it.
export default function TryOnSheet({ request, sessionId, photo, onPhoto, onClose, adjustable = false }) {
  const [result, setResult] = useState(null);   // object URL of the try-on image
  const [status, setStatus] = useState("");      // what is happening, also read out by screen readers
  const [wish, setWish] = useState("");          // what they are typing: how to wear it
  const [applied, setApplied] = useState("");    // the wish the current picture was made with
  const [busy, setBusy] = useState(false);
  const fileRef = useRef(null);

  useEffect(() => {                              // something new to try on: start without a wish
    setWish("");
    setApplied("");
  }, [request]);

  // Try it on whenever there is something to try and a photo to try it on (and again with a new wish).
  useEffect(() => {
    if (!request || !photo) return undefined;
    let url = null;
    let cancelled = false;
    setResult(null);
    setBusy(true);
    setStatus(`Putting the ${request.label} on your photo. This takes a few seconds…`);
    api
      .tryOn(sessionId, request, photo, applied)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setResult(url);
        setStatus(`Here you are wearing the ${request.label}.`);
      })
      .catch((error) => !cancelled && setStatus(error.message))
      .finally(() => !cancelled && setBusy(false));
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [request, photo, sessionId, applied]);

  const adjust = (event) => {
    event.preventDefault();
    if (wish.trim() && wish.trim() !== applied) setApplied(wish.trim());
  };

  const choosePhoto = (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";                     // choosing the same file again still counts
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => onPhoto({ data: String(reader.result).split(",")[1], mimeType: file.type });
    reader.readAsDataURL(file);
  };

  return (
    <Sheet open={Boolean(request)} onClose={onClose} title="Try on">
      <input ref={fileRef} type="file" accept="image/jpeg,image/png" hidden onChange={choosePhoto} />
      <TryOnItems items={request?.items || []} />
      {!photo ? (
        <div className="try-on__start">
          <p>
            Add a photo of yourself, standing and facing the camera, to see the {request?.label} on you.
          </p>
          <p className="muted">Your photo is only used for this and is never saved.</p>
          <button className="button" onClick={() => fileRef.current?.click()}>
            Choose a photo
          </button>
        </div>
      ) : (
        <div className="try-on__result">
          <p role="status" className="muted">
            {status}
          </p>
          {result && <img src={result} alt={`You wearing the ${request?.label}`} />}
          {adjustable && (
            <form className="try-on__adjust" onSubmit={adjust}>
              <label htmlFor="try-on-wish">Anything to adjust?</label>
              <input
                id="try-on-wish"
                value={wish}
                onChange={(e) => setWish(e.target.value)}
                placeholder="e.g. tuck the shirt in, roll up the sleeves"
                maxLength={200}
                autoComplete="off"
              />
              <button className="button" type="submit" disabled={busy || !wish.trim()}>
                Try again
              </button>
            </form>
          )}
          <button className="button button--secondary" onClick={() => fileRef.current?.click()}>
            Use a different photo
          </button>
        </div>
      )}
    </Sheet>
  );
}

// What is being tried on, with prices (their own pieces say so), and the total of the new ones.
function TryOnItems({ items }) {
  if (items.length === 0) return null;
  const priced = items.filter((item) => item.price != null);
  const total = priced.reduce((sum, item) => sum + item.price, 0);
  return (
    <div className="try-on__items">
      <ul role="list">
        {items.map((item, index) => (
          <li key={`${item.name}-${index}`}>
            <span className="try-on__item-name">{item.name}</span>
            <span className="try-on__item-price">{item.owned ? "Yours" : formatPrice(item.price)}</span>
          </li>
        ))}
      </ul>
      {priced.length > 1 && (
        <p className="try-on__total">
          <span>Total</span>
          <strong>{formatPrice(total)}</strong>
        </p>
      )}
    </div>
  );
}
