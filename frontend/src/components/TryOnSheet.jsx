import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import Sheet from "./Sheet.jsx";

// Virtual try-on: the shopper's photo wearing the chosen product (or the whole outfit).
// The photo stays in this page only (App keeps it while the page is open), so the shopper can try on
// several things without choosing it again; the server uses it for one request and never stores it.
export default function TryOnSheet({ request, sessionId, photo, onPhoto, onClose }) {
  const [result, setResult] = useState(null);   // object URL of the try-on image
  const [status, setStatus] = useState("");      // what is happening, also read out by screen readers
  const fileRef = useRef(null);

  // Try it on whenever there is something to try and a photo to try it on.
  useEffect(() => {
    if (!request || !photo) return undefined;
    let url = null;
    let cancelled = false;
    setResult(null);
    setStatus(`Putting the ${request.label} on your photo. This takes a few seconds…`);
    api
      .tryOn(sessionId, request, photo)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setResult(url);
        setStatus(`Here you are wearing the ${request.label}.`);
      })
      .catch((error) => !cancelled && setStatus(error.message));
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [request, photo, sessionId]);

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
          <button className="button button--secondary" onClick={() => fileRef.current?.click()}>
            Use a different photo
          </button>
        </div>
      )}
    </Sheet>
  );
}
