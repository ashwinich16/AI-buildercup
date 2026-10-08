import { useEffect, useId, useRef } from "react";
import Icon from "./Icons.jsx";

// A panel that slides in from the right (cart, settings). Built on the native <dialog>, which keeps
// keyboard focus inside while open, closes with Escape and returns focus to the button that opened it.
export default function Sheet({ open, onClose, title, children, footer }) {
  const ref = useRef(null);
  const titleId = useId();   // titles may contain spaces, which an id must not

  useEffect(() => {
    const dialog = ref.current;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="sheet"
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}   // a click on the dimmed backdrop
    >
      <div className="sheet__head">
        <h2 id={titleId} className="sheet__title">
          {title}
        </h2>
        <button className="icon-button" onClick={onClose} aria-label={`Close ${title.toLowerCase()}`}>
          <Icon name="close" />
        </button>
      </div>
      <div className="sheet__body">{children}</div>
      {footer && <div className="sheet__foot">{footer}</div>}
    </dialog>
  );
}
