import { useEffect, useId, useRef, useState } from "react";
import Icon from "./Icons.jsx";
import PanelResizer from "./PanelResizer.jsx";

// A resizable sheet can be dragged wider or narrower, within these limits (pixels).
const DEFAULT_SHEET_WIDTH = 420;
const MIN_SHEET_WIDTH = 320;
const MAX_SHEET_SHARE = 0.9;   // a little of the page stays visible beside it

// A panel that slides in from the right (cart, settings). Built on the native <dialog>, which keeps
// keyboard focus inside while open, closes with Escape and returns focus to the button that opened it.
// `resizable` adds a bar on its left edge to make it wider (the try-on, to see the picture bigger).
export default function Sheet({ open, onClose, title, children, footer, resizable = false }) {
  const ref = useRef(null);
  const [width, setWidth] = useState(DEFAULT_SHEET_WIDTH);
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
      style={resizable ? { "--sheet-width": `${width}px` } : undefined}
      aria-labelledby={titleId}
      onClose={onClose}
      onClick={(event) => event.target === ref.current && onClose()}   // a click on the dimmed backdrop
    >
      {resizable && (
        <PanelResizer
          className="sheet__resizer"
          width={width}
          onResize={setWidth}
          min={MIN_SHEET_WIDTH}
          maxShare={MAX_SHEET_SHARE}
          initial={DEFAULT_SHEET_WIDTH}
          label={`Resize the ${title.toLowerCase()} window`}
        />
      )}
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
