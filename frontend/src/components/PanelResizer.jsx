// A bar on the left edge of a panel that sits on the right of the window (Scout's panel, the try-on window):
// drag it, or focus it and use the arrow keys, to make the panel wider or narrower; double-click to reset.
// The width stays between `min` pixels and `maxShare` of the window.
export default function PanelResizer({ width, onResize, min, maxShare, initial, label, className = "" }) {
  const clamp = (value) => Math.round(Math.min(Math.max(value, min), window.innerWidth * maxShare));

  const startDrag = (event) => {
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const drag = (event) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) onResize(clamp(window.innerWidth - event.clientX));
  };
  const onKey = (event) => {
    const step = { ArrowLeft: 32, ArrowRight: -32 }[event.key];   // left makes the panel wider
    if (!step) return;
    event.preventDefault();
    onResize(clamp(width + step));
  };

  return (
    <div
      className={`resizer ${className}`}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={width}
      aria-valuemin={min}
      tabIndex={0}
      onPointerDown={startDrag}
      onPointerMove={drag}
      onKeyDown={onKey}
      onDoubleClick={() => onResize(initial)}
    />
  );
}
