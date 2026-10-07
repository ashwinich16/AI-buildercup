// Simple line icons, drawn with the text colour. Always decorative: the button or text next to
// them carries the meaning for screen readers.
const paths = {
  bag: "M6 7h12l1 13H5L6 7Zm3 0a3 3 0 0 1 6 0",
  settings: "M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M16 4v4M10 10v4M18 16v4",
  mic: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Zm-6 9a6 6 0 0 0 12 0M12 18v3",
  send: "M5 12h14M13 6l6 6-6 6",
  close: "M6 6l12 12M18 6 6 18",
  arrow: "M9 6l6 6-6 6",
  volumeOff: "M11 5 6 9H3v6h3l5 4V5Zm6 4 4 6m0-6-4 6",
};

export default function Icon({ name, className = "icon" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d={paths[name]} />
    </svg>
  );
}
