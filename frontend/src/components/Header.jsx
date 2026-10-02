export default function Header({ shopper, settings, onSettingsChange, onLeave }) {
  const set = (key, value) => onSettingsChange({ ...settings, [key]: value });

  return (
    <header className="header">
      <div className="header__brand">
        <span className="brand">Scout</span>
        {shopper.name && <span className="muted header__who">Shopping as {shopper.name}</span>}
      </div>

      <div className="header__controls" role="group" aria-label="Display and voice settings">
        <label className="toggle">
          <input type="checkbox" checked={settings.voice} onChange={(e) => set("voice", e.target.checked)} />
          <span>Scout speaks replies</span>
        </label>

        <label className="select">
          <span>Listening language</span>
          <select value={settings.language} onChange={(e) => set("language", e.target.value)}>
            <option value="en-IN">English (India)</option>
            <option value="en-US">English (US)</option>
            <option value="en-GB">English (UK)</option>
            <option value="hi-IN">Hindi</option>
            <option value="te-IN">Telugu</option>
            <option value="ta-IN">Tamil</option>
          </select>
        </label>

        <label className="select">
          <span>Speech speed</span>
          <select value={settings.rate} onChange={(e) => set("rate", Number(e.target.value))}>
            <option value={0.85}>Slower</option>
            <option value={1}>Normal</option>
            <option value={1.2}>Faster</option>
            <option value={1.45}>Fastest</option>
          </select>
        </label>

        <label className="toggle">
          <input
            type="checkbox"
            checked={settings.highContrast}
            onChange={(e) => set("highContrast", e.target.checked)}
          />
          <span>High contrast</span>
        </label>

        <label className="toggle">
          <input type="checkbox" checked={settings.largeText} onChange={(e) => set("largeText", e.target.checked)} />
          <span>Larger text</span>
        </label>

        <button className="button button--quiet" onClick={onLeave}>
          {shopper.leave.label}
        </button>
      </div>
    </header>
  );
}
