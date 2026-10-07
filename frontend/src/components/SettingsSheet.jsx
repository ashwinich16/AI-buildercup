import Sheet from "./Sheet.jsx";

function Switch({ label, hint, checked, onChange }) {
  return (
    <label className="switch">
      <span className="switch__text">
        <span>{label}</span>
        {hint && <span className="switch__hint">{hint}</span>}
      </span>
      <input type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </label>
  );
}

export default function SettingsSheet({ open, onClose, settings, onSettingsChange }) {
  const set = (key, value) => onSettingsChange({ ...settings, [key]: value });

  return (
    <Sheet open={open} onClose={onClose} title="Settings">
      <fieldset className="settings-group">
        <legend>Voice</legend>
        <Switch label="Scout speaks replies" checked={settings.voice} onChange={(v) => set("voice", v)} />
        <Switch
          label="Using headphones"
          hint="Interrupt Scout with any word. Without headphones it takes a short phrase."
          checked={settings.headphones}
          onChange={(v) => set("headphones", v)}
        />
        <label className="field">
          <span className="field__label">Speech speed</span>
          <select value={settings.rate} onChange={(e) => set("rate", Number(e.target.value))}>
            <option value={0.85}>Slower</option>
            <option value={1}>Normal</option>
            <option value={1.2}>Faster</option>
            <option value={1.45}>Fastest</option>
          </select>
        </label>
        <label className="field">
          <span className="field__label">Listening language</span>
          <select value={settings.language} onChange={(e) => set("language", e.target.value)}>
            <option value="en-IN">English (India)</option>
            <option value="en-US">English (US)</option>
            <option value="en-GB">English (UK)</option>
            <option value="hi-IN">Hindi</option>
            <option value="te-IN">Telugu</option>
            <option value="ta-IN">Tamil</option>
          </select>
        </label>
      </fieldset>

      <fieldset className="settings-group">
        <legend>Display</legend>
        <Switch label="High contrast" checked={settings.highContrast} onChange={(v) => set("highContrast", v)} />
        <Switch label="Larger text" checked={settings.largeText} onChange={(v) => set("largeText", v)} />
      </fieldset>
    </Sheet>
  );
}
