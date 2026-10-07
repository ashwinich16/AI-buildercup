import { useEffect, useRef, useState } from "react";
import Icon from "./Icons.jsx";

export default function Header({ shopper, cartCount, onOpenCart, onOpenSettings, onLeave }) {
  // A small bump on the cart count when something is added, so the change is noticed.
  const [bumped, setBumped] = useState(false);
  const previous = useRef(cartCount);
  useEffect(() => {
    const added = cartCount > previous.current;
    previous.current = cartCount;
    if (!added) return undefined;
    setBumped(true);
    const timer = setTimeout(() => setBumped(false), 400);
    return () => clearTimeout(timer);
  }, [cartCount]);

  const items = `${cartCount} ${cartCount === 1 ? "item" : "items"}`;

  return (
    <header className="header">
      <div className="header__brand">
        <span className="brand">
          <span className="brand__mark" aria-hidden="true" />
          Scout
        </span>
        {shopper.name && <span className="header__who muted">Shopping as {shopper.name}</span>}
      </div>

      <nav className="header__actions" aria-label="Cart and settings">
        <button className="icon-button" onClick={onOpenCart} aria-label={`Cart, ${items}`}>
          <Icon name="bag" />
          <span className="header__label" aria-hidden="true">
            Cart
          </span>
          {cartCount > 0 && (
            <span className={`icon-button__count ${bumped ? "is-bumped" : ""}`} aria-hidden="true">
              {cartCount}
            </span>
          )}
        </button>
        <button className="icon-button" onClick={onOpenSettings} aria-label="Settings">
          <Icon name="settings" />
          <span className="header__label" aria-hidden="true">
            Settings
          </span>
        </button>
        <button className="button button--ghost" onClick={onLeave}>
          {shopper.leave.label}
        </button>
      </nav>
    </header>
  );
}
