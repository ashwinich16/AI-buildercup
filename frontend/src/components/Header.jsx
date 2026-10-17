import { useEffect, useRef, useState } from "react";
import Icon from "./Icons.jsx";

export default function Header({
  shopper, cartCount, wishlistCount, onOpenCart, onOpenWishlist, onOpenWardrobe, onClearChat, onOpenSettings, onLeave,
}) {
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
        <span className="header__tagline muted">Conversational styling companion who knows your style</span>
        {/* {shopper.name && <span className="header__who muted">Shopping as {shopper.name}</span>} */}
      </div>

      <nav className="header__actions" aria-label="Conversation, wardrobe, wishlist, cart and settings">
        <button className="icon-button" onClick={onClearChat} aria-label="Clear the conversation and start again">
          <Icon name="reset" />
          <span className="header__label" aria-hidden="true">
            New chat
          </span>
        </button>
        <button className="icon-button" onClick={onOpenWardrobe} aria-label="Your wardrobe">
          <Icon name="hanger" />
          <span className="header__label" aria-hidden="true">
            Wardrobe
          </span>
        </button>
        <button
          className="icon-button"
          onClick={onOpenWishlist}
          aria-label={`Wishlist, ${wishlistCount} saved`}
        >
          <Icon name="heart" />
          <span className="header__label" aria-hidden="true">
            Wishlist
          </span>
          {wishlistCount > 0 && (
            <span className="icon-button__count" aria-hidden="true">
              {wishlistCount}
            </span>
          )}
        </button>
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
