// How values from the API are shown on screen, in one place.

export const formatPrice = (value) => (value == null ? "Price not listed" : `$${value.toFixed(2)}`);
