"""
Sizes: which size scale a product is chosen in, and reading a size from what the shopper said.

The catalog does not list sizes, so every product is taken to come in the usual sizes for its kind,
using the conventions shoppers in India know: UK shoe sizes, waist sizes in inches for jeans and
trousers, and letter sizes for tops and jackets. Bags, watches, hats and the like need no size.
"""
import re
from typing import Optional

from pydantic import BaseModel

from scout.domain.product import Product


class SizeScale(BaseModel):
    key: str                 # what the shopper's size is remembered by, e.g. "shoe" (one size for all their shoes)
    spoken: str              # how Scout describes the scale, e.g. "UK shoe sizes 6 to 12"
    options: list[str]       # every size, smallest first, spelled as shown in the cart

    def label(self, size: str) -> str:
        """The size as shown and said, e.g. "UK 9", "32 waist", "M"."""
        return {"shoe": f"UK {size}", "waist": f"{size} waist"}.get(self.key, size)


_LETTERS = ["XS", "S", "M", "L", "XL", "XXL"]
_TOPS = {"shirt", "top", "jacket"}
_BOTTOMS = {"jeans", "trousers"}
_SIZED_ACCESSORIES = {"belt"}


def size_scale(product: Product) -> Optional[SizeScale]:
    """The scale this product is chosen in, or None when it needs no size."""
    category = (product.category or "").lower()
    kind = (product.product_type or "").lower()
    women = product.gender == "women"
    men = product.gender == "men"
    if category == "shoe":
        low, high = (3, 8) if women else (6, 12) if men else (3, 12)
        return SizeScale(key="shoe", spoken=f"UK shoe sizes {low} to {high}",
                         options=[str(n) for n in range(low, high + 1)])
    if category in _BOTTOMS:
        low, high = (26, 36) if women else (28, 40)
        return SizeScale(key="waist", spoken=f"waist sizes {low} to {high} inches",
                         options=[str(n) for n in range(low, high + 1, 2)])
    if category in _TOPS:
        return SizeScale(key="top", spoken="sizes XS to XXL", options=_LETTERS)
    if kind in _SIZED_ACCESSORIES:
        return SizeScale(key="belt", spoken="belt sizes S to XL", options=_LETTERS[1:5])
    return None


def scale_of_label(label: str) -> SizeScale:
    """The scale a size shown in the cart belongs to ("UK 9", "32 waist", "M"), to change it later."""
    if label.startswith("UK "):
        return SizeScale(key="shoe", spoken="UK shoe sizes 3 to 12", options=[str(n) for n in range(3, 13)])
    if label.endswith(" waist"):
        return SizeScale(key="waist", spoken="waist sizes 26 to 40 inches", options=[str(n) for n in range(26, 41, 2)])
    return SizeScale(key="top", spoken="sizes XS to XXL", options=_LETTERS)


_WORDS = {
    "extra extra large": "XXL", "double extra large": "XXL", "double xl": "XXL", "2xl": "XXL", "xxl": "XXL",
    "extra large": "XL", "xl": "XL", "extra small": "XS", "xs": "XS",
    "large": "L", "medium": "M", "small": "S",
}
_NUMBER_WORDS = {
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "twenty six": 26, "twenty eight": 28, "thirty": 30, "thirty two": 32, "thirty four": 34,
    "thirty six": 36, "thirty eight": 38, "forty": 40,
}


def read_size(text: str, scale: SizeScale) -> Optional[str]:
    """The size in `text` if it is one of the scale's sizes ("medium" -> "M", "size ten" -> "10"), else None."""
    said = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
    if scale.options == _LETTERS or scale.key == "belt":
        for words, size in _WORDS.items():                     
            if f" {words} " in said:
                return size if size in scale.options else None
        if len(said.split()) > 3:         
            return None
        letter = re.search(r" (xs|s|m|l|xl|xxl) ", said)
        return letter.group(1).upper() if letter and letter.group(1).upper() in scale.options else None
    for words, number in sorted(_NUMBER_WORDS.items(), key=lambda item: -len(item[0])):
        said = said.replace(f" {words} ", f" {number} ")
    for number in re.findall(r"\d+", said):
        if number in scale.options:
            return number
    return None
