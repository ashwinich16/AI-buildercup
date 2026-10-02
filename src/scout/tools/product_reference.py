"""
ProductReferenceResolver: turns the position the Elicitation agent picked
("the second one", "the cheaper one", "it" -> 2) into the actual product.

Understanding the shopper's words is the Elicitation agent's job (any phrasing,
any language, no word lists here). This class only maps the number to a
product, and returns None when it is not clear, so Scout asks instead of
guessing.
"""

from typing import Optional

from scout.domain import Product


class ProductReferenceResolver:
    def resolve(self, position: Optional[int], presented: list[Product]) -> Optional[Product]:
        if position is not None and 1 <= position <= len(presented):
            return presented[position - 1]
        return None
