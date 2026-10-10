"""
Retrieval: turns a ShoppingNeed into a shortlist of products.

Plain Python, no LLM: the Orchestrator calls it directly, which is faster and
more predictable than letting a model decide how to search.

Exact matches come first. If there are only a few, it tops the list up with
related products from broader searches (details relaxed, budget kept), and
records which products are exact and what was relaxed for the rest, so Scout
can be honest ("I only have one glittery pair, but these flats would also
suit a wedding"). The budget is only relaxed when nothing at all fits.

Who the item is for (men's / women's) and the kind of product asked for
(a shirt, not any top) are never relaxed: showing a women's blouse to someone
who asked for a men's shirt is not "related", it is wrong. When those leave
only a few products, it lists the other kinds the shop has for the same
person instead, so Scout can OFFER them rather than slip them in.

It depends only on the CatalogRepository interface, so any catalog works.
"""

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Callable, Optional

from pydantic import BaseModel, Field

from scout.catalog import CatalogRepository, SearchQuery
from scout.domain import Product, ShoppingNeed


class RetrievalResult(BaseModel):
    products: list[Product] = Field(default_factory=list)    # exact matches first, then related ones
    exact_ids: list[str] = Field(default_factory=list)       # products that match everything asked for
    relaxed: list[str] = Field(default_factory=list)         # what was relaxed for the related ones, e.g. ["details"]
    other_types: list[str] = Field(default_factory=list)     # other kinds in stock for the same person, when few matched
    query_text: str = ""

    @property
    def found(self) -> bool:
        return bool(self.products)


class ProductRetriever:
    def __init__(self, catalog: CatalogRepository, limit: int = 8, target: int = 6, few: int = 3):
        self._catalog = catalog
        self._limit = limit      # products per search call
        self._target = target    # stop topping up once the shortlist has this many
        self._few = few          # below this many, offer the other kinds the shop has

    @staticmethod
    def fits(need: ShoppingNeed, product: Product) -> bool:
        """Does the product keep the need's firm limits (kind, whose style, price range)? Used to drop
        products found earlier once the shopper tightens the need ("under $70", "over $30")."""
        return SearchQuery(category=need.category, product_types=need.product_types, gender=need.gender_filter(),
                           min_price=need.budget_min, max_price=need.budget_max).matches(product)

    def find(self, need: ShoppingNeed, exclude_ids: Optional[set[str]] = None) -> RetrievalResult:
        """Exact matches first, topped up with related products. `exclude_ids` skips products already shown."""
        excluded = set(exclude_ids or ())
        in_catalog = self._recorded_in_catalog(need)   # does the catalog's data record a must-have? (asked lazily)
        collected: list[Product] = []
        exact_ids: list[str] = []
        relaxed_for_rest: list[str] = []

        within_budget: set[str] = set()
        attempts = self._attempts(need)
        # All searches at once: when little fits (nothing under the budget), waiting for each in turn was slow.
        with ThreadPoolExecutor(max_workers=len(attempts) or 1) as pool:
            found = list(pool.map(lambda attempt: self._catalog.search(attempt[1]), attempts))
        for (relaxed, _query), results in zip(attempts, found):
            if collected and "budget" in relaxed:
                break                          # never trade the budget away just to top up
            seen = excluded | {p.id for p in collected}
            # Some listings are the same product under another id (a size or colour variant with the same
            # title); presenting it twice wastes one of the three options.
            seen_titles = {_title_key(p.title) for p in collected}
            hits = []
            for p in self._without_avoided(results, need.avoid):
                if p.id not in seen and _title_key(p.title) not in seen_titles:
                    hits.append(p)
                    seen_titles.add(_title_key(p.title))
            if not hits:
                continue
            if relaxed:
                relaxed_for_rest = relaxed_for_rest or relaxed
            if "budget" not in relaxed:
                within_budget |= {p.id for p in hits}
            if not relaxed and not need.must_have:
                exact_ids += [p.id for p in hits]
            collected += hits
            # With must-haves (a colour, "stretchy"), keep looking: semantic search ranks, it does not check.
            if len(collected) >= self._target and (
                    not need.must_have or self._exact_count(collected, need, in_catalog) >= self._few):
                break

        if need.must_have:
            # Only must-haves the data can confirm or rule out count: if no product data ever states one,
            # it can't count against any product. Exact = states all checkable ones; best first.
            checkable = _checkable(need, collected, in_catalog)
            met = {p.id: sum(has_all(p, [t]) for t in checkable) for p in collected}
            exact_ids = [p.id for p in collected if p.id in within_budget and met[p.id] == len(checkable)]
            collected.sort(key=lambda p: (p.id not in exact_ids, -met[p.id]))   # stable: search order otherwise
            if len(exact_ids) < len(collected):
                relaxed_for_rest = relaxed_for_rest or ["details"]

        if "budget" in relaxed_for_rest:   # nothing within budget: the closest prices above it first
            collected.sort(key=lambda p: (p.price is None, p.price or 0))

        other_types = self._other_types(need) if need.product_types and len(collected) < self._few else []
        if not collected:
            return RetrievalResult(relaxed=["everything"], other_types=other_types, query_text=need.search_text())
        return RetrievalResult(products=collected[: self._limit], exact_ids=exact_ids, relaxed=relaxed_for_rest,
                               other_types=other_types, query_text=need.search_text())

    def _attempts(self, need: ShoppingNeed) -> list[tuple[list[str], SearchQuery]]:
        """Search plans from strictest to loosest.

        Budget is treated as a firm limit and details as nice-to-have, so details
        are relaxed first and the budget only as a last resort. A lower price limit ("over $30") is what
        they want, not a cost to save, so it is never relaxed.
        """
        full_text = need.search_text()
        short_text = need.looking_for or need.category or full_text
        broad_text = " ".join(p for p in [need.category or short_text, need.purpose or ""] if p)
        # An empty query with the filters lists every product that fits; searching for the bare word
        # "top" instead ranks by that word and can leave real matches out.
        category_only = "" if need.category else short_text
        budget = need.budget_max

        plans = [
            ([], full_text, budget),                        # everything the shopper said
            (["some details"], short_text, budget),         # just what they want, within budget
            (["details"], broad_text, budget),              # the category for the same purpose, within budget
            (["details"], category_only, budget),           # anything in the category, within budget
            (["budget"], full_text, None),                  # full need, any price
            (["budget", "details"], category_only, None),   # anything in the category
        ]

        attempts, seen = [], set()
        for relaxed, text, max_price in plans:
            key = (text, max_price)
            if (text or need.category) and key not in seen:  # skip duplicate searches
                seen.add(key)
                attempts.append((relaxed, SearchQuery(
                    text=text, category=need.category, product_types=need.product_types,
                    gender=need.gender_filter(), min_price=need.budget_min, max_price=max_price,
                    limit=self._limit)))
        return attempts

    @staticmethod
    def _exact_count(products: list[Product], need: ShoppingNeed, in_catalog: Callable[[str], bool]) -> int:
        checkable = _checkable(need, products, in_catalog)
        return sum(1 for p in products if has_all(p, checkable)) if checkable else len(products)

    def _recorded_in_catalog(self, need: ShoppingNeed) -> Callable[[str], bool]:
        """Does the catalog's data record this must-have somewhere: stated by a product in the same category, or
        a value the catalog records as an attribute of any product (a colour such as "black" even when no black
        pair of jeans exists)? The answer is kept for the whole search, so the search loop and the final
        scoring agree, and the catalog is only asked about must-haves no found product already states."""
        samples: list[list[Product]] = []

        @cache
        def recorded(term: str) -> bool:
            if not samples:
                samples.append(self._catalog.search(
                    SearchQuery(category=need.category, gender=need.gender_filter(), limit=50)))
            return any(has_all(p, [term]) for p in samples[0]) or self._recorded_value(term)

        return recorded

    def _recorded_value(self, term: str) -> bool:
        """Is this the value of some product's attribute anywhere in the catalog?"""
        stems = word_stems(term)
        return bool(stems) and any(stems <= word_stems(a.value)
                                   for p in self._catalog.search(SearchQuery(text=term, limit=20))
                                   for a in p.attributes)

    def _other_types(self, need: ShoppingNeed) -> list[str]:
        """Other kinds of product in the same category for the same person, most common first."""
        query = SearchQuery(text="" if need.category else need.search_text(), category=need.category,
                            gender=need.gender_filter(), min_price=need.budget_min, max_price=need.budget_max,
                            limit=50)
        wanted = {t.lower() for t in need.product_types}
        counts = Counter(p.product_type for p in self._catalog.search(query)
                         if p.product_type and p.product_type.lower() not in wanted)
        return [kind for kind, _ in counts.most_common(3)]

    @staticmethod
    def _without_avoided(products: list[Product], avoid: list[str]) -> list[Product]:
        """Drop products whose stated facts mention something the shopper wants to avoid.

        Checks attributes and style tags only (not titles), so "avoid heels" does not
        remove a shoe whose title mentions "heel pain relief". If everything would be
        removed, the original list is kept and the narrator can mention the conflict.
        """
        if not avoid:
            return products
        terms = [term.lower() for term in avoid]

        def mentions_avoided(product: Product) -> bool:
            facts = " ".join([a.value for a in product.attributes] + product.style_tags).lower()
            return any(term in facts for term in terms)

        kept = [p for p in products if not mentions_avoided(p)]
        return kept or products


def has_all(product: Product, must_have: list[str]) -> bool:
    """Do the product's own facts (title, type, attributes, style, description) mention every must-have?

    Word-level and forgiving about endings ("stretchy" matches "stretch"), so it checks what is stated
    rather than trusting the search ranking: a "black jeans" search must not call blue jeans exact.
    """
    facts = word_stems(" ".join([product.title, product.product_type or "", product.description or "",
                                  *product.style_tags,
                                  *(f"{a.name} {a.value}" for a in product.attributes)]))
    return all(word_stems(term) <= facts for term in must_have if word_stems(term))


# Words of the language that carry no product meaning (articles, prepositions, how people phrase a request),
# so they are never required to appear in product data. This is about English, not about any catalog.
_FILLER_WORDS = {"a", "an", "the", "and", "or", "but", "with", "of", "in", "on", "for", "to", "at", "my", "me",
                 "it", "is", "be", "that", "this", "very", "really", "some", "something", "i", "want", "need",
                 "looking", "like", "good"}


def _checkable(need: ShoppingNeed, products: list[Product], in_catalog: Callable[[str], bool]) -> list[str]:
    """Must-haves that can count against a product: stated by one of these products, or recorded in the catalog.
    Something no product data ever records (often a feeling such as "soft") can't count against any product."""
    return [term for term in need.must_have if any(has_all(p, [term]) for p in products) or in_catalog(term)]


def word_stems(text: str) -> set[str]:
    """The meaningful words of a text, lower-case and without common endings ("stretchy" -> "stretch"), for
    matching what a shopper says against what product data states. Used by every tool that matches words."""
    text = (text or "").lower()
    # Hyphenated words count whole as well as by their parts, so "v-neck" needs "v-neck", not just any "neck".
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text) + re.findall(r"[a-z0-9]+", text)
    stems = set()
    for word in words:
        if word in _FILLER_WORDS or (len(word) < 2 and not word.isdigit()):
            continue
        for suffix in ("iness", "ness", "able", "ing", "ed", "es", "y", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
        stems.add(word)
    return stems


def _title_key(title: str) -> str:
    """Titles compared without case, punctuation or bracketed notes such as a size or colour."""
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"[\(\[].*?[\)\]]", " ", (title or "").lower()))[:60]
