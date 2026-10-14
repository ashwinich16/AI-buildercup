"""
Recommender: one item that goes with what is in the cart ("a belt would go nicely with those trousers").

How it chooses, like a good shop assistant:
  1. Which KINDS go with the chosen item. A PairingAdvisor picks them from the shop's own kinds
     (catalog.product_types()), so nothing here names a product. Kinds already in the cart, or
     already suggested this conversation, are skipped.
  2. Which PRODUCTS: real catalog items of those kinds, for the same person, within what is left
     of the budget, never one already suggested, and never one that mentions something the
     shopper avoids.
  3. The BEST one: shares the style of what they chose, shares a fact (the same colour, the same
     material...), and is well rated by enough reviewers. The advisor's first kinds win ties.

The reasons it is a good match are returned with it, so Scout can say why in one sentence.
"""

import json
import logging
import math
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from google.genai import types
from pydantic import BaseModel, Field

from scout.catalog import CatalogRepository, SearchQuery
from scout.config import Settings
from scout.domain import Product, ShoppingNeed
from scout.tools.cart import Cart
from scout.tools.retrieval import has_all

logger = logging.getLogger(__name__)


class SuggestionRules(BaseModel):
    """What every suggestion must respect."""
    need: ShoppingNeed = Field(default_factory=ShoppingNeed)   # budget and whose style
    avoid: list[str] = Field(default_factory=list)             # things the shopper avoids
    suggested_ids: set[str] = Field(default_factory=set)       # never suggest a product twice
    suggested_kinds: set[str] = Field(default_factory=set)     # never offer the same kind twice
    occasion: str = ""                                         # what they are shopping for, if they said
    taste: str = ""                                            # the colours, materials and styles they keep choosing
    wardrobe: list[str] = Field(default_factory=list)          # what they already own, in words


class Suggestion(BaseModel):
    product: Product
    goes_with: str                                    # short name of the cart item it completes
    reasons: list[str] = Field(default_factory=list)  # facts Scout can say: "both formal", "matching navy"

    def to_prompt_text(self) -> str:
        why = "; ".join(self.reasons) or "it goes with what they chose"
        return (f"{self.product.to_prompt_text()}\n"
                f"Goes with: {self.goes_with} in their cart. Why it is a good match: {why}.")


# ---------------------------------------------------------------------
# Which kinds go together
# ---------------------------------------------------------------------

class PairingAdvisor(ABC):
    @abstractmethod
    def kinds_that_go_with(self, kind: str, shop_kinds: list[str]) -> list[str]:
        """Kinds from `shop_kinds` that complete `kind`, most essential first."""


class _Pairing(BaseModel):
    kinds: list[str] = Field(default_factory=list, max_length=4)


PAIRING_PROMPT = """A shopper has just chosen a {kind}.
The shop sells these kinds of product: {shop_kinds}.

Which kinds from that list would a good shop assistant suggest to go with it, most essential first?
Pick up to 3. Never pick a kind that would be used instead of a {kind} (it would replace it,
not complete it), and never the {kind} itself. Spell each kind exactly as listed."""


class GeminiPairingAdvisor(PairingAdvisor):
    """Asks Gemini once per kind and remembers the answer, in a file too: what goes with what does not change,
    so a restart (or every reload while developing) does not ask Gemini about all the shop's kinds again."""

    SAVED = Path(__file__).with_name("pairings.json")

    def __init__(self, settings: Settings):
        self._client = settings.genai_client()
        self._model = settings.model_for("recommender")
        tuning = settings.tuning("recommender")
        self._config = types.GenerateContentConfig(
            temperature=tuning.temperature,
            response_mime_type="application/json",
            response_schema=_Pairing,
            thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget),
        )
        self._known: dict[str, list[str]] = self._load()

    def _load(self) -> dict[str, list[str]]:
        try:
            return json.loads(self.SAVED.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        try:
            self.SAVED.write_text(json.dumps(self._known, indent=1, sort_keys=True), encoding="utf-8")
        except OSError as error:      # read-only deploys still work; they just ask again next time
            logger.info("Could not save pairings: %s", error)

    def prepare(self, shop_kinds: list[str]) -> None:
        """Ask about every kind the shop has, all at once, so adding to the cart never waits for Gemini
        (one call took about 9 seconds when made at add-to-cart time)."""
        missing = [k for k in shop_kinds if k not in self._known]
        if not missing:
            logger.info("Pairings for %d kinds loaded from %s.", len(self._known), self.SAVED.name)
            return

        def ask(kind: str) -> None:
            try:
                self.kinds_that_go_with(kind, shop_kinds)
            except Exception as error:   # one kind failing must not stop the others
                logger.warning("Could not prepare pairings for %s: %s", kind, error)

        with ThreadPoolExecutor(max_workers=3) as pool:     # gently: the shopper's own calls share the quota
            list(pool.map(ask, missing))
        self._save()
        logger.info("Prepared pairings for %d kinds.", len(self._known))

    def kinds_that_go_with(self, kind: str, shop_kinds: list[str]) -> list[str]:
        if kind not in self._known:
            prompt = PAIRING_PROMPT.format(kind=kind, shop_kinds=", ".join(shop_kinds))
            response = self._client.models.generate_content(model=self._model, contents=prompt, config=self._config)
            answer = _Pairing.model_validate(json.loads(response.text))
            spelled = {k.lower(): k for k in shop_kinds}
            self._known[kind] = [spelled[k.lower()] for k in answer.kinds
                                 if k.lower() in spelled and k.lower() != kind.lower()]
        return self._known[kind]


# ---------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------

class SuggestionPicker(ABC):
    @abstractmethod
    def pick(self, added: list[Product], rules: SuggestionRules, candidates: list[Suggestion],
             limit: int) -> Optional[list[Suggestion]]:
        """The best few candidates for this shopper, best first, each with a reason Scout can say; None when it
        could not decide (the scored order is used then)."""


class _Pick(BaseModel):
    product_id: str
    why: str = Field(description="Why it is a good match for them, in at most twelve plain words")


class _Picks(BaseModel):
    picks: list[_Pick] = Field(default_factory=list)


PICK_PROMPT = """You are a stylist suggesting what would go well with something a shopper just chose.

THEY JUST CHOSE
{added}

WHAT IT IS FOR: {occasion}
THEIR TASTE (what they keep choosing): {taste}
WHAT THEY ALREADY OWN:
{wardrobe}

CANDIDATES FROM THE SHOP
{candidates}

Pick up to {limit} candidates a good stylist would really suggest, best first, of different kinds where you can:
- They must suit what it is for (nothing casual or sporty for an office or a formal event, nothing formal for the
  beach) and go with what they just chose (colour, how dressed up it is).
- Prefer what fits their taste, and what also goes with clothes they already own.
- Never pick something they already own a very similar piece of.
- Pick none rather than something that does not fit.
For each, why: one short reason, from real details, that you could say aloud (e.g. "a crisp white shirt for the
office that goes with your navy blazer"). Use the ids exactly as given."""


class GeminiSuggestionPicker(SuggestionPicker):
    """One quick Gemini call. If Gemini is slow, adding to the cart does not wait: the scored order is used."""

    TIMEOUT_S = 8

    def __init__(self, settings: Settings):
        self._client = settings.genai_client(timeout_s=self.TIMEOUT_S)
        self._model = settings.model_for("picker")
        tuning = settings.tuning("picker")
        self._config = types.GenerateContentConfig(
            temperature=tuning.temperature,
            response_mime_type="application/json",
            response_schema=_Picks,
            thinking_config=types.ThinkingConfig(thinking_budget=tuning.thinking_budget),
        )

    def pick(self, added: list[Product], rules: SuggestionRules, candidates: list[Suggestion],
             limit: int) -> Optional[list[Suggestion]]:
        if not candidates:
            return []
        prompt = PICK_PROMPT.format(
            added="\n".join(f"- {_short(p)}" for p in added),
            occasion=rules.occasion or rules.need.purpose or "not said",
            taste=rules.taste or "not known yet",
            wardrobe="\n".join(f"- {item}" for item in rules.wardrobe) or "- nothing added yet",
            candidates="\n".join(f"[{s.product.id}] {_short(s.product)} (goes with their {s.goes_with})"
                                  for s in candidates),
            limit=limit,
        )
        try:
            response = self._client.models.generate_content(model=self._model, contents=prompt, config=self._config)
            picks = _Picks.model_validate(json.loads(response.text)).picks
        except Exception as error:      # a suggestion is a bonus: never hold up adding to the cart
            logger.warning("Could not pick suggestions: %s", error)
            return None
        by_id = {s.product.id: s for s in candidates}
        chosen = []
        for pick in picks:
            if pick.product_id in by_id and all(s.product.id != pick.product_id for s in chosen):
                chosen.append(by_id[pick.product_id].model_copy(update={"reasons": [pick.why] if pick.why else []}))
        return chosen[:limit]


def _short(product: Product) -> str:
    facts = [product.title[:90], product.product_type or ""]
    facts += [a.value for a in product.attributes[:4]] + product.style_tags[:3]
    if product.price is not None:
        facts.append(f"${product.price:.0f}")
    return "; ".join(f for f in facts if f)


class Recommender:
    def __init__(self, catalog: CatalogRepository, advisor: PairingAdvisor, candidates_per_kind: int = 8,
                 picker: Optional[SuggestionPicker] = None):
        self._catalog = catalog
        self._advisor = advisor
        self._picker = picker
        self._limit = candidates_per_kind
        self._products: dict[str, Product] = {}      # cart products by id, kept for the process

    def prepare(self) -> None:
        """Work out what goes with every kind in the shop ahead of time (called at server start-up)."""
        prepare = getattr(self._advisor, "prepare", None)
        if prepare:
            prepare(self._catalog.product_types())

    def for_new_item(self, added: Product, cart: Cart, rules: SuggestionRules) -> Optional[Suggestion]:
        """The best item to go with a product just added to the cart."""
        options = self.options_for_new_item(added, cart, rules, limit=1)
        return options[0] if options else None

    def options_for_new_item(self, added: Product, cart: Cart, rules: SuggestionRules,
                             limit: int = 4) -> list[Suggestion]:
        """A few items that go with a product just added, best first and of different kinds where possible
        (a belt, a t-shirt, sneakers), so the shopper can pick rather than take or leave one."""
        self._products[added.id] = added
        return self._ranked([added], cart, rules, limit)

    def for_cart(self, cart: Cart, rules: SuggestionRules) -> Optional[Suggestion]:
        """At checkout: anything that would complete the cart as a whole ("you might have forgotten...")."""
        anchors = [p for p in (self._product(item.product_id) for item in cart.items) if p]
        return self._best(anchors, cart, rules)

    def _best(self, anchors: list[Product], cart: Cart, rules: SuggestionRules) -> Optional[Suggestion]:
        ranked = self._ranked(anchors, cart, rules, limit=1)
        return ranked[0] if ranked else None

    def _ranked(self, anchors: list[Product], cart: Cart, rules: SuggestionRules, limit: int) -> list[Suggestion]:
        skip_ids = {item.product_id for item in cart.items} | rules.suggested_ids
        skip_kinds = {item.kind for item in cart.items if item.kind} | rules.suggested_kinds
        need = rules.need

        # (cart item, rank, kind) for every kind that would complete it, searched all at once.
        plans = [(anchor, rank, kind) for anchor in anchors
                 for rank, kind in enumerate(k for k in self._kinds_for(anchor) if k not in skip_kinds)]
        if not plans:
            return []
        # The budget they gave was for the item they were looking for ("shoes around $60"), not for what goes
        # with it; an extra simply never costs more than the item it goes with.
        with ThreadPoolExecutor(max_workers=min(len(plans), 8)) as pool:
            found = list(pool.map(lambda plan: self._candidates(plan[0], plan[2], plan[0].price, need), plans))

        scored: list[tuple[float, Suggestion]] = []
        for (anchor, rank, _kind), candidates in zip(plans, found):
            for candidate in candidates:
                if candidate.id in skip_ids or any(has_all(candidate, [term]) for term in rules.avoid if term):
                    continue
                score, reasons = _score(anchor, candidate)
                score -= 0.6 * rank                # the most essential kinds first
                scored.append((score, Suggestion(product=candidate, goes_with=anchor.kind or "item", reasons=reasons)))
        scored.sort(key=lambda item: item[0], reverse=True)

        if self._picker is not None:       # a stylist chooses among the best few of each kind
            shortlist: list[Suggestion] = []
            per_kind: dict[str, int] = {}
            for _score_value, suggestion in scored:
                kind = suggestion.product.kind or ""
                if per_kind.get(kind, 0) < 3 and all(s.product.id != suggestion.product.id for s in shortlist):
                    shortlist.append(suggestion)
                    per_kind[kind] = per_kind.get(kind, 0) + 1
            picked = self._picker.pick(anchors, rules, shortlist[:15], limit)
            if picked is not None:
                return picked

        # Best of each kind first (variety: a belt, a t-shirt, sneakers), then the next best of any kind.
        chosen: list[Suggestion] = []
        kinds_taken: set[str] = set()
        for _score_value, suggestion in scored:
            if len(chosen) < limit and suggestion.product.kind not in kinds_taken \
                    and all(s.product.id != suggestion.product.id for s in chosen):
                chosen.append(suggestion)
                kinds_taken.add(suggestion.product.kind)
        for _score_value, suggestion in scored:
            if len(chosen) < limit and all(s.product.id != suggestion.product.id for s in chosen):
                chosen.append(suggestion)
        return chosen

    def _kinds_for(self, anchor: Product) -> list[str]:
        kind = anchor.kind
        if not kind:
            return []
        try:
            return self._advisor.kinds_that_go_with(kind, self._catalog.product_types())
        except Exception as error:      # a suggestion is a bonus; it must never break adding to the cart
            logger.warning("Could not decide what goes with %s: %s", kind, error)
            return []

    def _candidates(self, anchor: Product, kind: str, max_price: Optional[float],
                    need: ShoppingNeed) -> list[Product]:
        gender = anchor.gender if anchor.gender in ("men", "women") else need.gender_filter()
        query = SearchQuery(text=" ".join([*anchor.style_tags[:3], kind]), product_types=[kind],
                            gender=gender, max_price=max_price, limit=self._limit)
        try:
            return self._catalog.search(query)
        except Exception as error:
            logger.warning("Suggestion search for %s failed: %s", kind, error)
            return []

    def _product(self, product_id: str) -> Optional[Product]:
        if product_id not in self._products:
            try:
                found = self._catalog.get_product(product_id)
            except Exception as error:
                logger.warning("Could not look up cart item %s: %s", product_id, error)
                return None
            if found is None:
                return None
            self._products[product_id] = found
        return self._products[product_id]


def _score(anchor: Product, candidate: Product) -> tuple[float, list[str]]:
    reasons = []
    shared_style = sorted({t.lower() for t in anchor.style_tags} & {t.lower() for t in candidate.style_tags})
    score = 1.5 * min(len(shared_style), 2)
    if shared_style:
        reasons.append(f"both {', '.join(shared_style[:2])}")

    # A fact both state the same way (a colour, a material) makes them look like they belong together.
    anchor_facts = {(a.name.lower(), a.value.lower()) for a in anchor.attributes}
    shared_facts = [a.value.lower() for a in candidate.attributes if (a.name.lower(), a.value.lower()) in anchor_facts]
    if shared_facts:
        score += 1.0
        reasons.append(f"matching {shared_facts[0]}")

    if candidate.rating is not None:
        # A good rating counts for more when many people gave it.
        confidence = min(1.0, math.log10((candidate.rating_count or 0) + 1) / 2)
        score += (candidate.rating - 3) * confidence
        if candidate.rating >= 4 and (candidate.rating_count or 0) >= 50:
            reasons.append(f"rated {candidate.rating:.1f} by {candidate.rating_count} people")
    if candidate.price is not None:
        score += 0.2                      # a known price is easier to decide on
    return score, reasons

