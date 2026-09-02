"""The public API: answers in, recommendations out.

This is the whole library from a caller's point of view. Everything below it
— the projection, the quantised vectors, the scoring blend — is detail.

Nothing here touches a database or the network. It reads one artifact from
disk and does arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote_plus

from . import questions as questionnaire
from .model import ModelUnavailable, Query, Scored, TasteModel, explain, load_model, score

MAX_FREE_TEXT = 300

# Clicking a result searches for the title rather than landing on one site's
# page for it. A search finds the trailer, where it streams, and what people
# said about it — a database entry finds one of those.
SEARCH_URL = "https://duckduckgo.com/?q={query}"
ANILIST_URL = "https://anilist.co/anime/{id}"


class InvalidAnswer(ValueError):
    """The caller sent something the questionnaire does not define."""


@dataclass(frozen=True)
class Recommendation:
    anilist_id: int
    name: str
    year: int | None
    episodes: int | None
    poster: str
    colour: str
    community_score: int | None
    match: int
    reasons: tuple[str, ...]
    neighbours: tuple[str, ...]

    @property
    def url(self) -> str:
        """Where a click goes: a web search for the title."""
        return SEARCH_URL.format(query=quote_plus(self.name))

    @property
    def anilist_url(self) -> str:
        """The entry the model learned this title from."""
        return ANILIST_URL.format(id=self.anilist_id)

    def as_dict(self) -> dict:
        return {
            "id": self.anilist_id,
            "name": self.name,
            "year": self.year,
            "episodes": self.episodes,
            "poster": self.poster,
            "colour": self.colour,
            "score": self.community_score,
            "match": self.match,
            "reasons": list(self.reasons),
            "neighbours": list(self.neighbours),
            "url": self.url,
            "anilistUrl": self.anilist_url,
        }


@dataclass
class Ask:
    """One request. Answers are `{question key: [option values]}`."""

    answers: dict[str, list[str]] = field(default_factory=dict)
    text: str = ""
    # Titles the caller already knows about, by AniList id.
    exclude: frozenset[int] = frozenset()
    limit: int = 9


def build_query(ask: Ask, model: TasteModel) -> Query:
    """Validate the answers and express them as model terms.

    Unknown questions and unknown options are rejected rather than ignored:
    silently dropping half a request and returning confident recommendations
    for the other half is worse than an error.
    """
    query = Query()

    for key, values in (ask.answers or {}).items():
        question = questionnaire.QUESTIONS_BY_KEY.get(key)
        if question is None:
            raise InvalidAnswer(f"There is no question called {key!r}.")

        chosen = [value for value in (values or []) if value]
        if not chosen:
            if question.optional:
                continue
            raise InvalidAnswer(f"Pick an answer for {question.prompt!r}.")
        if len(chosen) > question.max_choices:
            raise InvalidAnswer(
                f"{question.prompt!r} takes at most {question.max_choices} "
                f"answer{'s' if question.max_choices > 1 else ''}."
            )

        for value in chosen:
            option = questionnaire.option_for(key, value)
            if option is None:
                raise InvalidAnswer(f"{value!r} is not an answer to {question.prompt!r}.")
            query.add_terms(option.terms)
            query.add_avoid(option.avoid)
            if option.facet:
                facet, facet_value = option.facet
                query.facets[facet] = facet_value

    text = (ask.text or "").strip()[:MAX_FREE_TEXT]
    if text:
        query.add_free_text(text, model)

    query.seen_ids = frozenset(ask.exclude)

    if not query.terms and not query.facets and not query.references and not query.avoid:
        if text:
            raise InvalidAnswer(
                "None of those words are ones the model knows. Try a mood, a "
                "genre, or the name of an anime you liked."
            )
        raise InvalidAnswer("Answer at least one question so there is something to go on.")

    return query


def recommend(ask: Ask, *, model: TasteModel | None = None) -> list[Recommendation]:
    """Rank the catalogue against one request."""
    model = model or load_model()
    query = build_query(ask, model)
    return [_present(scored) for scored in score(model, query, limit=ask.limit)]


def _present(scored: Scored) -> Recommendation:
    item = scored.item
    return Recommendation(
        anilist_id=item.anilist_id,
        name=item.name,
        year=item.year,
        episodes=item.episodes,
        poster=item.cover,
        colour=item.colour or "#1a2440",
        community_score=item.community_score,
        match=int(round(scored.fit * 100)),
        reasons=tuple(explain(scored)),
        neighbours=item.neighbours[:2],
    )


def questions() -> list[dict]:
    """The questionnaire, as a client should render it."""
    return questionnaire.as_payload()


def status() -> dict:
    """Whether there is a usable model, and what it knows."""
    try:
        model = load_model()
    except ModelUnavailable as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "titles": len(model.items),
        "trainedAt": model.trained_at,
        "metrics": model.metrics,
    }
