"""Export the trained model as the artifact the server reads.

Every title the model learned a vector for ships: the vectors themselves,
quantised, the projection matrix, and the metadata a result card needs.
"""

from __future__ import annotations

import base64
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..questions import FACET_ERA, FACET_LENGTH
from ..text import era_facet, length_facet, normalise_phrase, normalise_term
from .harvest import HarvestedTitle
from .train import TrainedModel

ARTIFACT_VERSION = 3
FLOAT_PRECISION = 4

# Vectors ship as signed bytes, not as JSON numbers.
#
# Written out as text, fifteen thousand 96-dimension vectors and a projection
# matrix the same size come to 21 MB and take over a second to parse. Every
# one of those numbers is a coordinate in a unit-length vector being compared
# by cosine, where the eighth decimal place has never once changed a ranking.
# Quantised to int8 with a scale factor the same data is 2.4 MB and loads with
# a single frombytes, and the largest coordinate error it can introduce is
# under 0.004 — two orders of magnitude below the gap between adjacent
# results.
QUANTISE_SCALE = 127.0

# Reference names and the sentences typed at them go through one normaliser,
# so they cannot disagree about what "Attack on Titan" looks like.
_normalise_name = normalise_phrase


def _tokens(name: str) -> set[str]:
    return set(_normalise_name(name).split())


def _round(values) -> list[float]:
    return [round(float(value), FLOAT_PRECISION) for value in values]


def _quantise(matrix: np.ndarray) -> tuple[str, list[float]]:
    """Pack a matrix of row vectors as base64 int8, one scale factor per row.

    Per row rather than one scale for the whole matrix: the projection's rows
    differ in magnitude by orders of magnitude, and a single shared scale
    would flatten the small ones to zero.
    """
    rows = np.asarray(matrix, dtype=np.float32)
    scales = np.abs(rows).max(axis=1)
    safe = np.where(scales > 0, scales, 1.0)
    quantised = np.rint(rows / safe[:, None] * QUANTISE_SCALE)
    packed = np.clip(quantised, -127, 127).astype(np.int8)
    return (
        base64.b64encode(packed.tobytes()).decode("ascii"),
        [round(float(scale), 6) for scale in scales],
    )


def build_title_entries(*, model: TrainedModel, corpus: list[HarvestedTitle]) -> list[dict]:
    """One artifact entry per title the model learned a vector for.

    Everything in the graph ships, not just the titles ANIFLIX carries. A
    recommender that can only answer with the nine things on the shelf is a
    filter, not a recommendation; whether we happen to stock a title is a
    question for the request, and the request answers it by looking the
    AniList id up in our own database.
    """
    by_id = {title.anilist_id: title for title in corpus}
    entries: list[dict] = []

    for position, anilist_id in enumerate(model.corpus_ids):
        title = by_id.get(anilist_id)
        if title is None:
            continue
        entries.append(
            {
                "id": anilist_id,
                "name": title.english or title.romaji,
                "year": title.year,
                "episodes": title.episodes,
                "score": title.score,
                "popularity": title.popularity,
                "cover": title.cover_url,
                "colour": title.cover_colour,
                "terms": _defining_terms(title),
                "neighbours": _neighbours(title, corpus, limit=2),
                "row": position,
                "facets": {
                    FACET_LENGTH: _facet_value(length_facet(title.episodes)),
                    FACET_ERA: _facet_value(era_facet(title.year)),
                },
            }
        )

    return entries


def _facet_value(facet: str | None) -> str | None:
    return facet.split(":")[-1] if facet else None


def _defining_terms(title: HarvestedTitle, *, limit: int = 12) -> list[str]:
    """The terms that describe a title, for matching and for explaining.

    The length and era facets are added *after* the cap, not before it. They
    used to be appended to the list and truncated away on any title with a
    dozen strong tags, which silently disarmed "not too long" — the refusal
    had nothing to match against on exactly the long-running shows it was
    meant to exclude.
    """
    subject = [normalise_term("genre", genre) for genre in title.genres]
    ranked = sorted(title.tags, key=lambda tag: -tag[1])
    subject += [normalise_term("tag", name) for name, rank in ranked if rank >= 60]

    terms = [term for term in dict.fromkeys(subject) if term][:limit]
    terms += [
        facet
        for facet in (era_facet(title.year), length_facet(title.episodes))
        if facet
    ]
    return list(dict.fromkeys(terms))


def _neighbours(
    title: HarvestedTitle, corpus: list[HarvestedTitle], *, limit: int
) -> list[str]:
    """The titles the community most often recommends alongside this one.

    Flavour for the explanation, not an input to the ranking.
    """
    names = {other.anilist_id: (other.english or other.romaji) for other in corpus}
    ordered = sorted(title.recommendations, key=lambda edge: -edge[1])
    out: list[str] = []
    for target_id, votes in ordered:
        if votes <= 0:
            break
        name = names.get(target_id)
        if name:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def facet_of(entry: dict) -> dict[str, str | None]:
    """The metadata facets a stated preference is checked against."""
    return {
        FACET_LENGTH: (length_facet(entry.get("episodes")) or "").split(":")[-1] or None,
        FACET_ERA: (era_facet(entry.get("year")) or "").split(":")[-1] or None,
    }


# How many of the best-known titles ship with their own vector, so that
# "something like Cowboy Bebop" is answered by that title's actual place in
# the recommendation graph instead of by the words in its name.
REFERENCE_TITLES = 800


def build_reference_titles(
    *, model: TrainedModel, corpus: list[HarvestedTitle]
) -> dict[str, dict]:
    """Normalised title name -> its learned vector, for the popular titles.

    Capped by popularity: the point is to recognise the titles a viewer would
    name, and nobody asks for "something like" a show with 200 viewers. Names
    shorter than four characters are dropped — "Air" and "Gto" match inside
    ordinary sentences and would hijack the query.
    """
    index = {anilist_id: i for i, anilist_id in enumerate(model.corpus_ids)}
    ranked = sorted(
        (title for title in corpus if title.anilist_id in index),
        key=lambda title: -title.popularity,
    )[:REFERENCE_TITLES]

    table: dict[str, dict] = {}
    for title in ranked:
        vector = _round(model.item_embedding[index[title.anilist_id]])
        for name in (title.english, title.romaji):
            key = _normalise_name(name)
            if len(key) < 4 or key in table:
                continue
            table[key] = {"label": title.english or title.romaji, "vector": vector}
    return table


def write_model(
    *,
    model: TrainedModel,
    titles: list[dict],
    references: dict[str, dict],
    path: Path,
    trained_from: str,
) -> Path:
    """Write the runtime artifact.

    The vocabulary and its IDF, the projection matrix, and one vector per
    title. The co-occurrence graph, the corpus and numpy all stay behind in
    the build.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Only the rows actually exported, in the order the titles reference.
    rows = [entry.pop("row") for entry in titles]
    vectors, vector_scales = _quantise(model.item_embedding[rows])
    projection, projection_scales = _quantise(model.projection)

    payload = {
        "version": ARTIFACT_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_from": trained_from,
        "dimensions": int(model.projection.shape[1]),
        "quantise_scale": QUANTISE_SCALE,
        "metrics": model.metrics,
        "vocabulary": model.vocabulary.terms,
        "idf": [round(value, FLOAT_PRECISION) for value in model.vocabulary.idf],
        "projection": projection,
        "projection_scales": projection_scales,
        "vectors": vectors,
        "vector_scales": vector_scales,
        "titles": titles,
        "reference_titles": references,
    }

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path
