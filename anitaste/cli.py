"""Command line: harvest a corpus, train a model, ask it something.

    anitaste harvest --by-year
    anitaste train
    anitaste ask "something like Cowboy Bebop"
    anitaste status

`harvest` and `train` are build steps and need the extras in
requirements-ml.txt. `ask` and `status` need nothing but the standard library.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "anilist_corpus.json.gz"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "data" / "taste_model.json.gz"
DEFAULT_MAX_TITLES = 9000


def _harvest(args: argparse.Namespace) -> int:
    from .ml.harvest import PAGE_SIZE, HarvestError, harvest, harvest_by_year, write_corpus

    started = time.monotonic()
    if args.by_year:
        last = args.last_year
        print(
            f"Harvesting {args.first_year}-{last}, up to {args.pages_per_year} "
            "pages a year. AniList caps one sorted query at 5,000 results, "
            "which is why this goes year by year."
        )

        def progress(index, total, year, count):
            print(f"  {year}  ({index}/{total} years, {count} titles)", flush=True)

        stream = harvest_by_year(
            first_year=args.first_year,
            last_year=last,
            pages_per_year=args.pages_per_year,
            progress=progress,
        )
    else:
        print(f"Harvesting up to {args.pages * PAGE_SIZE} titles…")

        def progress(page, total):
            if page % 5 == 0 or page == total:
                print(f"  page {page}/{total}", flush=True)

        stream = harvest(pages=args.pages, progress=progress)

    titles = []
    try:
        for title in stream:
            titles.append(title)
    except HarvestError as error:
        if len(titles) < 500:
            print(f"error: {error}, and only {len(titles)} titles in hand", file=sys.stderr)
            return 1
        print(f"warning: stopped early ({error}); keeping {len(titles)} titles")
    except KeyboardInterrupt:
        if not titles:
            raise
        print(f"\ninterrupted; keeping {len(titles)} titles")

    write_corpus(titles, args.out)
    edges = sum(len(title.recommendations) for title in titles)
    print(
        f"Wrote {len(titles)} titles and {edges} recommendation edges to "
        f"{args.out} in {time.monotonic() - started:.0f}s."
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        print(
            "error: training needs numpy — pip install -r requirements-ml.txt.\n"
            "Serving does not; it reads the artifact this writes.",
            file=sys.stderr,
        )
        return 1

    from .ml.export import build_reference_titles, build_title_entries, write_model
    from .ml.harvest import read_corpus
    from .ml.train import train
    from .questions import all_terms

    if not args.corpus.exists():
        print(f"error: no corpus at {args.corpus}. Run `anitaste harvest` first.", file=sys.stderr)
        return 1

    started = time.monotonic()
    print(f"Reading {args.corpus}…")
    corpus = read_corpus(args.corpus)
    print(f"  {len(corpus)} titles in the corpus")

    if args.max_titles and len(corpus) > args.max_titles:
        corpus = sorted(corpus, key=lambda title: -title.popularity)[: args.max_titles]
        print(f"  training on the {args.max_titles} most popular")

    print("Training…")
    model = train(corpus, log=print)

    # Every answer must be a term the model knows. Without this a renamed tag
    # turns an answer into a no-op that still looks like it works.
    missing = sorted(term for term in all_terms() if term not in set(model.vocabulary.terms))
    if missing:
        print(
            "error: these answers name terms the trained vocabulary does not "
            "contain, so choosing them would change nothing:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        return 1
    print("  every questionnaire term is in the vocabulary")

    titles = build_title_entries(model=model, corpus=corpus)
    references = build_reference_titles(model=model, corpus=corpus)
    print(f"Shipping {len(titles)} titles and {len(references)} recognisable names.")

    write_model(
        model=model,
        titles=titles,
        references=references,
        path=args.out,
        trained_from=args.corpus.name,
    )
    size = args.out.stat().st_size / 1_048_576
    print(f"Wrote {args.out} ({size:.2f} MB) in {time.monotonic() - started:.0f}s.")
    return 0


def _ask(args: argparse.Namespace) -> int:
    from .model import ModelUnavailable
    from .recommend import Ask, InvalidAnswer, recommend

    try:
        results = recommend(Ask(text=" ".join(args.text), limit=args.limit))
    except ModelUnavailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except InvalidAnswer as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for item in results:
        reasons = ", ".join(item.reasons)
        print(f"{item.match:>3}%  {item.name}")
        if reasons:
            print(f"      {reasons}")
        print(f"      {item.url}")
    return 0


def _status(args: argparse.Namespace) -> int:
    from .recommend import status

    report = status()
    if not report["available"]:
        print(report["reason"], file=sys.stderr)
        return 1
    print(f"{report['titles']} titles, trained {report['trainedAt']}")
    for key, value in sorted(report["metrics"].items()):
        print(f"  {key:<26} {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anitaste", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    harvest = sub.add_parser("harvest", help="download the training corpus from AniList")
    harvest.add_argument("--by-year", action="store_true",
                         help="go year by year, to get past AniList's 5,000-result cap")
    harvest.add_argument("--pages", type=int, default=100)
    harvest.add_argument("--first-year", type=int, default=1980)
    harvest.add_argument("--last-year", type=int, default=time.gmtime().tm_year)
    harvest.add_argument("--pages-per-year", type=int, default=8)
    harvest.add_argument("--out", type=Path, default=DEFAULT_CORPUS)
    harvest.set_defaults(run=_harvest)

    train = sub.add_parser("train", help="fit the model and write the artifact")
    train.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    train.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--max-titles", type=int, default=DEFAULT_MAX_TITLES)
    train.set_defaults(run=_train)

    ask = sub.add_parser("ask", help="get recommendations for a sentence")
    ask.add_argument("text", nargs="+")
    ask.add_argument("--limit", type=int, default=6)
    ask.set_defaults(run=_ask)

    status = sub.add_parser("status", help="what the trained model knows")
    status.set_defaults(run=_status)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
