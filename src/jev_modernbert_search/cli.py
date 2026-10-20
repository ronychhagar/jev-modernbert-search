from __future__ import annotations

import argparse
import asyncio
import csv
import shutil
from pathlib import Path

from .data import download_quora_competition, iter_pair_rows, reservoir_sample, write_sample
from .graph import Neo4jProductGraph
from .index import load_or_build_index
from .pim import Product, load_products
from .search import DEFAULT_MODEL, ModernBertSearch, SearchResult, rerank_with_jev

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "quora_sample.csv"
DEFAULT_INDEX = ROOT / "data" / "question_index.npz"
DEFAULT_PRODUCTS = ROOT / "data" / "products.csv"
DEFAULT_PRODUCT_INDEX = ROOT / "data" / "product_index.npz"
SAMPLE_PRODUCTS = ROOT / "examples" / "products.csv"


def _read_documents(path: Path) -> list[str]:
    documents: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        expected = {"question1", "question2", "is_duplicate"}
        missing = expected.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Sample is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            for field in ("question1", "question2"):
                text = (row.get(field) or "").strip()
                if text:
                    documents.setdefault(text.casefold(), text)
    if not documents:
        raise ValueError(f"No questions found in sample data: {path}")
    return list(documents.values())


def _prepare(args: argparse.Namespace) -> None:
    source = args.csv if args.csv else download_quora_competition()
    sample = reservoir_sample(iter_pair_rows(source), args.sample_size, args.seed)
    count = write_sample(sample, args.output)
    if not count:
        raise ValueError("No valid question pairs were found.")
    print(f"Wrote {count:,} sampled question pairs to {args.output}")


async def _search(args: argparse.Namespace) -> None:
    documents = _read_documents(args.data)
    model = ModernBertSearch(args.model)
    embeddings = load_or_build_index(model, args.model, documents, args.index)
    results = model.retrieve(args.query, documents, embeddings, limit=args.candidates)
    if not args.embedding_only:
        results = await rerank_with_jev(args.query, results)

    for rank, result in enumerate(results[: args.top_k], start=1):
        if result.jev_score is None:
            print(f"{rank:>2}. [{result.embedding_score:.3f}] {result.text}")
        else:
            print(
                f"{rank:>2}. [Jev {result.jev_score:.3f} | "
                f"ModernBERT {result.embedding_score:.3f}] {result.text}"
            )


def _pim_sample(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SAMPLE_PRODUCTS, args.output)
    print(f"Copied {SAMPLE_PRODUCTS} to {args.output}")


def _pim_download(args: argparse.Namespace) -> None:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before downloading Kaggle data.") from exc

    downloaded = Path(kagglehub.dataset_download(args.handle, path=args.file))
    if downloaded.is_dir():
        csv_files = sorted(downloaded.rglob("*.csv"))
        if len(csv_files) != 1:
            names = ", ".join(str(path) for path in csv_files) or "none"
            raise ValueError(
                "Kaggle download must resolve to one CSV; specify --file to select one. "
                f"CSV files found: {names}"
            )
        downloaded = csv_files[0]
    if downloaded.suffix.casefold() != ".csv":
        raise ValueError(f"Expected a Kaggle CSV file, received: {downloaded}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, args.output)
    print(f"Downloaded Kaggle data to {args.output}")


def _pim_import(args: argparse.Namespace) -> None:
    products = load_products(args.csv, limit=args.limit)
    with Neo4jProductGraph.from_env() as graph:
        graph.upsert_products(products, batch_size=args.batch_size)
    print(f"Upserted {len(products):,} products and ontology relationships into Neo4j.")


async def _pim_search(args: argparse.Namespace) -> None:
    products = load_products(args.csv, limit=args.max_products)
    if not args.query.strip():
        raise ValueError("Search query must not be empty.")
    model = ModernBertSearch(args.model)
    documents = [product.to_search_text() for product in products]
    embeddings = load_or_build_index(model, args.model, documents, args.index)
    results = model.retrieve(args.query, documents, embeddings, limit=args.candidates)
    products_by_text = {product.to_search_text(): product for product in products}
    selected_products = [products_by_text[result.text] for result in results]

    with Neo4jProductGraph.from_env() as graph:
        contexts_by_id = graph.product_context(
            product.product_id for product in selected_products
        )

    contexts = [contexts_by_id.get(product.product_id, "") for product in selected_products]
    if not args.embedding_only:
        results = await rerank_with_jev(args.query, results, contexts)

    for rank, result in enumerate(results[: args.top_k], start=1):
        product = products_by_text[result.text]
        if result.jev_score is None:
            score_text = f"ModernBERT {result.embedding_score:.3f}"
        else:
            score_text = (
                f"Jev {result.jev_score:.3f} | ModernBERT {result.embedding_score:.3f}"
            )
        print(f"{rank:>2}. [{score_text}] {product.name} ({product.product_id})")
        print(f"    Category: {' > '.join(product.categories) or 'Uncategorized'}")
        if product.brand:
            print(f"    Brand: {product.brand}")
        if product.price is not None:
            print(f"    Price: {product.price:g}")
        if product.description:
            print(f"    {product.description}")
        if product.attributes:
            values = ", ".join(
                f"{key}={value}" for key, value in sorted(product.attributes.items())
            )
            print(f"    Attributes: {values}")
        context = contexts_by_id.get(product.product_id)
        if context:
            print(f"    Ontology: {context}")

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jev-search",
        description="ModernBERT semantic retrieval with TypeSafe Jev reranking.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Download and sample Kaggle Quora data.")
    prepare.add_argument("--csv", type=Path, help="Use a previously downloaded train.csv or ZIP.")
    prepare.add_argument("--output", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--sample-size", type=int, default=10_000)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(func=_prepare)

    search = commands.add_parser("search", help="Search the prepared question sample.")
    search.add_argument("query")
    search.add_argument("--data", type=Path, default=DEFAULT_DATA)
    search.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search.add_argument("--model", default=DEFAULT_MODEL)
    search.add_argument("--candidates", type=int, default=20)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument(
        "--embedding-only",
        action="store_true",
        help="Skip Jev and return ModernBERT retrieval results only.",
    )
    search.set_defaults(func=lambda args: asyncio.run(_search(args)))

    pim = commands.add_parser("pim", help="Manage and search the product information graph.")
    pim_commands = pim.add_subparsers(dest="pim_command", required=True)

    pim_sample = pim_commands.add_parser("sample", help="Copy the included product sample CSV.")
    pim_sample.add_argument("--output", type=Path, default=DEFAULT_PRODUCTS)
    pim_sample.set_defaults(func=_pim_sample)

    pim_download = pim_commands.add_parser(
        "download", help="Download a product CSV from a Kaggle dataset."
    )
    pim_download.add_argument("--handle", required=True, help="Kaggle handle, owner/dataset.")
    pim_download.add_argument("--file", help="CSV path within the Kaggle dataset.")
    pim_download.add_argument("--output", type=Path, default=DEFAULT_PRODUCTS)
    pim_download.set_defaults(func=_pim_download)

    pim_import = pim_commands.add_parser(
        "import", help="Upsert products and ontology relationships into Neo4j."
    )
    pim_import.add_argument("--csv", type=Path, default=DEFAULT_PRODUCTS)
    pim_import.add_argument("--batch-size", type=int, default=250)
    pim_import.add_argument("--limit", type=int, default=10_000)
    pim_import.set_defaults(func=_pim_import)

    pim_search = pim_commands.add_parser(
        "search", help="Search products with ontology-aware ModernBERT and Jev."
    )
    pim_search.add_argument("query")
    pim_search.add_argument("--csv", type=Path, default=DEFAULT_PRODUCTS)
    pim_search.add_argument("--index", type=Path, default=DEFAULT_PRODUCT_INDEX)
    pim_search.add_argument("--model", default=DEFAULT_MODEL)
    pim_search.add_argument("--candidates", type=int, default=20)
    pim_search.add_argument("--top-k", type=int, default=5)
    pim_search.add_argument("--max-products", type=int, default=10_000)
    pim_search.add_argument(
        "--embedding-only",
        action="store_true",
        help="Skip Jev reranking but keep ModernBERT and Neo4j ontology context.",
    )
    pim_search.set_defaults(func=lambda args: asyncio.run(_pim_search(args)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
