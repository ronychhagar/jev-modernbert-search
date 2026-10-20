# JEv + ModernBERT Search

Semantic search with a ModernBERT-based retrieval model and optional
[TypeSafe Jev](https://typesafe.ai) reranking. The project includes both the
original question-pair demo and a product-information-management (PIM) demo
backed by a Neo4j knowledge graph and product ontology.

## How it works

1. `Alibaba-NLP/gte-modernbert-base` embeds questions and retrieves the closest
   candidates by cosine similarity. This is an embedding model built on
   ModernBERT and intended for search; it is used as pretrained, not fine-tuned
   on the sample.
2. Jev evaluates candidate relevance with typed yes/no (`Noul`) questions and
   reranks the candidates. It is a separate API service, not a ModernBERT
   component.
3. The question-search sample comes from Kaggle's Quora Question Pairs
   competition. The PIM demo includes a small example catalog and can load
   product CSV files downloaded from Kaggle.

The Kaggle sample is a small local search corpus, not a training set in this
starter implementation. The dataset and embedding cache are kept out of Git.

## Requirements

- Python 3.12 or newer (`jevclient` requires Python 3.12)
- Kaggle account with access to the [Quora Question Pairs competition](https://www.kaggle.com/competitions/quora-question-pairs)
- Neo4j 5.x database for PIM import and graph-backed product search
- `JEV_API_KEY` from TypeSafe for hybrid search

## Setup

In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Authenticate with Kaggle (`kagglehub.login()` or a configured Kaggle API token),
then accept the competition rules on Kaggle. Set the Jev API key for hybrid
search:

```powershell
$env:JEV_API_KEY = "your-api-key"
```

## Prepare a sample

Download and deterministically sample 10,000 rows by default:

```powershell
jev-search prepare --sample-size 10000
```

To sample an already downloaded `train.csv` or ZIP instead:

```powershell
jev-search prepare --csv .\train.csv --sample-size 10000
```

The generated `data\quora_sample.csv` is ignored by Git. Kaggle may require
accepting the competition terms before the first download.

## Product information management, graph, and ontology

PIM uses a product catalog schema with `Product`, `ProductType`, `Category`,
`Brand`, and `Attribute` entities. Neo4j stores products and their ontology
relationships:

- `(Product)-[:INSTANCE_OF]->(ProductType)`
- `(Product)-[:IN_CATEGORY]->(Category)`
- `(Product)-[:HAS_BRAND]->(Brand)`
- `(Product)-[:HAS_ATTRIBUTE]->(Attribute)`
- `(Category)-[:SUBCATEGORY_OF]->(Category)`

CSV imports accept common headers such as `product_id`/`sku`/`asin`,
`name`/`title`, `description`, `category`, `brand`, and `price`. Category
hierarchies can be written as `Parent > Child`. Other non-empty CSV columns
become product attributes; an `attributes` column can alternatively contain a
JSON object. Imports upsert products and replace their category, brand, and
attribute links.

Set the Neo4j connection before importing or searching:

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "your-password"
# Optional:
$env:NEO4J_DATABASE = "neo4j"
```

Copy the included sample catalog and load it into Neo4j:

```powershell
jev-search pim sample
jev-search pim import --csv .\data\products.csv
```

To use a product CSV from Kaggle instead, configure Kaggle authentication and
download a dataset file. The CSV loader maps common product field names and
supports optional `--file` selection when a dataset has multiple files:

```powershell
jev-search pim download --handle owner/dataset --file products.csv
jev-search pim import --csv .\data\products.csv --limit 10000
```

The default import/search limit is 10,000 products to keep sample runs
manageable; increase it explicitly for a larger catalog.

Search includes taxonomy and product attributes in ModernBERT's retrieval
text, looks up Neo4j ontology context for retrieved candidates, and supplies
that context to Jev for typed relevance reranking:

```powershell
jev-search pim search "water bottle for hiking"
```

To skip the Jev API while testing retrieval and the Neo4j-backed catalog:

```powershell
jev-search pim search "water bottle for hiking" --embedding-only
```

The demo catalog in `examples\products.csv` is illustrative, not Kaggle data.
Use `pim download` to load an actual Kaggle catalog. Kaggle dataset licenses and
terms apply. Product CSVs and generated embedding indexes are ignored by Git.

## Search

Hybrid ModernBERT + Jev reranking:

```powershell
jev-search search "How can I improve my sleep?"
```

ModernBERT retrieval baseline without calling Jev:

```powershell
jev-search search "How can I improve my sleep?" --embedding-only
```

The first search downloads the embedding model and creates a local index. Later
searches reuse the index unless the sample or model name changes. Hybrid mode
sends the query and retrieved candidates (and, for PIM, ontology context) to
TypeSafe's Jev API; review the provider's terms and usage costs before sending
catalog or search data.

## Tests

```powershell
python -m pytest
```
