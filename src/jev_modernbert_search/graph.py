from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .pim import Product

_SCHEMA = (
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (n:Category) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (n:Brand) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT product_type_name IF NOT EXISTS FOR (n:ProductType) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT attribute_id IF NOT EXISTS FOR (n:Attribute) REQUIRE n.id IS UNIQUE",
)

_UPSERT_PRODUCTS = """
UNWIND $products AS row
MERGE (p:Product {id: row.id})
SET p.name = row.name,
    p.description = row.description,
    p.price = row.price
MERGE (type:ProductType {name: 'Product'})
MERGE (p)-[:INSTANCE_OF]->(type)
"""

_CLEAR_PRODUCT_LINKS = """
UNWIND $products AS row
MATCH (p:Product {id: row.id})
OPTIONAL MATCH (p)-[r:IN_CATEGORY|HAS_BRAND|HAS_ATTRIBUTE]->()
WITH collect(r) AS relationships
UNWIND relationships AS relationship
WITH relationship WHERE relationship IS NOT NULL
DELETE relationship
"""

_UPSERT_CATEGORIES = """
UNWIND $products AS row
MATCH (p:Product {id: row.id})
UNWIND row.categories AS category_name
MERGE (category:Category {name: category_name})
MERGE (p)-[:IN_CATEGORY]->(category)
"""

_UPSERT_CATEGORY_HIERARCHY = """
UNWIND $parents AS pair
MERGE (parent:Category {name: pair.parent})
MERGE (child:Category {name: pair.child})
MERGE (child)-[:SUBCATEGORY_OF]->(parent)
"""

_UPSERT_BRANDS = """
UNWIND $products AS row
WITH row WHERE row.brand <> ''
MATCH (p:Product {id: row.id})
MERGE (brand:Brand {name: row.brand})
MERGE (p)-[:HAS_BRAND]->(brand)
"""

_UPSERT_ATTRIBUTES = """
UNWIND $products AS row
MATCH (p:Product {id: row.id})
UNWIND row.attributes AS attribute
MERGE (value:Attribute {id: attribute.id})
SET value.name = attribute.name, value.value = attribute.value
MERGE (p)-[:HAS_ATTRIBUTE]->(value)
"""

_CONTEXT_QUERY = """
UNWIND $ids AS product_id
MATCH (p:Product {id: product_id})
OPTIONAL MATCH (p)-[:INSTANCE_OF]->(type:ProductType)
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(category:Category)
OPTIONAL MATCH (p)-[:HAS_BRAND]->(brand:Brand)
OPTIONAL MATCH (p)-[:HAS_ATTRIBUTE]->(attribute:Attribute)
RETURN p.id AS id,
       collect(DISTINCT type.name) AS types,
       collect(DISTINCT category.name) AS categories,
       collect(DISTINCT brand.name) AS brands,
       collect(DISTINCT CASE
           WHEN attribute IS NULL THEN null
           ELSE attribute.name + ': ' + attribute.value
       END) AS attributes
"""


class Neo4jProductGraph:
    def __init__(self, uri: str, username: str, password: str, database: str | None = None):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies to use Neo4j.") from exc
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database

    @classmethod
    def from_env(cls) -> Neo4jProductGraph:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise RuntimeError("Set NEO4J_PASSWORD to connect to the product graph.")
        return cls(uri, username, password, os.environ.get("NEO4J_DATABASE"))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jProductGraph:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _write(self, operation: Any) -> Any:
        with self._driver.session(database=self._database) as session:
            return session.execute_write(operation)

    def upsert_products(self, products: list[Product], batch_size: int = 250) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        records = [product.graph_record() for product in products]
        parents: set[tuple[str, str]] = set()
        for product in products:
            parents.update(zip(product.categories, product.categories[1:]))
        parent_rows = [{"parent": parent, "child": child} for parent, child in parents]

        def write_batch(tx: Any, batch: list[dict[str, object]]) -> None:
            tx.run(_CLEAR_PRODUCT_LINKS, products=batch).consume()
            tx.run(_UPSERT_PRODUCTS, products=batch).consume()
            tx.run(_UPSERT_CATEGORIES, products=batch).consume()
            tx.run(_UPSERT_BRANDS, products=batch).consume()
            tx.run(
                _UPSERT_ATTRIBUTES,
                products=[
                    {
                        **record,
                        "attributes": [
                            {
                                "id": f"{record['id']}:{name}",
                                "name": name,
                                "value": value,
                            }
                            for name, value in record["attributes"].items()
                        ],
                    }
                    for record in batch
                ],
            ).consume()

        def write_schema(tx: Any) -> None:
            for statement in _SCHEMA:
                tx.run(statement).consume()

        self._write(write_schema)
        for start in range(0, len(records), batch_size):
            self._write(lambda tx, batch=records[start : start + batch_size]: write_batch(tx, batch))
        if parent_rows:
            self._write(
                lambda tx: tx.run(_UPSERT_CATEGORY_HIERARCHY, parents=parent_rows).consume()
            )

    def product_context(self, product_ids: Iterable[str]) -> dict[str, str]:
        ids = list(dict.fromkeys(product_ids))
        if not ids:
            return {}
        with self._driver.session(database=self._database) as session:
            records = session.run(_CONTEXT_QUERY, ids=ids)
            context: dict[str, str] = {}
            for record in records:
                details = [
                    *(f"ontology type: {item}" for item in record["types"] if item),
                    *(f"category: {item}" for item in record["categories"] if item),
                    *(f"brand: {item}" for item in record["brands"] if item),
                    *(f"attribute: {item}" for item in record["attributes"] if item),
                ]
                context[record["id"]] = "; ".join(details)
            return context
