from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Product:
    product_id: str
    name: str
    description: str
    categories: tuple[str, ...]
    brand: str
    price: float | None
    attributes: dict[str, str]

    def to_search_text(self) -> str:
        details = [
            "Ontology type: Product",
            f"Product ID: {self.product_id}",
            f"Name: {self.name}",
            f"Category: {' > '.join(self.categories)}" if self.categories else "",
            f"Brand: {self.brand}" if self.brand else "",
            f"Description: {self.description}" if self.description else "",
            (
                "Attributes: "
                + "; ".join(f"{key}: {value}" for key, value in sorted(self.attributes.items()))
                if self.attributes
                else ""
            ),
        ]
        return ". ".join(part for part in details if part)

    def graph_record(self) -> dict[str, object]:
        return {
            "id": self.product_id,
            "name": self.name,
            "description": self.description,
            "categories": list(self.categories),
            "brand": self.brand,
            "price": self.price,
            "attributes": self.attributes,
        }


_ALIASES = {
    "product_id": ("product_id", "id", "sku", "asin", "item_id"),
    "name": ("name", "title", "product_name", "product_title"),
    "description": ("description", "about_product", "product_description"),
    "category": ("category", "category_path", "product_category", "main_category"),
    "brand": ("brand", "brand_name", "manufacturer"),
    "price": ("price", "discounted_price", "selling_price", "actual_price"),
}
_RESERVED = {alias for aliases in _ALIASES.values() for alias in aliases}


def _column(fieldnames: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {name.strip().casefold(): name for name in fieldnames if name}
    return next((normalized[alias] for alias in aliases if alias in normalized), None)


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _parse_price(value: str | None, row_number: int) -> float | None:
    text = _clean(value)
    if not text:
        return None
    normalized = text.replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip()
    try:
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid price on CSV row {row_number}: {text!r}") from exc


def _parse_attributes(
    row: dict[str, str | None], fieldnames: list[str], category_column: str | None
) -> dict[str, str]:
    attributes: dict[str, str] = {}
    attribute_column = _column(fieldnames, ("attributes", "product_attributes"))
    if attribute_column:
        raw_attributes = _clean(row.get(attribute_column))
        if raw_attributes:
            try:
                parsed = json.loads(raw_attributes)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON attributes on CSV row: {raw_attributes!r}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError("The attributes column must contain a JSON object.")
            attributes.update(
                {
                    str(key).strip(): str(value).strip()
                    for key, value in parsed.items()
                    if str(key).strip() and str(value).strip()
                }
            )

    for column in fieldnames:
        if not column or column.strip().casefold() in _RESERVED:
            continue
        value = _clean(row.get(column))
        if value:
            attributes.setdefault(column.strip(), value)
    if attribute_column:
        attributes.pop(attribute_column, None)
    if category_column:
        attributes.pop(category_column, None)
    return attributes


def load_products(path: Path, limit: int | None = None) -> list[Product]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        id_column = _column(fieldnames, _ALIASES["product_id"])
        name_column = _column(fieldnames, _ALIASES["name"])
        if not id_column or not name_column:
            raise ValueError(
                "Product CSV must include an ID column (product_id, id, sku, or asin) "
                "and a name column (name, title, or product_name)."
            )
        description_column = _column(fieldnames, _ALIASES["description"])
        category_column = _column(fieldnames, _ALIASES["category"])
        brand_column = _column(fieldnames, _ALIASES["brand"])
        price_column = _column(fieldnames, _ALIASES["price"])

        products: list[Product] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            product_id = _clean(row.get(id_column))
            name = _clean(row.get(name_column))
            if not product_id or not name:
                continue
            if product_id in seen_ids:
                raise ValueError(f"Duplicate product ID {product_id!r} on CSV row {row_number}.")
            seen_ids.add(product_id)
            category_text = _clean(row.get(category_column)) if category_column else ""
            categories = tuple(
                part.strip() for part in category_text.replace("/", ">").split(">") if part.strip()
            )
            products.append(
                Product(
                    product_id=product_id,
                    name=name,
                    description=_clean(row.get(description_column))
                    if description_column
                    else "",
                    categories=categories,
                    brand=_clean(row.get(brand_column)) if brand_column else "",
                    price=_parse_price(row.get(price_column), row_number)
                    if price_column
                    else None,
                    attributes=_parse_attributes(row, fieldnames, category_column),
                )
            )
            if limit is not None and len(products) >= limit:
                break
    if not products:
        raise ValueError(f"No valid products were found in {path}.")
    return products
