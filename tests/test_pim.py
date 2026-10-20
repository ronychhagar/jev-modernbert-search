import tempfile
import unittest
from pathlib import Path

from jev_modernbert_search.pim import load_products


class ProductCatalogTests(unittest.TestCase):
    def test_loads_common_catalog_fields_and_builds_ontology_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.csv"
            source.write_text(
                "sku,title,description,category,brand,price,color\n"
                "A-1,Trail Bottle,Insulated bottle,Outdoor > Hydration,Camp Co,$24.50,green\n",
                encoding="utf-8",
            )

            product = load_products(source)[0]

        self.assertEqual(product.product_id, "A-1")
        self.assertEqual(product.categories, ("Outdoor", "Hydration"))
        self.assertEqual(product.price, 24.5)
        self.assertEqual(product.attributes, {"color": "green"})
        self.assertIn("Ontology type: Product", product.to_search_text())
        self.assertIn("Product ID: A-1", product.to_search_text())
        self.assertIn("Category: Outdoor > Hydration", product.to_search_text())

    def test_loads_json_attributes_and_honors_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.csv"
            source.write_text(
                "id,name,attributes\n"
                '1,First,"{""material"": ""steel""}"\n'
                '2,Second,"{""material"": ""cotton""}"\n',
                encoding="utf-8",
            )

            products = load_products(source, limit=1)

        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].attributes, {"material": "steel"})

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.csv"
            source.write_text("id,name\nsame,First\nsame,Second\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate product ID"):
                load_products(source)

    def test_invalid_json_attributes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "catalog.csv"
            source.write_text(
                "id,name,attributes\n1,Product,not-json\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Invalid JSON attributes"):
                load_products(source)


if __name__ == "__main__":
    unittest.main()
