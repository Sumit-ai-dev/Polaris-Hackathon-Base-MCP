import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from search import Catalog

CATALOG_PATH = Path(__file__).resolve().parent.parent / 'data/products.json'


class RealCatalogRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if CATALOG_PATH.exists():
            cls.catalog = Catalog.load(str(CATALOG_PATH))
        else:
            cls.catalog = None

    def search_results(self, **args):
        if not self.catalog:
            self.skipTest('Catalog data/products.json not found')
        return self.catalog.search(args)['results']

    def test_90s_grunge_returns_results(self):
        # Must return at least 5 products, not empty
        res = self.search_results(query='90s grunge rock band look', limit=10)
        self.assertGreaterEqual(len(res), 5, f'Expected >= 5 results for grunge query, got {len(res)}')

    def test_lazy_sunday_loungewear_pyjamas_breadth(self):
        # Must return full page of 10 pyjamas, not just 1
        res = self.search_results(query='lazy Sunday loungewear pyjamas', limit=10)
        self.assertEqual(len(res), 10, f'Expected 10 pyjama results, got {len(res)}')

    def test_gym_training_athletic_apparel_breadth(self):
        # Must return full page of 10 athletic/gym items, not just 2
        res = self.search_results(query='gym training athletic apparel', limit=10)
        self.assertEqual(len(res), 10, f'Expected 10 athletic results, got {len(res)}')

    def test_cotton_polo_and_chinos_has_both_intents(self):
        # Must return both polo and chinos in results
        res = self.search_results(query='cotton polo and chinos', limit=10)
        self.assertEqual(len(res), 10)
        names = [self.catalog.products_by_id[r['id']]['name'].lower() for r in res]
        has_polo = any('polo' in n for n in names)
        has_chino = any('chino' in n for n in names)
        self.assertTrue(has_polo, 'Expected polo in mixed choice results')
        self.assertTrue(has_chino, 'Expected chino in mixed choice results')


if __name__ == '__main__':
    unittest.main()
