"""Rigorous local performance and concurrency benchmark for Polaris MCP Shopping Assistant."""
import concurrent.futures
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from search import Catalog, tokens

CATALOG_PATH = ROOT / 'data/products.json'

REPRESENTATIVE_QUERIES = [
    # Narrow queries
    {'query': 'black slim fit jeans under INR 2000', 'limit': 10},
    {'query': 'white cotton t-shirt', 'filters': {'color': 'White'}, 'limit': 10},
    {'query': 'floral summer dress', 'sort_by': 'price', 'sort_order': 'asc', 'limit': 10},
    # Broad queries
    {'query': 'dress', 'limit': 10},
    {'query': 'cotton', 'limit': 10},
    {'filters': {'category': 'Garment Upper body', 'min_rating': 4.0}, 'limit': 10},
    # Mixed-choice queries
    {'query': 'black jeans and white t-shirts under INR 2000 each', 'limit': 10},
    {'query': 'cotton polo and chinos', 'limit': 10},
    # Unusual & descriptive queries
    {'query': 'funeral attire and mourning dress', 'limit': 10},
    {'query': '90s grunge rock band look', 'limit': 10},
    {'query': 'monsoon rainwear waterproof', 'limit': 10},
    {'query': 'gym training athletic apparel', 'limit': 10},
    # Recommendation queries
    {'query': 'A dress for dinner under INR 2000 with a matching bag under INR 1000', 'include_recommendations': True, 'limit': 5},
    # Filter-only queries
    {'filters': {'color': 'Black', 'max_price': 2000}, 'sort_by': 'price', 'sort_order': 'asc', 'limit': 10},
    {'sort_by': 'rating', 'sort_order': 'desc', 'limit': 10},
]


def run_benchmark_concurrency(cat, concurrency=1, rounds=10):
    all_tasks = REPRESENTATIVE_QUERIES * rounds
    latencies = []
    failures = 0

    def execute_call(args):
        t0 = time.perf_counter()
        try:
            res = cat.search(args)
            dt = (time.perf_counter() - t0) * 1000.0
            return dt, len(res.get('results', []))
        except Exception as e:
            return None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(execute_call, q) for q in all_tasks]
        for f in concurrent.futures.as_completed(futures):
            dt, count = f.result()
            if dt is not None:
                latencies.append(dt)
            else:
                failures += 1

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    return {
        'total': n,
        'failures': failures,
        'p50': p50,
        'p95': p95,
        'p99': p99,
        'mean': sum(latencies) / n,
    }


def main():
    print("=" * 65)
    print("      POLARIS MCP CONCURRENCY & LATENCY BENCHMARK")
    print("=" * 65)
    print(f"Loading catalog: {CATALOG_PATH.name}...")
    t0 = time.time()
    cat = Catalog.load(str(CATALOG_PATH))
    print(f"Catalog loaded in {time.time() - t0:.2f}s.\n")

    # Warm-up phase (15 calls)
    print("Warming up index and caches (15 calls)...")
    for q in REPRESENTATIVE_QUERIES:
        cat.search(q)
    print("Warm-up complete.\n")

    # Concurrency 1 (150 calls)
    print("Running Concurrency 1 (150 calls)...")
    c1 = run_benchmark_concurrency(cat, concurrency=1, rounds=10)
    print(f"  Calls: {c1['total']} | Failures: {c1['failures']}")
    print(f"  p50: {c1['p50']:.2f} ms | p95: {c1['p95']:.2f} ms | p99: {c1['p99']:.2f} ms | Mean: {c1['mean']:.2f} ms\n")

    # Concurrency 5 (150 calls)
    print("Running Concurrency 5 (150 calls)...")
    c5 = run_benchmark_concurrency(cat, concurrency=5, rounds=10)
    print(f"  Calls: {c5['total']} | Failures: {c5['failures']}")
    print(f"  p50: {c5['p50']:.2f} ms | p95: {c5['p95']:.2f} ms | p99: {c5['p99']:.2f} ms | Mean: {c5['mean']:.2f} ms\n")

    print("=" * 65)
    print(f"SUMMARY: Concurrency 1 p95 = {c1['p95']:.2f} ms | Concurrency 5 p95 = {c5['p95']:.2f} ms")
    assert c1['p95'] < 200.0, f"Concurrency 1 p95 exceeded 200ms: {c1['p95']:.2f}ms"
    assert c5['p95'] < 2000.0, f"Concurrency 5 p95 exceeded 2000ms: {c5['p95']:.2f}ms"
    print("PASS: Sub-200ms p95 at C=1 and sub-2000ms p95 at C=5 achieved!")
    print("=" * 65)


if __name__ == '__main__':
    main()
