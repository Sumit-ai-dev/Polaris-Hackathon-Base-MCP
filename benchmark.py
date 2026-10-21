"""Automated 100-Case Local Benchmark Runner for Polaris Hackathon.
Simulates the exact organizer evaluation suite and scoring formulas from hackathon-brief.md:
1. Validity & Constraints: 30 cases (10 invalid + 20 valid with filters/sorts/pagination) -> 25 pts
2. Ranking Relevance: 25 single-intent queries evaluated by DCG@10 -> 30 pts
3. Unusual & Irrelevant: 10 unusual + 10 irrelevant/impossible (must return []) -> 15 pts
4. Mixed Choices: 15 dual-intent queries (choice coverage + DCG@10) -> 15 pts
5. Complementary Recommendations: 10 recommendation queries (cross-category + for_product_id) -> 10 pts
6. Performance: 50 requests timed for p95 latency under 200ms -> 5 pts
Total: 100 points
"""
import json
import math
import time
from pathlib import Path

from search import Catalog, INPUT_SCHEMA, OUTPUT_SCHEMA
from jsonschema import Draft202012Validator

VALIDATOR_OUT = Draft202012Validator(OUTPUT_SCHEMA)

# --- 1. VALIDITY CASES (30 cases: 10 invalid, 20 valid) ---
INVALID_CALLS = [
    {'limit': 0},
    {'limit': 101},
    {'limit': True},
    {'offset': -1},
    {'offset': 10001},
    {'sort_by': 'invalid_column'},
    {'filters': {'color': '   '}},
    {'filters': {'min_price': 5000, 'max_price': 1000}},
    {'filters': {'min_rating': 6.0}},
    {'query': None},
]

VALID_FILTER_CALLS = [
    ({'filters': {'color': 'Black', 'max_price': 2000}}, 'color_price'),
    ({'filters': {'category': 'Garment Upper body', 'min_rating': 4.0}}, 'cat_rating'),
    ({'filters': {'audience': 'Ladieswear', 'min_review_count': 1}}, 'aud_review'),
    ({'filters': {'product_type': 'Dress'}}, 'type_dress'),
    ({'filters': {'product_type': 'T-shirt', 'color': 'White'}}, 'type_color'),
    ({'sort_by': 'price', 'sort_order': 'asc', 'limit': 10}, 'sort_price_asc'),
    ({'sort_by': 'price', 'sort_order': 'desc', 'limit': 10}, 'sort_price_desc'),
    ({'sort_by': 'rating', 'sort_order': 'desc', 'limit': 10}, 'sort_rating_desc'),
    ({'sort_by': 'rating', 'sort_order': 'asc', 'limit': 10}, 'sort_rating_asc'),
    ({'sort_by': 'review_count', 'sort_order': 'desc', 'limit': 10}, 'sort_count_desc'),
    ({'sort_by': 'id', 'sort_order': 'asc', 'limit': 10}, 'sort_id_asc'),
    ({'sort_by': 'id', 'sort_order': 'desc', 'limit': 10}, 'sort_id_desc'),
    ({'offset': 0, 'limit': 5}, 'page_1'),
    ({'offset': 5, 'limit': 5}, 'page_2'),
    ({'offset': 10, 'limit': 10}, 'page_3'),
    ({'filters': {'pattern': 'Solid'}}, 'pattern_solid'),
    ({'filters': {'section': 'Womens Everyday Basics'}}, 'section_basics'),
    ({'filters': {'color': 'Blue', 'min_price': 1000, 'max_price': 3000}}, 'color_price_range'),
    ({'filters': {'category': 'Shoes', 'max_price': 4000}}, 'shoes_price'),
    ({'query': '   '}, 'whitespace_query'),
]

# --- 2. RANKING RELEVANCE CASES (25 cases) ---
RELEVANCE_QUERIES = [
    ("black slim fit jeans", ["jeans", "trousers", "black"]),
    ("white cotton t-shirt", ["t-shirt", "top", "white"]),
    ("floral summer dress", ["dress", "summer", "floral"]),
    ("blue denim jacket", ["jacket", "denim", "blue"]),
    ("grey pullover hoodie", ["hoodie", "sweatshirt", "sweater", "grey"]),
    ("formal black trousers", ["trousers", "pants", "formal", "black"]),
    ("casual linen shirt", ["shirt", "linen", "casual"]),
    ("leather shoulder bag", ["bag", "shoulder", "leather"]),
    ("running shoes for sport", ["shoes", "sneakers", "running", "sport"]),
    ("cozy knit sweater", ["sweater", "knit", "jumper"]),
    ("striped cotton polo", ["polo", "shirt", "striped"]),
    ("high waist denim skirt", ["skirt", "denim"]),
    ("wool winter coat", ["coat", "jacket", "winter", "wool"]),
    ("sleeveless cotton vest top", ["vest", "top", "sleeveless"]),
    ("cargo trousers with pockets", ["trousers", "cargo", "pants"]),
    ("red evening cocktail dress", ["dress", "red"]),
    ("unisex canvas sneakers", ["sneakers", "shoes", "canvas"]),
    ("plain black ankle boots", ["boots", "shoes", "black"]),
    ("silk blend formal blazer", ["blazer", "jacket", "formal"]),
    ("printed beach swimsuit", ["swimsuit", "swimwear", "bikini"]),
    ("women pyjama nightwear set", ["pyjama", "nightwear", "sleepwear"]),
    ("men cotton boxer briefs", ["underwear", "boxer", "briefs"]),
    ("chiffon long sleeve blouse", ["blouse", "shirt", "top"]),
    ("leather waist belt with buckle", ["belt", "accessories", "leather"]),
    ("warm fleece zip jacket", ["jacket", "fleece", "outerwear"]),
]

# --- 3. UNUSUAL (10) & IRRELEVANT (10) CASES ---
UNUSUAL_QUERIES = [
    ("funeral attire and mourning dress", ["dress", "suit", "black", "formal"]),
    ("90s grunge rock band look", ["denim", "jacket", "boots", "shirt"]),
    ("monsoon rainwear waterproof", ["jacket", "coat", "boots", "umbrella"]),
    ("lazy Sunday loungewear pyjamas", ["pyjama", "jersey", "hoodie", "sweatshirt"]),
    ("wedding guest formal attire", ["dress", "suit", "blazer"]),
    ("cozy cabin winter thermal knit", ["sweater", "cardigan", "wool", "jacket"]),
    ("retro vintage denim look", ["denim", "jeans", "jacket"]),
    ("minimalist neutral beige essentials", ["beige", "top", "trousers"]),
    ("gym training athletic apparel", ["sport", "t-shirt", "shorts", "leggings"]),
    ("beach resort summer holiday outfit", ["swimwear", "dress", "sandals", "linen"]),
]

IRRELEVANT_QUERIES = [
    "spaceship engine rocket parts",
    "Apple iPhone 15 pro max smartphone",
    "replacement car battery 12V automotive",
    "prescription antibiotics amoxicillin medicine",
    "fresh organic pepperoni pizza delivery",
    "plumbing pipe wrench tool kit",
    "macbook air 65w usb-c charger",
    "cheap international flight tickets",
    "pedigree adult dog food 10kg",
    "pure 24k gold coin investment",
]

# --- 4. MIXED CHOICE CASES (15 cases) ---
MIXED_QUERIES = [
    ("black jeans and white t-shirts under INR 2000 each", ["jeans", "trousers"], ["t-shirt", "top"]),
    ("blue skirts and pink tops", ["skirt"], ["top", "t-shirt", "shirt"]),
    ("denim jacket and black trousers", ["jacket"], ["trousers", "jeans", "pants"]),
    ("cotton shirts and casual shorts", ["shirt"], ["shorts"]),
    ("summer dress and beach hat", ["dress"], ["hat", "cap"]),
    ("hoodie and sweatpants", ["hoodie", "sweatshirt"], ["trousers", "pants"]),
    ("blazer and formal trousers", ["blazer", "jacket"], ["trousers", "pants"]),
    ("sneakers and athletic socks", ["sneakers", "shoes"], ["socks"]),
    ("knitted sweater and winter scarf", ["sweater", "cardigan"], ["scarf"]),
    ("cotton polo and chinos", ["polo", "shirt"], ["trousers", "pants"]),
    ("sports bra and running leggings", ["bra"], ["leggings", "tights"]),
    ("silk blouse and pencil skirt", ["blouse", "top"], ["skirt"]),
    ("leather belt and dress shoes", ["belt"], ["shoes", "boots"]),
    ("linen trousers and sleeveless top", ["trousers", "pants"], ["top", "vest"]),
    ("swimsuit and sunglasses", ["swimsuit", "bikini"], ["sunglasses", "accessories"]),
]

# --- 5. COMPLEMENTARY RECOMMENDATION CASES (10 cases) ---
RECOMMENDATION_QUERIES = [
    "A dress for dinner under INR 2000 with a matching bag under INR 1000",
    "black evening gown with matching clutch",
    "formal two-piece suit with matching shoes",
    "casual summer dress with a matching bag",
    "women cotton top with matching accessories",
    "denim jacket with matching shoes",
    "floral party dress with matching jewellery",
    "blue party dress with matching handbag",
    "men white shirt with matching belt",
    "summer beach dress with matching sunglasses",
]


def score_dcg10(results, cat, target_keywords):
    dcg = 0.0
    for rank, r in enumerate(results[:10]):
        p = cat.products_by_id.get(r['id'])
        if not p:
            continue
        text = ' '.join(str(p.get(k) or '').lower() for k in ('name', 'product_type', 'category', 'color', 'description'))
        # Judge relevance: 3 if multiple target keywords match, 2 if at least 1 matches, 1 if weak
        matches = sum(1 for kw in target_keywords if kw in text)
        grade = 3 if matches >= 2 else (2 if matches >= 1 else 0)
        dcg += (2**grade - 1) / math.log2(rank + 2)
    # Ideal DCG with all grade 3s
    idcg = sum((2**3 - 1) / math.log2(i + 2) for i in range(10))
    return dcg / idcg if idcg > 0 else 0.0


def run_benchmark():
    print("=" * 65)
    print("   POLARIS MCP SHOPPING ASSISTANT BENCHMARK (100 CASES)")
    print("=" * 65)

    catalog_path = Path(__file__).parent / 'data/products.json'
    print(f"Loading catalog: {catalog_path.name}...")
    t0 = time.time()
    cat = Catalog.load(str(catalog_path))
    t_load = time.time() - t0
    print(f"Catalog loaded with {len(cat.products)} products in {t_load:.2f}s.\n")

    # -------------------------------------------------------------
    # CATEGORY 1: VALIDITY & CONSTRAINTS (25 Points)
    # -------------------------------------------------------------
    print("Evaluating Category 1: Validity & Hard Constraints (30 cases)...")
    val_passed = 0
    total_val = len(INVALID_CALLS) + len(VALID_FILTER_CALLS)

    for args in INVALID_CALLS:
        try:
            cat.search(args)
            # If search didn't raise, it failed invalid check
        except (ValueError, Exception):
            val_passed += 1

    for args, label in VALID_FILTER_CALLS:
        try:
            res = cat.search(args)
            assert 'results' in res and 'recommendations' in res
            VALIDATOR_OUT.validate(res)
            # Check price/rating filter constraints if present
            filters = args.get('filters', {})
            all_comply = True
            for r in res['results']:
                p = cat.products_by_id[r['id']]
                if 'max_price' in filters and p['price']['amount'] > filters['max_price']:
                    all_comply = False
                if 'min_price' in filters and p['price']['amount'] < filters['min_price']:
                    all_comply = False
                if 'min_rating' in filters and (p['rating']['average'] is None or p['rating']['average'] < filters['min_rating']):
                    all_comply = False
            if all_comply:
                val_passed += 1
        except Exception as e:
            pass

    score_validity = 25.0 * (val_passed / total_val)
    print(f"  Passed {val_passed}/{total_val} validity checks -> Score: {score_validity:.2f} / 25.0\n")

    # -------------------------------------------------------------
    # CATEGORY 2: RANKING RELEVANCE (30 Points)
    # -------------------------------------------------------------
    print("Evaluating Category 2: Ranking Relevance (25 single-intent cases)...")
    ndcg_scores = []
    for q_str, keywords in RELEVANCE_QUERIES:
        res = cat.search({'query': q_str, 'limit': 10})
        ndcg = score_dcg10(res['results'], cat, keywords)
        ndcg_scores.append(ndcg)

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    score_relevance = 30.0 * mean_ndcg
    print(f"  Mean nDCG@10: {mean_ndcg:.4f} -> Score: {score_relevance:.2f} / 30.0\n")

    # -------------------------------------------------------------
    # CATEGORY 3: UNUSUAL & IRRELEVANT (15 Points)
    # -------------------------------------------------------------
    print("Evaluating Category 3: Unusual & Irrelevant Queries (20 cases)...")
    # 10 Unusual
    unusual_ndcg = []
    for q_str, keywords in UNUSUAL_QUERIES:
        res = cat.search({'query': q_str, 'limit': 10})
        ndcg = score_dcg10(res['results'], cat, keywords)
        unusual_ndcg.append(ndcg)
    mean_unusual = sum(unusual_ndcg) / len(unusual_ndcg) if unusual_ndcg else 0
    score_unusual = 7.5 * mean_unusual

    # 10 Irrelevant (must return empty [])
    irrelevant_passed = 0
    for q_str in IRRELEVANT_QUERIES:
        res = cat.search({'query': q_str})
        if res.get('results') == [] and res.get('recommendations') == []:
            irrelevant_passed += 1

    score_irrelevant = 7.5 * (irrelevant_passed / len(IRRELEVANT_QUERIES))
    score_cat3 = score_unusual + score_irrelevant
    print(f"  Unusual mean nDCG: {mean_unusual:.4f} ({score_unusual:.2f} pts)")
    print(f"  Irrelevant abstention: {irrelevant_passed}/10 ({score_irrelevant:.2f} pts)")
    print(f"  Category 3 Score: {score_cat3:.2f} / 15.0\n")

    # -------------------------------------------------------------
    # CATEGORY 4: MIXED CHOICES (15 Points)
    # -------------------------------------------------------------
    print("Evaluating Category 4: Mixed Choices (15 cases)...")
    mixed_scores = []
    for q_str, c1_kws, c2_kws in MIXED_QUERIES:
        res = cat.search({'query': q_str, 'limit': 10})
        results = res['results']
        # Check coverage
        has_c1 = False
        has_c2 = False
        for r in results:
            p = cat.products_by_id.get(r['id'])
            if not p:
                continue
            text = ' '.join(str(p.get(k) or '').lower() for k in ('name', 'product_type', 'category', 'description'))
            if any(k in text for k in c1_kws):
                has_c1 = True
            if any(k in text for k in c2_kws):
                has_c2 = True
        coverage = 1.0 if (has_c1 and has_c2) else (0.5 if (has_c1 or has_c2) else 0.0)
        ndcg = score_dcg10(results, cat, c1_kws + c2_kws)
        case_score = 0.5 * coverage + 0.5 * ndcg
        mixed_scores.append(case_score)

    mean_mixed = sum(mixed_scores) / len(mixed_scores) if mixed_scores else 0
    score_mixed = 15.0 * mean_mixed
    print(f"  Mean Mixed Choice Score: {mean_mixed:.4f} -> Score: {score_mixed:.2f} / 15.0\n")

    # -------------------------------------------------------------
    # CATEGORY 5: COMPLEMENTARY RECOMMENDATIONS (10 Points)
    # -------------------------------------------------------------
    print("Evaluating Category 5: Complementary Recommendations (10 cases)...")
    rec_passed = 0
    for q_str in RECOMMENDATION_QUERIES:
        res = cat.search({'query': q_str, 'limit': 10, 'include_recommendations': True})
        prim_results = res.get('results', [])
        recs = res.get('recommendations', [])
        if not prim_results:
            continue
        prim_ids = {r['id'] for r in prim_results}
        valid_recs = True
        for rec in recs:
            if rec['id'] in prim_ids:  # Cannot duplicate primary
                valid_recs = False
            if rec['for_product_id'] not in prim_ids:  # Must reference on-page primary
                valid_recs = False
            if not rec.get('reason'):
                valid_recs = False
        if valid_recs and len(recs) > 0:
            rec_passed += 1

    score_recs = 10.0 * (rec_passed / len(RECOMMENDATION_QUERIES))
    print(f"  Valid Recommendations: {rec_passed}/{len(RECOMMENDATION_QUERIES)} -> Score: {score_recs:.2f} / 10.0\n")

    # -------------------------------------------------------------
    # CATEGORY 6: PERFORMANCE & LATENCY (5 Points)
    # -------------------------------------------------------------
    print("Evaluating Category 6: Performance & Latency (50 calls)...")
    latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        cat.search({'filters': {'color': 'Black', 'max_price': 2000}, 'limit': 10, 'offset': 0})
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        latencies.append(t_elapsed)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    functional_total = score_validity + score_relevance + score_cat3 + score_mixed + score_recs
    eligible = functional_total >= 85.5

    # Formula: 5 * max(0, min(1, (2000 - p95) / 1800))
    score_perf = 5.0 * max(0.0, min(1.0, (2000.0 - p95) / 1800.0)) if eligible else 0.0
    print(f"  p50 Latency: {p50:.2f} ms | p95 Latency: {p95:.2f} ms")
    print(f"  Eligible: {eligible} -> Score: {score_perf:.2f} / 5.0\n")

    # -------------------------------------------------------------
    # FINAL BENCHMARK SUMMARY
    # -------------------------------------------------------------
    total_score = functional_total + score_perf
    print("=" * 65)
    print("                     FINAL SCORE SUMMARY                     ")
    print("=" * 65)
    print(f"  1. Validity & Constraints:         {score_validity:6.2f} / 25.00")
    print(f"  2. Ranking Relevance:              {score_relevance:6.2f} / 30.00")
    print(f"  3. Unusual & Irrelevant Queries:   {score_cat3:6.2f} / 15.00")
    print(f"  4. Mixed Choices ('X and Y'):      {score_mixed:6.2f} / 15.00")
    print(f"  5. Complementary Recommendations:  {score_recs:6.2f} / 10.00")
    print(f"  6. Latency & Performance:          {score_perf:6.2f} /  5.00")
    print("-" * 65)
    print(f"  TOTAL BENCHMARK SCORE:             {total_score:6.2f} / 100.00")
    print("=" * 65)

    return total_score


if __name__ == '__main__':
    run_benchmark()
