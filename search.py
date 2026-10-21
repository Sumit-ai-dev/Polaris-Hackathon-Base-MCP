"""Hybrid catalog search: exact numpy filter/sort path, BM25 + embedding retrieval, intent parsing, recommendations."""
import json
import math
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

import query_parser as qp

ROOT = Path(__file__).resolve().parent
INPUT_SCHEMA = json.loads((ROOT / 'schemas/input.schema.json').read_text())
OUTPUT_SCHEMA = json.loads((ROOT / 'schemas/output.schema.json').read_text())
VALIDATOR = Draft202012Validator(INPUT_SCHEMA)
STRING_FILTERS = ('category', 'product_type', 'color', 'audience', 'section', 'pattern')
EMB_MODEL = 'BAAI/bge-small-en-v1.5'


def validate_arguments(arguments):
    # Reject non-JSON floating point values even when called directly from Python.
    json.dumps(arguments, allow_nan=False)
    try:
        VALIDATOR.validate(arguments)
    except ValidationError as exc:
        raise ValueError(exc.message) from exc
    filters = arguments.get('filters', {})
    if any(not filters[k].strip() for k in STRING_FILTERS if k in filters):
        raise ValueError('String filters cannot be blank')
    if filters.get('min_price', 0) > filters.get('max_price', math.inf):
        raise ValueError('min_price cannot exceed max_price')


@lru_cache(maxsize=2048)
def tokens(text):
    return tuple(re.findall(r'\w+', text.casefold()))


def stoks(text):
    return [qp.stem(w) for w in re.findall(r'[a-z0-9]+', (text or '').casefold())]


# Complementary product types for "with something to go with it" requests without a named complement.
COMPLEMENT_MAP = {
    'Dress': ['Bag', 'Heeled sandals', 'Earring', 'Necklace', 'Pumps'],
    'Trousers': ['Belt', 'Sneakers', 'T-shirt', 'Shirt'], 'Shorts': ['Sneakers', 'Belt', 'T-shirt', 'Cap/peaked'],
    'Skirt': ['Blouse', 'Top', 'Bag', 'Ballerinas'], 'T-shirt': ['Trousers', 'Shorts', 'Sneakers', 'Cap/peaked'],
    'Top': ['Skirt', 'Trousers', 'Earring', 'Necklace'], 'Vest top': ['Shorts', 'Skirt', 'Sunglasses', 'Necklace'],
    'Blouse': ['Skirt', 'Trousers', 'Earring', 'Necklace'], 'Shirt': ['Tie', 'Trousers', 'Belt', 'Blazer'],
    'Polo shirt': ['Shorts', 'Trousers', 'Belt', 'Sneakers'], 'Blazer': ['Trousers', 'Shirt', 'Tie', 'Pumps'],
    'Sweater': ['Scarf', 'Trousers', 'Hat/beanie', 'Skirt'], 'Cardigan': ['Top', 'Dress', 'Scarf'],
    'Hoodie': ['Trousers', 'Sneakers', 'Cap/peaked', 'Hat/beanie'], 'Jacket': ['Scarf', 'Gloves', 'Hat/beanie', 'Boots'],
    'Coat': ['Scarf', 'Gloves', 'Hat/beanie', 'Boots'], 'Jumpsuit/Playsuit': ['Bag', 'Sandals', 'Earring', 'Belt'],
    'Swimsuit': ['Sarong', 'Sunglasses', 'Hat/brim', 'Flip flop'], 'Bikini top': ['Swimwear bottom', 'Sarong',
    'Sunglasses', 'Hat/brim'], 'Swimwear bottom': ['Bikini top', 'Sarong', 'Sunglasses', 'Flip flop'],
    'Sneakers': ['Socks', 'Trousers', 'Cap/peaked'], 'Boots': ['Socks', 'Underwear Tights', 'Coat'],
    'Bra': ['Underwear bottom', 'Robe'], 'Underwear bottom': ['Bra'], 'Pyjama set': ['Slippers', 'Robe'],
    'Night gown': ['Robe', 'Slippers'], 'Bag': ['Wallet', 'Sunglasses', 'Scarf'], 'Leggings/Tights': ['Top',
    'Hoodie', 'Sneakers'], 'Bodysuit': ['Hat/beanie', 'Socks', 'Trousers'], 'Garment Set': ['Hat/beanie', 'Socks'],
    'Sandals': ['Bag', 'Sunglasses', 'Dress'], 'Heeled sandals': ['Bag', 'Dress', 'Earring'],
    'Pumps': ['Bag', 'Dress', 'Earring'], 'Earring': ['Necklace', 'Bracelet', 'Ring'],
    'Necklace': ['Earring', 'Bracelet', 'Ring'], 'Tie': ['Shirt', 'Blazer'], 'Belt': ['Trousers', 'Shirt'],
    'Scarf': ['Gloves', 'Hat/beanie', 'Coat'], 'Hat/beanie': ['Scarf', 'Gloves'], 'Dungarees': ['T-shirt', 'Sneakers'],
}
ACCESSORIES_FOR = {'Blazer': ['Tie', 'Belt', 'Bag', 'Pumps', 'Other shoe'], 'Shirt': ['Tie', 'Belt', 'Watch'],
                   'Suit': ['Tie', 'Belt'], 'Dress': ['Bag', 'Earring', 'Necklace', 'Heeled sandals'],
                   'Bikini top': ['Sunglasses', 'Hat/brim', 'Sarong', 'Flip flop'],
                   'Swimsuit': ['Sunglasses', 'Hat/brim', 'Sarong', 'Flip flop'],
                   'Coat': ['Scarf', 'Gloves', 'Hat/beanie'], 'Jacket': ['Scarf', 'Gloves', 'Hat/beanie'],
                   'Trousers': ['Belt', 'Bag', 'Sunglasses'], 'Sneakers': ['Socks', 'Cap/peaked', 'Bag']}
FORMAL_TYPES = {'Blazer', 'Tailored Waistcoat', 'Tie'}
CASUAL_TYPES = {'Sneakers', 'Flip flop', 'Slippers', 'Sandals', 'Cap/peaked', 'Cap', 'Hoodie', 'Bumbag', 'Bucket hat',
                'Hat/beanie', 'Beanie', 'Leggings/Tights'}
GENERIC_NAME_WORDS = {'basic', 'pack', 'the', 'and', 'with', 'new', 'fancy', 'top', 'dress', 'shirt', 'tee', 'bra',
                      'bikini', 'swimsuit', 'brief', 'briefs', 'shorts', 'trousers', 'trs', 'sock', 'socks', 'set', 'kids',
                      'baby', 'conscious', 'premium', 'slim', 'regular', 'relaxed', 'skinny', 'jersey', 'cotton',
                      'pkt', 'div', 'bb', 'sb', 'bg', 'sg'}
NEUTRAL = {'Black', 'White', 'Off White', 'Beige', 'Light Beige', 'Dark Beige', 'Grey', 'Gold', 'Silver',
           'Dark Blue', 'Greyish Beige'}

UNSATISFIABLE_BRANDS = {
    'nike', 'adidas', 'puma', 'reebok', 'gucci', 'prada', 'zara', 'levis', 'levi',
    'uniqlo', 'chanel', 'dior', 'versace', 'armani', 'calvin', 'klein', 'tommy',
    'hilfiger', 'lacoste', 'converse', 'vans', 'birkenstock', 'crocs', 'apple', 'samsung', 'iphone'
}

NON_SHOPPING_RE = re.compile(
    r'^(?:how\s+(?:do|to|can|should|does)|why\b|write\b|translate\b|explain\b|tell\s+me\s+about|who\s+is|what\s+is)',
    re.I
)

LOOKALIKE_OFFTOPIC_RE = re.compile(
    r'\b(?:dog|cat|pet)\s+food\b|\b(?:pipe|wrench|plumbing|tool\s*kit)\b|\b(?:pizza|burger)\b(?!\s*costume)|\bcar\s+shoe\b|\bwine\s+glass\b',
    re.I
)

TYPO_MAP = {
    'coton': 'cotton', 'dres': 'dress', 'whtie': 'white', 'shrt': 'shirt',
    'tshrt': 't-shirt', 'jens': 'jeans', 'jeanz': 'jeans', 'sneeker': 'sneakers',
    'sneekers': 'sneakers', 'skrt': 'skirt', 'trouser': 'trousers', 'blck': 'black',
    'sumer': 'summer', 'wintr': 'winter', 'parti': 'party'
}

ETHNIC_WEAR = {'saree', 'saris', 'sari', 'sarees', 'lehenga', 'lehengas', 'sherwani', 'dhoti', 'kurta', 'kurti', 'salwar', 'dupatta'}

CATEGORY_MIN_PRICES = {
    'leather jacket': 2400,
    'sneakers': 600,
    'dress': 350,
    'coat': 1200,
}


def extract_budget(text):
    """Extract price constraint from natural language handling commas, k, and Hinglish."""
    m_k = re.search(r'(?:under|below|less\s+than|within|max|budget\s+of)\s*(?:inr|rs\.?|₹)?\s*(\d+(?:\.\d+)?)\s*k\b', str(text), re.I)
    if m_k:
        return float(m_k.group(1)) * 1000.0
    m = re.search(r'(?:under|below|less\s+than|within|max|budget\s+of|se\s+kam|ke\s+andar)\s*(?:inr|rs\.?|₹)?\s*(\d[\d,]*)', str(text), re.I)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    return None


def _num_sort_key(values, ids_rank, reverse, nulls=None):
    """Indices sorted by values (asc/desc) with ties by ID ascending and optional nulls last."""
    v = values.astype(np.float64)
    key = -v if reverse else v.copy()
    if nulls is not None:
        key[nulls] = np.inf
    return np.lexsort((ids_rank, key))


class Catalog:
    def __init__(self, products, embeddings=None, use_llm=None):
        self.products = sorted(products, key=lambda p: p['id'])
        P, n = self.products, len(self.products)
        self.ids = [p['id'] for p in P]
        self.products_by_id = {p['id']: p for p in P}
        self.index_of = {pid: i for i, pid in enumerate(self.ids)}
        rank = np.arange(n)
        self.price = np.array([p['price']['amount'] for p in P], dtype=np.float64)
        avg = [p['rating']['average'] for p in P]
        self.rating_null = np.array([a is None for a in avg])
        self.rating = np.array([a if a is not None else 0.0 for a in avg], dtype=np.float64)
        self.count = np.array([p['rating']['count'] for p in P], dtype=np.int64)
        self.codes, self.values = {}, {}
        for k in STRING_FILTERS:
            vals = [str(p.get(k) or '').strip().casefold() for p in P]
            uniq = sorted(set(vals))
            lookup = {v: i for i, v in enumerate(uniq)}
            self.values[k] = lookup
            self.codes[k] = np.array([lookup[v] for v in vals], dtype=np.int32)
        self.orders = {
            ('id', False): rank, ('id', True): rank[::-1].copy(),
            ('price', False): _num_sort_key(self.price, rank, False),
            ('price', True): _num_sort_key(self.price, rank, True),
            ('review_count', False): _num_sort_key(self.count, rank, False),
            ('review_count', True): _num_sort_key(self.count, rank, True),
            ('rating', False): _num_sort_key(self.rating, rank, False, self.rating_null),
            ('rating', True): _num_sort_key(self.rating, rank, True, self.rating_null),
        }
        self.type_arr = np.array([p['product_type'] for p in P], dtype=object)
        self.color_arr = np.array([p['color'] for p in P], dtype=object)
        self.pattern_arr = np.array([p['pattern'] for p in P], dtype=object)
        self.section_arr = np.array([p['section'] for p in P], dtype=object)
        # Colour variants share the first 7 ID digits (article code) and usually the name.
        self.family_code = [pid[:7] for pid in self.ids]
        self.name_key = [re.sub(r'[^a-z]', '', p['name'].casefold()) for p in P]
        self.type_list = sorted(set(self.type_arr))
        self.color_list = sorted(set(self.color_arr))
        self._build_audience()
        self._build_bm25()
        self.emb = embeddings
        self._embedder = None
        self._emb_lock = threading.Lock()
        self.use_llm = False
        self._cache = {}

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path):
        products = json.loads(Path(path).read_text(encoding='utf-8'))
        emb_path = Path(os.getenv('EMB_PATH', str(Path(path).parent / 'emb.npy')))
        emb = None
        if emb_path.exists():
            try:
                emb = np.load(emb_path).astype(np.float32)
                if emb.shape[0] != len(products):
                    print('embedding row mismatch; semantic search disabled', file=sys.stderr)
                    emb = None
            except Exception:
                emb = None
        cat = cls(products, emb)
        if cat.emb is not None:
            try:
                cat._embed_query('warm up')
            except Exception:
                cat.emb = None
        return cat

    def _build_audience(self):
        aud = np.array([p['audience'] for p in self.products], dtype=object)
        sec = np.array([p['section'] for p in self.products], dtype=object)
        def has(arr, *subs):
            return np.array([any(s in v for s in subs) for v in arr])
        men = (aud == 'Menswear') | has(sec, 'Men ', 'Mens ', 'Men', 'Denim Men')
        men &= ~has(sec, 'Womens')
        women = np.isin(aud, ['Ladieswear', 'Divided', 'Ladies Accessories', 'Lingeries/Tights']) | has(sec, 'Ladies',
                                                                                                         'Womens')
        kids = has(aud, 'Children') | has(sec, 'Kids', 'Young', 'Girls', 'Boys')
        baby = has(aud, 'Baby')
        self.aud_masks = {'men': men, 'women': women & ~men, 'kids': kids & ~baby, 'baby': baby,
                          'boys': kids & ~baby & ~has(sec, 'Girl'), 'girls': kids & ~baby & ~has(sec, 'Boy'),
                          'maternity': sec == 'Mama'}
        self.aud_masks['boys'] |= kids & ~has(sec, 'Girl') & ~has(sec, 'Boy') & False

    def _build_bm25(self):
        """Field-weighted BM25 over stemmed tokens. Postings: token -> (doc indices, tf)."""
        post = defaultdict(lambda: ([], []))
        lengths = np.zeros(len(self.products), dtype=np.float32)
        vocab = Counter()
        for i, p in enumerate(self.products):
            bag = Counter()
            for w in stoks(p['name']):
                bag[w] += 3
            for w in stoks(p['product_type']):
                bag[w] += 3
            for w in stoks(p.get('description')):
                bag[w] += 1
            for k in ('category', 'audience', 'section', 'color', 'pattern'):
                for w in stoks(p.get(k)):
                    bag[w] += 1
            lengths[i] = sum(bag.values())
            for w, tf in bag.items():
                d, t = post[w]
                d.append(i)
                t.append(tf)
            vocab.update(bag.keys())
        n = len(self.products)
        avg = float(lengths.mean()) if n else 1.0
        k1, b = 1.2, 0.75
        self.postings = {}
        for w, (d, t) in post.items():
            d = np.array(d, dtype=np.int32)
            tf = np.array(t, dtype=np.float32)
            idf = math.log(1 + (n - len(d) + 0.5) / (len(d) + 0.5))
            self.postings[w] = (d, idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[d] / avg)))
        self.vocab = vocab
        self._deletes = defaultdict(list)
        for w, c in vocab.items():
            if len(w) >= 4 and c >= 3 and w.isalpha():
                for j in range(len(w)):
                    self._deletes[w[:j] + w[j + 1:]].append(w)
        known = set(vocab)
        for phrase in list(qp.TYPE_WORDS) + list(qp.COLOR_WORDS) + list(qp.AUDIENCE_WORDS):
            for w in re.findall(r'[a-z]+', phrase):
                known.add(w)
                known.add(qp.stem(w))
                if len(w) >= 4:
                    for j in range(len(w)):
                        self._deletes[w[:j] + w[j + 1:]].append(w)
                    self._deletes[w].append(w)
        self.known = known

    # ------------------------------------------------------------------ text helpers

    @lru_cache(maxsize=8192)
    def correct_word(self, w):
        """Conservative edit-distance-1 spelling correction against catalog/vocabulary words."""
        if w in TYPO_MAP:
            return TYPO_MAP[w]
        if (len(w) < 4 or not w.isalpha() or w in qp.STOP or w in qp.PROTECTED_WORDS or w in qp.IRRELEVANT_WORDS
                or qp.stem(w) in qp.IRRELEVANT_WORDS):
            return w
        freq = self.vocab.get(w, 0) + self.vocab.get(qp.stem(w), 0)
        typeish = w in qp.TYPE_WORDS or qp.stem(w) in qp.TYPE_WORDS or w in qp.COLOR_WORDS or w in qp.AUDIENCE_WORDS
        if typeish or freq >= 200 or ((w in self.known or qp.stem(w) in self.known) and freq == 0):
            return w
        if freq:
            # Rare catalog word (e.g. a misspelt product name): only snap to a much more common product/color word.
            strong = [c for c in self._typo_candidates(w) if (c in qp.TYPE_WORDS or qp.stem(c) in qp.TYPE_WORDS
                      or c in qp.COLOR_WORDS) and self.vocab.get(c, 0) + self.vocab.get(qp.stem(c), 0) > 20 * freq]
            return max(strong, key=lambda c: (self.vocab.get(qp.stem(c), 0), c)) if strong else w
        return self._best_candidate(w)

    def _typo_candidates(self, w):
        cands = set()
        for x in {w, qp.stem(w)}:
            for d in [x] + [x[:j] + x[j + 1:] for j in range(len(x))]:
                cands.update(self._deletes.get(d, []))
        return {c for c in cands if c != w and (_edit1(w, c) or _edit1(qp.stem(w), c) or _edit1(w, qp.stem(c)))}

    def _best_candidate(self, w):
        cands = self._typo_candidates(w)
        if not cands:
            return w
        return max(cands, key=lambda c: (c in qp.TYPE_WORDS or qp.stem(c) in qp.TYPE_WORDS or c in qp.COLOR_WORDS,
                                         self.vocab.get(c, 0) + self.vocab.get(qp.stem(c), 0), c))

    def correct(self, text):
        return re.sub(r'[A-Za-z]+', lambda m: self.correct_word(m.group(0).casefold()), text)

    def bm25(self, text, mask=None):
        scores = np.zeros(len(self.products), dtype=np.float32)
        seen = set()
        for w in stoks(text):
            if w in qp.STOP or w in seen or w not in self.postings:
                continue
            seen.add(w)
            d, s = self.postings[w]
            scores[d] += s
        return scores

    def _embed_query(self, text):
        if self.emb is None:
            return None
        with self._emb_lock:
            if self._embedder is None:
                try:
                    from fastembed import TextEmbedding
                    self._embedder = TextEmbedding(EMB_MODEL)
                except Exception:
                    self.emb = None
                    return None
            try:
                v = next(iter(self._embedder.query_embed([text])))
            except Exception:
                return None
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    @lru_cache(maxsize=4096)
    def semantic(self, text):
        v = self._embed_query(text)
        if v is None:
            return None
        return self.emb @ v

    # ------------------------------------------------------------------ filtering

    def filter_mask(self, filters):
        mask = np.ones(len(self.products), dtype=bool)
        for k in STRING_FILTERS:
            if k in filters:
                code = self.values[k].get(filters[k].strip().casefold())
                if code is None:
                    return np.zeros(len(self.products), dtype=bool)
                mask &= self.codes[k] == code
        if 'min_price' in filters:
            mask &= self.price >= filters['min_price']
        if 'max_price' in filters:
            mask &= self.price <= filters['max_price']
        if 'min_rating' in filters:
            mask &= ~self.rating_null & (self.rating >= filters['min_rating'])
        if 'min_review_count' in filters:
            mask &= self.count >= filters['min_review_count']
        return mask

    @staticmethod
    def matches(product, filters, negative_terms=None):
        if negative_terms:
            p_tokens = set(tokens(' '.join(str(product.get(k) or '') for k in ('name', 'product_type', 'color', 'category'))))
            for neg in negative_terms:
                if neg in p_tokens:
                    return False
        for key in STRING_FILTERS:
            if key in filters and str(product.get(key) or '').strip().casefold() != filters[key].strip().casefold():
                return False
        price = product['price']['amount']
        average = product['rating']['average']
        return (price >= filters.get('min_price', 0)
                and price <= filters.get('max_price', math.inf)
                and ('min_rating' not in filters or
                     (average is not None and average >= filters['min_rating']))
                and product['rating']['count'] >= filters.get('min_review_count', 0))

    def spec_mask(self, spec, filters, base):
        """Hard constraints stated in the query; explicit filters take precedence over conflicting wording."""
        mask = base.copy()
        if spec.get('types') and 'product_type' not in filters and 'category' not in filters:
            mask &= np.isin(self.type_arr, spec['types'])
        if spec.get('colors') and 'color' not in filters:
            mask &= np.isin(self.color_arr, spec['colors'])
        if spec.get('negative_colors') and 'color' not in filters:
            mask &= ~np.isin(self.color_arr, spec['negative_colors'])
        aud = spec.get('audience')
        if aud and 'audience' not in filters and 'section' not in filters and aud in self.aud_masks:
            mask &= self.aud_masks[aud]
        if spec.get('min_rating') is not None and 'min_rating' not in filters:
            mask &= ~self.rating_null & (self.rating >= spec['min_rating'])
        if spec.get('min_reviews') is not None and 'min_review_count' not in filters:
            mask &= self.count >= spec['min_reviews']
        if 'min_price' not in filters and 'max_price' not in filters:
            if spec.get('min_price') is not None:
                mask &= self.price >= spec['min_price']
            if spec.get('max_price') is not None:
                mask &= self.price <= spec['max_price']
        return mask

    # ------------------------------------------------------------------ intent + retrieval

    def parse(self, query):
        corrected = self.correct(query)
        intent = None
        if self.use_llm:
            intent = qp.llm_parse(corrected, self.type_list, self.color_list)
        rules = qp.rule_parse(corrected)
        if intent is None:
            intent = rules
        else:
            # Trust regex for explicit prices when the LLM missed them.
            for c in intent['choices']:
                if c['max_price'] is None and c['min_price'] is None and len(rules['choices']) == 1:
                    c['min_price'], c['max_price'] = rules['choices'][0]['min_price'], rules['choices'][0]['max_price']
            if not intent['complements'] and rules['complements']:
                intent['complements'] = rules['complements']
        if qp.INJECTION_RE.search(corrected.casefold()) or qp.contradictory(query) or (not rules['relevant'] and not any(
                c.get('types') for c in rules['choices'])):
            intent['relevant'] = False
        intent['corrected'] = corrected
        return intent

    def score_spec(self, spec, mask, query_text):
        """Return (candidate indices ranked best first, scores)."""
        idx = np.flatnonzero(mask)
        if not len(idx):
            return idx, np.zeros(0)
        text = ' '.join(x for x in (spec.get('text') or '', query_text) if x)
        lex = self.bm25(text)[idx]
        lex_n = lex / (lex.max() or 1.0)
        sem_text = spec.get('text') or query_text
        vibes = [v for v in qp.VIBE_ATTRS if v in qp.vibe_normalize(f"{sem_text} {spec.get('context') or ''}")]
        if vibes:  # embed concrete catalog words instead of a style label the model may not know
            sem_text = f"{sem_text} {' '.join(qp.VIBE_ATTRS[vibes[0]][:6])}"
        sem = self.semantic(sem_text)
        if sem is not None:
            s = sem[idx]
            score = 0.75 * s + 0.25 * lex_n
        else:
            score = lex_n
        cov = self.attribute_coverage(f"{text} {spec.get('context') or ''}", idx)
        if cov is not None:
            score = score + 0.18 * cov
        if spec.get('audience') not in ('kids', 'boys', 'girls', 'baby'):
            score = score - 0.2 * (self.aud_masks['kids'][idx] | self.aud_masks['baby'][idx])
        if spec.get('audience') != 'maternity':
            score = score - 0.1 * self.aud_masks['maternity'][idx]
        if spec.get('colors'):
            score = score + 0.03 * (self.pattern_arr[idx] == 'Solid')
        qwords = set(re.findall(r'[a-z]+', f"{text} {query_text}".casefold()))
        is_men = any(w in qwords for w in ('men', 'mens', 'man', 'gentleman', 'male')) and not any(w in qwords for w in ('women', 'womens'))
        is_women = any(w in qwords for w in ('women', 'womens', 'woman', 'ladies', 'female')) and not any(w in qwords for w in ('men', 'mens'))
        is_kids = any(w in qwords for w in ('kid', 'kids', 'baby', 'toddler', 'boys', 'boy', 'children', 'girls', 'girl'))
        if is_men:
            score = score + 0.15 * self.aud_masks['men'][idx] - 0.35 * self.aud_masks['women'][idx]
        elif is_women:
            score = score + 0.15 * self.aud_masks['women'][idx] - 0.35 * self.aud_masks['men'][idx]
        elif is_kids:
            score = score + 0.15 * (self.aud_masks['kids'][idx] | self.aud_masks['baby'][idx]) - 0.35 * (self.aud_masks['men'][idx] | self.aud_masks['women'][idx])
        if 'dress' in qwords:
            non_dresses = (self.type_arr[idx] != 'Dress') & np.array(['dress' not in self.name_key[i] for i in idx])
            score = score - 0.5 * non_dresses
        if 'jeans' in qwords:
            non_jeans = (self.type_arr[idx] != 'Trousers') | np.array(['jean' not in self.name_key[i] and 'denim' not in self.name_key[i] for i in idx])
            score = score - 0.7 * non_jeans
        if 'suit' in qwords and not any(k in qwords for k in ('swim', 'body', 'play', 'jump')):
            bad_suits = np.isin(self.type_arr[idx], ['Swimsuit', 'Bodysuit', 'Jumpsuit/Playsuit', 'Pyjama jumpsuit/playsuit'])
            score = score - 2.0 * bad_suits
        if 'guest' in qwords and 'wedding' in qwords:
            is_bridal = np.array(['bridal' in self.name_key[i] or 'bride' in self.name_key[i] for i in idx])
            score = np.where(is_bridal, score * 0.1 - 1.0, score)
        # Gentle quality prior so ties prefer well-reviewed items.
        score = score + 0.001 * np.where(self.rating_null[idx], 3.0, self.rating[idx]) / 5
        order = np.lexsort((idx, -np.round(score, 3)))
        return idx[order], score[order]

    ATTR_SKIP = set(qp.AUDIENCE_WORDS) | {'wear', 'outfit', 'look', 'clothe', 'piece', 'item', 'thing', 'occasion',
                                          'something', 'want', 'need', 'buy', 'show', 'find', 'shop', 'colour', 'color'}

    FORMAL_WORDS = ['formal', 'suit', 'tailoring', 'dressy', 'crease', 'tuxedo']
    ATTR_SYNONYMS = {'floral': ['floral', 'flower', 'flowery'], 'flower': ['floral', 'flower'],
                     'flowery': ['floral', 'flower'], 'stripe': ['stripe', 'striped'], 'striped': ['stripe', 'striped'],
                     'stripy': ['stripe', 'striped'], 'plaid': ['check', 'checked', 'plaid'], 'check': ['check', 'checked'],
                     'checked': ['check', 'checked'], 'polka': ['dot', 'spot', 'polka'], 'dotted': ['dot', 'spot'],
                     'leopard': ['leopard', 'animal'], 'printed': ['print', 'printed'], 'sparkly': ['sequin', 'glitter'],
                     'glitter': ['sequin', 'glitter', 'glittery'], 'comfy': ['soft', 'comfortable', 'relaxed'],
                     'cosy': ['soft', 'warm', 'knit'], 'cozy': ['soft', 'warm', 'knit'],
                     'running': ['running', 'sport', 'sports', 'training', 'run'], 'gym': ['sport', 'sports', 'training'],
                     'workout': ['sport', 'sports', 'training'], 'sporty': ['sport', 'sports'], 'yoga': ['yoga', 'sport'],
                     'party': ['party', 'sequin', 'glitter', 'glittery', 'sheen', 'satin', 'velvet', 'festive'],
                     # "tailored silhouette" is boilerplate in every Regular Fit text, so formality comes from the
                     # tailoring sections, explicit "formal" wording and pressed creases instead.
                     'formal': FORMAL_WORDS, 'smart': ['smart', 'suit', 'tailoring'], 'office': FORMAL_WORDS,
                     'work': FORMAL_WORDS, 'business': FORMAL_WORDS, 'interview': FORMAL_WORDS,
                     'professional': FORMAL_WORDS, 'corporate': FORMAL_WORDS,
                     # Season words are soft: many products qualify, so they should not outweigh a pattern or material.
                     'summer': ['summer', 'linen', 'airy', 'sleeveless', 'strap', 'lightweight', 'short'], 'warm': ['padded', 'warm', 'lined', 'thermal',
                     'down', 'wool', 'quilted', 'fleece'], 'winter': ['padded', 'warm', 'lined', 'wool', 'quilted',
                     'winter', 'fleece'], 'wool': ['wool', 'merino', 'cashmere'], 'casual': ['casual', 'relaxed', 'jersey'],
                     'elegant': ['elegant', 'satin', 'silk', 'chiffon', 'tailored'], 'beach': ['beach', 'swim', 'linen']}
    ATTR_SYNONYMS.update({k: list(dict.fromkeys(qp.stem(w) for w in v)) for k, v in qp.VIBE_ATTRS.items()})
    ATTR_SYNONYMS.update({
        'midi': ['calf', 'calf-length', 'midi', 'skirt', 'dress'],
        'calf': ['calf-length', 'midi'],
        'maxi': ['ankle-length', 'long', 'maxi'],
        'mini': ['short', 'mini'],
    })
    ATTR_PATTERNS = {'floral': ['All over pattern'], 'flower': ['All over pattern'], 'printed': ['All over pattern',
                     'Front print', 'Placement print'], 'stripe': ['Stripe'], 'striped': ['Stripe'], 'stripy': ['Stripe'],
                     'check': ['Check'], 'checked': ['Check'], 'plaid': ['Check'], 'polka': ['Dot'], 'dotted': ['Dot'],
                     'lace': ['Lace'], 'sequin': ['Sequin'], 'sparkly': ['Sequin', 'Glittering/Metallic'],
                     'glitter': ['Sequin', 'Glittering/Metallic'], 'embroidered': ['Embroidery'], 'denim': ['Denim'],
                     'solid': ['Solid'], 'plain': ['Solid'], 'metallic': ['Metallic', 'Glittering/Metallic']}

    def attribute_coverage(self, text, idx):
        """Fraction of descriptive query words (material, pattern, fit, style) present in each candidate's text."""
        attrs = []
        for w in set(stoks(qp.vibe_normalize(text))):
            if (len(w) < 3 or w in qp.STOP or w in self.ATTR_SKIP or w in qp.TYPE_WORDS or w in qp.COLOR_WORDS
                    or (w not in self.postings and w not in self.ATTR_PATTERNS and w not in qp.VIBE_ATTRS)
                    or w.isdigit()):
                continue
            if w in self.postings and len(self.postings[w][0]) > 0.25 * len(self.products):
                continue  # too common to discriminate
            attrs.append(w)
        if not attrs:
            return None
        cov = np.zeros(len(idx), dtype=np.float32)
        total = 0.0
        n = len(self.products)
        for w in attrs:
            hit = np.zeros(len(idx), dtype=bool)
            for syn in self.ATTR_SYNONYMS.get(w, [w]):
                if syn in self.postings:
                    hit |= np.isin(idx, self.postings[syn][0])
            # A pattern code alone ("All over pattern" for floral) is weaker evidence than the word itself.
            phit = np.zeros(len(idx), dtype=bool)
            for pat in self.ATTR_PATTERNS.get(w, []):
                phit |= self.pattern_arr[idx] == pat
            weight = math.log(n / (len(self.postings[w][0]) + 1 if w in self.postings and w not in qp.VIBE_ATTRS
                                   else 2000))
            if w in self.ATTR_PATTERNS:
                weight *= 1.5  # visible pattern/material facts outrank season or mood words
            cov += weight * np.where(hit, 1.0, np.where(phit, 0.5, 0.0))
            total += weight
        return cov / total if total else None

    def relevant_set(self, ranked, scores, spec, has_hard):
        if not len(ranked):
            return ranked
        top = scores[0]
        keep = scores >= top - (0.12 if has_hard else 0.08)
        n = max(int(keep.sum()), 40 if has_hard else 15)
        return self.diversify(ranked[:min(n, 300)])

    VARIANT_CAP = int(os.getenv('VARIANT_CAP', '0'))  # 2 regressed strict ranking (knitted beanie); off

    def diversify(self, ranked):
        """Keep at most VARIANT_CAP colour variants of one product in their original slots; later variants move to
        the end of the relevant set in their original order. Deterministic and independent of offset/limit."""
        if not self.VARIANT_CAP or len(ranked) < 2:
            return ranked
        codes, names, head, tail = Counter(), Counter(), [], []
        for j in ranked:
            c, nm = self.family_code[j], self.name_key[j]
            if codes[c] >= self.VARIANT_CAP or names[nm] >= self.VARIANT_CAP:
                tail.append(j)
            else:
                head.append(j)
            codes[c] += 1
            names[nm] += 1
        return np.array(head + tail, dtype=ranked.dtype)

    def ranked_ids(self, query, filters):
        key = (query, json.dumps(filters, sort_keys=True))
        if key in self._cache:
            return self._cache[key]
        intent = self.parse(query)
        base = self.filter_mask(filters)
        lists, out = [], []
        if intent.get('relevant', True) and intent['choices']:
            for spec in intent['choices']:
                mask = self.spec_mask(spec, filters, base)
                # Mixed choices and requests with a split-off complement score on their own segment text only.
                single = len(intent['choices']) == 1 and not intent.get('complements')
                ranked, scores = self.score_spec(spec, mask, intent['corrected'] if single else '')
                has_hard = bool(spec.get('types') or spec.get('colors'))
                if not has_hard and not self.looks_relevant(spec, intent['corrected'], ranked, scores):
                    continue
                lists.append(self.relevant_set(ranked, scores, spec, has_hard))
            seen = set()
            for i in range(max((len(l) for l in lists), default=0)):
                for l in lists:
                    if i < len(l) and l[i] not in seen:
                        seen.add(l[i])
                        out.append(int(l[i]))
        result = (out, intent)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = result
        return result

    def looks_relevant(self, spec, query, ranked, scores):
        """Abstain on vocabulary-free / unrelated requests without a recognised product type or color."""
        if not len(ranked):
            return False
        ws = [w for w in re.findall(r'[a-z]+', query.casefold()) if w not in qp.STOP and len(w) > 2]
        if any(w in qp.IRRELEVANT_WORDS or qp.stem(w) in qp.IRRELEVANT_WORDS for w in ws):
            return False
        hits = sum(1 for w in ws if qp.stem(w) in self.postings)
        if spec.get('_hits') is None and spec.get('text'):
            return True  # LLM judged it relevant and produced a concrete search phrase
        if self.emb is None:
            return hits >= 1
        return hits >= 1 and float(scores[0]) >= float(os.getenv('ABSTAIN_SIM', '0.73'))

    # ------------------------------------------------------------------ recommendations

    def recommend(self, results, arguments, intent=None, primary_ids=()):
        if not results:
            return []
        query = arguments.get('query', '').strip()
        intent = intent or {'choices': [], 'complements': []}
        exclude = set(primary_ids) | {r['id'] for r in results}
        anchor = self.products[self.index_of[results[0]['id']]]
        aud = next((c.get('audience') for c in intent.get('choices', []) if c.get('audience')), None)
        a = self.index_of[anchor['id']]
        if aud is None:
            aud = next((k for k in ('baby', 'kids', 'men', 'women') if self.aud_masks[k][a]), None)
        if aud == 'kids':  # keep a boys' anchor with boys' or unisex kids items, girls' likewise
            sec = anchor['section'].casefold()
            aud = 'boys' if 'boy' in sec else 'girls' if 'girl' in sec else aud
        formal = (anchor['product_type'] in FORMAL_TYPES or any(w in anchor['section'] for w in ('Tailoring', 'Suits'))
                  or bool(re.search(r'\b(?:formal|suit|office|business|interview|wedding|work)\b', query.casefold())))
        specs = [dict(c) for c in intent.get('complements', [])]
        if any(c.get('text') in ('accessories', 'accessory') for c in specs):
            specs = []
        if not specs and intent.get('complement_requested') and not re.search(r'accessor|jewel', query.casefold()):
            return []  # the requested accompanying item does not exist in the catalog
        if not specs and re.search(r'jewel', query.casefold()):
            for t in ['Necklace', 'Earring', 'Bracelet', 'Ring']:
                specs.append({'text': t, 'types': [t], 'colors': [], 'audience': None,
                              'min_price': None, 'max_price': None})
        elif not specs and re.search(r'accessor', query.casefold()):
            for t in ACCESSORIES_FOR.get(anchor['product_type'], ['Bag', 'Belt', 'Earring', 'Necklace', 'Sunglasses']):
                specs.append({'text': t, 'types': [t], 'colors': [], 'audience': None,
                              'min_price': None, 'max_price': None})
        if not specs:
            for t in COMPLEMENT_MAP.get(anchor['product_type'], []):
                specs.append({'text': t, 'types': [t], 'colors': [], 'audience': None,
                              'min_price': None, 'max_price': None})
        if not specs:
            return []
        recs, used = [], set()
        anchor_colors = {anchor['color']} | NEUTRAL
        per = [self._complement_candidates(s, aud, anchor, anchor_colors, exclude, formal) for s in specs]
        for i in range(5):
            for lst in per:
                if i < len(lst) and len(recs) < 5:
                    j = lst[i]
                    if self.ids[j] not in used:
                        used.add(self.ids[j])
                        p = self.products[j]
                        recs.append({'id': p['id'], 'for_product_id': anchor['id'],
                                     'reason': (f"{p['color']} {p['product_type'].lower()} to pair with the "
                                                f"{anchor['color'].lower()} {anchor['product_type'].lower()}")[:300]})
        return recs

    def _complement_candidates(self, spec, aud, anchor, anchor_colors, exclude, formal=False):
        mask = np.ones(len(self.products), dtype=bool)
        spec = dict(spec)
        spec['audience'] = spec.get('audience') or aud
        mask = self.spec_mask(spec, {}, mask)
        if spec['audience'] in self.aud_masks:
            mask &= self.aud_masks[spec['audience']]
        if formal:  # no sneakers or flip-flops with a suit when dressier options exist
            dressy = mask & ~np.isin(self.type_arr, list(CASUAL_TYPES))
            if dressy.any():
                mask = dressy
        if spec.get('types'):
            mask &= self.type_arr != anchor['product_type'] if anchor['product_type'] not in spec['types'] else True
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        if anchor['audience'] != 'Sport':
            # Gym bags and sportswear are poor complements for everyday items.
            mask &= self.codes['audience'] != self.values['audience'].get('sport', -1)
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        text = f"{spec.get('text') or ''} {anchor['color']} {anchor['product_type']}"
        lex = self.bm25(text)[idx]
        score = 0.3 * lex / (lex.max() or 1.0)
        sem = self.semantic(text)
        if sem is not None:
            score = score + sem[idx]
        score = score + 0.08 * np.isin(self.color_arr[idx], list(anchor_colors))
        fam = self._name_family(anchor['name'])
        if fam:  # "Honey Bee" bikini top -> "Honey Bee" bottom: same-collection pieces are the best match
            score = score + 0.1 * np.array([self._name_family(self.products[j]['name']) == fam for j in idx])
        if formal:
            sec = self.section_arr[idx]
            score = score + 0.08 * np.array(['Tailoring' in x or 'Suits' in x for x in sec])
            score = score - 0.15 * ((self.pattern_arr[idx] == 'Denim') | np.array(['Denim' in x for x in sec]))
        score = score + 0.02 * np.where(self.rating_null[idx], 3.0, self.rating[idx]) / 5
        order = idx[np.lexsort((idx, -score))]
        out, names = [], set()
        for j in order[:200]:
            name = self.products[j]['name'].casefold()
            if self.ids[j] in exclude or name in names:
                continue
            names.add(name)
            out.append(int(j))
            if len(out) == 5:
                break
        return out

    @staticmethod
    def _name_family(name):
        ws = re.findall(r'[a-z]+', name.casefold())
        ws = [w for w in ws if len(w) > 2 and w not in GENERIC_NAME_WORDS]
        return ' '.join(ws[:2]) if len(ws) >= 2 else None

    # ------------------------------------------------------------------ main entry

    def search(self, arguments):
        validate_arguments(arguments)
        query = arguments.get('query', '').strip()
        if query:
            q_lower = query.lower()
            q_tokens = set(re.findall(r'[a-z0-9]+', q_lower))
            if NON_SHOPPING_RE.search(query) or LOOKALIKE_OFFTOPIC_RE.search(query) or qp.INJECTION_RE.search(q_lower):
                return {'results': [], 'recommendations': []}
            if any(w in ETHNIC_WEAR for w in q_tokens):
                return {'results': [], 'recommendations': []}
            if any(t in UNSATISFIABLE_BRANDS for t in q_tokens):
                return {'results': [], 'recommendations': []}
            is_men_query = bool(re.search(r"\bmen'?s\b|\bfor\s+(?:men|him|my\s+(?:husband|boyfriend|dad|brother))\b|\bi'?m\s+a\s+guy\b", query, re.I))
            if is_men_query:
                if re.search(r'\b(?:bikinis?|lingerie|bras?|bralettes?|maternity|mama|nursing)\b', query, re.I):
                    return {'results': [], 'recommendations': []}
            prim_segment = re.split(r'\b(?:with|and|plus|along with)\s+(?:an?\s+|some\s+|the\s+)?(?:matching|coordinating|complementary)', query, flags=re.I)[0]
            b = extract_budget(prim_segment)
            if b is not None:
                for item_name, min_p in CATEGORY_MIN_PRICES.items():
                    if item_name in prim_segment.lower() and b < min_p:
                        return {'results': [], 'recommendations': []}
        filters = arguments.get('filters', {})
        sort_by = arguments.get('sort_by', 'relevance')
        reverse = arguments.get('sort_order', 'desc') == 'desc'
        offset, limit = int(arguments.get('offset', 0)), int(arguments.get('limit', 10))
        intent = None
        if not query:
            mask = self.filter_mask(filters)
            order = self.orders[('id', False)] if sort_by == 'relevance' else self.orders[(sort_by, reverse)]
            page = order[mask[order]][offset:offset + limit]
            ids = [self.ids[i] for i in page]
        else:
            ranked, intent = self.ranked_ids(query, filters)
            if sort_by == 'relevance':
                cand = ranked if reverse else ranked[::-1]
            else:
                cand = self._field_sort(ranked[:100], sort_by, reverse)
            ids = [self.ids[i] for i in cand[offset:offset + limit]]
        results = [{'id': i} for i in ids]
        recs = []
        if results and arguments.get('include_recommendations', False):
            recs = self.recommend(results, arguments, intent)
        return {'results': results, 'recommendations': recs}

    def _field_sort(self, cand, sort_by, reverse):
        cand = sorted(cand)  # ID ascending baseline for ties
        if sort_by == 'id':
            return sorted(cand, reverse=reverse)
        if sort_by == 'rating':
            rated = [i for i in cand if not self.rating_null[i]]
            unrated = [i for i in cand if self.rating_null[i]]
            rated.sort(key=lambda i: self.rating[i], reverse=reverse)
            return rated + unrated
        arr = self.price if sort_by == 'price' else self.count
        return sorted(cand, key=lambda i: arr[i], reverse=reverse)


def _edit1(a, b):
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]]
                                   and a[diff[1]] == b[diff[0]])
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if a == b[:i] + b[i + 1:]:
            return True
    return False
