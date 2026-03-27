"""
Step 1 — Feature Extraction (Regex Brand Intelligence Engine)
    • Extracts item_name from catalog_content (the field brand lives in)
    • Builds unigram + bigram brand vocabulary from training data
    • Three-pass brand extraction (bigram → unigram → by/from pattern)
    • Bayesian-smoothed brand stats → 6 brand features
    • Extracts IPQ (Item Pack Quantity) via regex
    • Saves parquet files for all downstream steps

WHY THIS IS BETTER THAN GLINER:
    • 100x faster — pure Python regex, no GPU needed, runs in seconds not 30+ min
    • 6 rich brand features vs 1 single encoded value:
        brand_count, brand_freq_log, brand_smooth_mean, brand_std,
        brand_premium, is_known_brand
    • brand_premium (smoothed_mean / global_mean) directly tells the model
        whether a brand is budget or luxury — a signal GLiNER never captured
    • No model download, no CUDA memory overhead for this step

Run:
    nohup python steps/01_extract_features.py > logs/01_extract_features.out 2>&1 &
    tail -f logs/01_extract_features.out
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from utils.logger import get_logger

log = get_logger("01_extract_features")

# ── Generic words that are NOT brands ─────────────────────────────────────────
GENERIC_WORDS = {
    'pack', 'set', 'combo', 'kit', 'bundle', 'box', 'case',
    'pair', 'lot', 'batch', 'collection', 'assorted', 'variety',
    'new', 'premium', 'pure', 'original', 'classic', 'standard',
    'regular', 'basic', 'generic', 'unbranded', 'imported',
    'branded', 'best', 'super', 'ultra', 'pro', 'plus', 'max',
    'mini', 'micro', 'old', 'big', 'great', 'real', 'true',
    'wild', 'raw', 'dry', 'hot', 'cold', 'warm', 'soft', 'hard',
    'light', 'dark', 'rich', 'mild', 'bold', 'smooth', 'crispy',
    'whole', 'certified', 'all', 'made', 'good', 'bulk', 'deluxe',
    'black', 'white', 'red', 'blue', 'green', 'brown', 'pink',
    'silver', 'gold', 'wooden', 'plastic', 'metal', 'cotton',
    'men', 'women', 'boys', 'girls', 'kids', 'adult',
    'mens', 'womens', 'unisex', 'ladies',
    'item', 'product', 'the', 'a', 'an', 'and', 'or', 'for',
    'with', 'by', 'food', 'tea', 'coffee', 'gourmet', 'candy',
    'chocolate', 'sugar', 'sauce', 'spice', 'simply', 'sweet',
    'nature', 'la', 'bob', 'eden', 'slim', 'log', 'bakery',
    'crystal', 'amazon', 'marshalls', 'ranch', 'dairy',
    'mountain', 'royal', 'organic', 'fresh', 'natural',
    'fruit', 'milk', 'snack', 'vanilla', 'rice', 'honey',
    'coconut', 'peanut', 'orange', 'lemon', 'bar', 'dried',
    'peach', 'kettle', 'ginger', 'strawberry', 'raspberry',
    'pineapple', 'apple', 'mango', 'berry', 'grape', 'cherry',
    'garlic', 'onion', 'pepper', 'chili', 'butter', 'cream',
    'cheese', 'egg', 'wheat', 'corn', 'oat', 'bean', 'nut',
    'seed', 'herb', 'mint', 'cinnamon', 'lime', 'gift', 'mix',
    'blend', 'powder', 'oil', 'from', 'blueberry', 'ben',
    'cranberry', 'island', 'owen', 'apricot', 'peppermint',
    '100', '200', '500', '1000', '1', '2', '3', '4', '5', '6', '12', '24',
}


# ── IPQ extraction ─────────────────────────────────────────────────────────────

def extract_ipq(text: str) -> float:
    """
    Extract Item Pack Quantity from catalog_content.
    Searches for 'Value: <number>' pattern first; falls back to first
    numeric value found; defaults to 1.0 if nothing found.
    """
    text = str(text)
    m = re.search(r'Value:\s*([\d\.]+)', text)
    if m:
        return float(m.group(1))
    # Fallback: first standalone number in the text
    m2 = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
    if m2:
        val = float(m2.group(1))
        return val if val > 0 else 1.0
    return 1.0


# ── Item name extraction ───────────────────────────────────────────────────────

def extract_item_name(text: str) -> str:
    """
    catalog_content is structured: "Item Name: X\nBullet Point 1: ..."
    Brand lives in Item Name only — extract that field.
    Falls back to first 10 tokens if pattern not found.
    """
    if not isinstance(text, str):
        return ''
    m = re.search(
        r'[Ii]tem\s+[Nn]ame\s*:\s*(.+?)(?:\n|\\n|Bullet|Value:|Unit:|$)',
        text
    )
    if m:
        return m.group(1).strip()
    return ' '.join(text.split()[:10])


def clean_name(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Brand token normalisation ──────────────────────────────────────────────────

def normalize_brand_token(token: str) -> str:
    """
    Strip possessives and non-alphanumeric chars.
    Scott's → scotts,  L'Oréal → loreal,  Häagen → haagen
    """
    token = token.lower()
    token = re.sub(r"'s$|'s$", '', token)   # possessive
    token = re.sub(r"'", '', token)           # remaining apostrophes
    token = re.sub(r'[^a-z0-9]', '', token)  # non-alphanumeric
    return token


# ── Vocabulary builders ────────────────────────────────────────────────────────

def build_unigram_vocab(item_names: pd.Series) -> set:
    """
    Vocabulary = first tokens that appear >= BRAND_UNIGRAM_MIN_FREQ times
    and are not in GENERIC_WORDS and are >= BRAND_MIN_TOKEN_LEN chars.
    """
    first_tokens = item_names.str.split().str[0].fillna('')
    normed = first_tokens.apply(normalize_brand_token)
    freq   = normed.value_counts()
    vocab  = set(
        freq[
            (freq >= config.BRAND_UNIGRAM_MIN_FREQ) &
            (~freq.index.isin(GENERIC_WORDS)) &
            (freq.index.str.len() >= config.BRAND_MIN_TOKEN_LEN)
        ].index.tolist()
    )
    log.info(f"Unigram vocab size: {len(vocab)}")
    log.info(f"Top 20 unigrams:\n{freq[freq.index.isin(vocab)].head(20).to_string()}")
    return vocab


def build_bigram_vocab(item_names: pd.Series) -> set:
    """
    Bigram vocabulary of <first token> + <second token> pairs.
    Handles two-word brand names like 'del monte', 'gold medal'.
    Lower threshold than unigrams since there are fewer products per exact pair.
    """
    def get_bigram(name: str) -> str:
        tokens = str(name).split()
        if len(tokens) >= 2:
            return normalize_brand_token(tokens[0]) + ' ' + normalize_brand_token(tokens[1])
        return ''

    bigrams = item_names.apply(get_bigram)
    freq    = bigrams[bigrams != ''].value_counts()
    vocab   = set(
        freq[
            (freq >= config.BRAND_BIGRAM_MIN_FREQ) &
            (freq.index.str.len() >= 5)
        ].index.tolist()
    )
    log.info(f"Bigram vocab size: {len(vocab)}")
    log.info(f"Top 20 bigrams:\n{freq[freq.index.isin(vocab)].head(20).to_string()}")
    return vocab


# ── Three-pass brand extraction ────────────────────────────────────────────────

def extract_brand(item_name: str, unigram_vocab: set, bigram_vocab: set) -> str:
    """
    Pass 1 — first two tokens as bigram  (e.g. "del monte")
    Pass 2 — first token as unigram      (e.g. "heinz")
    Pass 3 — "by Brand" / "from Brand"   (e.g. "Pasta by Barilla")
    Falls back to 'unknown' if nothing matches.
    """
    if not isinstance(item_name, str) or item_name.strip() == '':
        return 'unknown'

    tokens      = item_name.split()
    norm_tokens = [normalize_brand_token(t) for t in tokens]

    # Pass 1: bigram
    if len(norm_tokens) >= 2:
        bigram = norm_tokens[0] + ' ' + norm_tokens[1]
        if bigram in bigram_vocab:
            return bigram

    # Pass 2: unigram
    if norm_tokens and norm_tokens[0] in unigram_vocab:
        return norm_tokens[0]

    # Pass 3: "by Brand" / "from Brand" pattern
    m = re.search(r'\b(?:by|from)\s+(\w+)', item_name.lower())
    if m:
        candidate = normalize_brand_token(m.group(1))
        if candidate in unigram_vocab:
            return candidate

    return 'unknown'


# ── Bayesian-smoothed brand stats ──────────────────────────────────────────────

def compute_brand_stats(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    """
    Compute per-brand stats + Bayesian-smoothed mean.
    smoothed_mean = (n * brand_mean + K * global_mean) / (n + K)
    A brand with 1 product → heavily pulled toward global mean.
    A brand with 100+ products → trusted.
    """
    K = config.BAYESIAN_SMOOTHING_K
    stats = (
        df[df['brand'] != 'unknown']
        .groupby('brand')['price']
        .agg(brand_count='count', brand_mean='mean',
             brand_median='median', brand_std='std')
        .reset_index()
    )
    stats['brand_smooth_mean'] = (
        (stats['brand_count'] * stats['brand_mean'] + K * global_mean) /
        (stats['brand_count'] + K)
    )
    stats['brand_std']      = stats['brand_std'].fillna(0)
    stats['brand_freq_log'] = np.log1p(stats['brand_count'])
    stats['brand_premium']  = (stats['brand_smooth_mean'] / global_mean).round(4)
    return stats


def make_brand_features(df: pd.DataFrame, brand_lookup: pd.DataFrame,
                        global_mean: float) -> pd.DataFrame:
    """
    Map brand names → 6 numeric features per row.
    Unknown brands get: count=0, freq_log=0, smooth_mean=global_mean,
                        std=0, premium=1.0, is_known=0
    """
    b     = df['brand']
    feats = pd.DataFrame(index=df.index)
    feats['brand']             = b
    feats['brand_count']       = b.map(brand_lookup['brand_count']).fillna(0)
    feats['brand_freq_log']    = b.map(brand_lookup['brand_freq_log']).fillna(0)
    feats['brand_smooth_mean'] = b.map(brand_lookup['brand_smooth_mean']).fillna(global_mean)
    feats['brand_std']         = b.map(brand_lookup['brand_std']).fillna(0)
    feats['brand_premium']     = b.map(brand_lookup['brand_premium']).fillna(1.0)
    feats['is_known_brand']    = (feats['brand_count'] > 0).astype(int)
    return feats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading CSVs …")
    train = pd.read_csv(config.TRAIN_CSV)
    test  = pd.read_csv(config.TEST_CSV)
    log.info(f"Train: {len(train)} rows | Test: {len(test)} rows")

    # ── IPQ extraction ────────────────────────────────────────────────────────
    log.info("Extracting IPQ values …")
    train['ipq_value'] = train['catalog_content'].apply(extract_ipq)
    test['ipq_value']  = test['catalog_content'].apply(extract_ipq)
    log.info(f"IPQ — train mean={train['ipq_value'].mean():.2f}, "
             f"max={train['ipq_value'].max()}, nulls={train['ipq_value'].isna().sum()}")

    # ── Item name extraction ──────────────────────────────────────────────────
    log.info("Extracting item_name from catalog_content …")
    train['item_name'] = train['catalog_content'].apply(extract_item_name)
    test['item_name']  = test['catalog_content'].apply(extract_item_name)
    log.info("Sample item names:")
    for row in train[['item_name']].head(5).itertuples():
        log.info(f"  {row.item_name}")

    # ── Build vocab from TRAIN ONLY (no leakage) ─────────────────────────────
    log.info("Building brand vocabulary from training set …")
    unigram_vocab = build_unigram_vocab(train['item_name'])
    bigram_vocab  = build_bigram_vocab(train['item_name'])
    brand_vocab   = unigram_vocab.union(bigram_vocab)

    # ── Brand extraction ──────────────────────────────────────────────────────
    log.info("Running brand extraction (3-pass regex) …")
    train['brand'] = train['item_name'].apply(
        lambda x: extract_brand(x, unigram_vocab, bigram_vocab)
    )
    test['brand'] = test['item_name'].apply(
        lambda x: extract_brand(x, unigram_vocab, bigram_vocab)
    )

    train_cov = (train['brand'] != 'unknown').mean()
    test_cov  = (test['brand'] != 'unknown').mean()
    log.info(f"Brand coverage — train: {train_cov*100:.1f}%  |  test: {test_cov*100:.1f}%")
    log.info(f"Top 25 brands:\n{train['brand'].value_counts().head(25).to_string()}")

    # ── Bayesian brand stats (train only) ─────────────────────────────────────
    log.info("Computing Bayesian-smoothed brand stats …")
    global_mean = train['price'].mean()
    brand_stats = compute_brand_stats(train, global_mean)
    brand_lookup = brand_stats.set_index('brand')[[
        'brand_count', 'brand_freq_log', 'brand_smooth_mean',
        'brand_std', 'brand_premium'
    ]]

    log.info(f"Global mean price: {global_mean:.2f}")
    log.info(f"Top 10 premium brands:\n"
             f"{brand_stats.nlargest(10, 'brand_premium')[['brand','brand_count','brand_premium']].to_string(index=False)}")
    log.info(f"Top 10 budget brands:\n"
             f"{brand_stats.nsmallest(10, 'brand_premium')[['brand','brand_count','brand_premium']].to_string(index=False)}")

    # ── Build feature frames ──────────────────────────────────────────────────
    log.info("Building brand feature columns …")
    train_brand_feats = make_brand_features(train, brand_lookup, global_mean)
    test_brand_feats  = make_brand_features(test, brand_lookup, global_mean)

    corr         = train_brand_feats['brand_smooth_mean'].corr(train['price'])
    unknown_rate = (train_brand_feats['is_known_brand'] == 0).mean()
    log.info(f"brand_smooth_mean vs price correlation: {corr:.3f}  (expect 0.3–0.6)")
    log.info(f"Unknown brand rate (train): {unknown_rate*100:.1f}%  (expect 40–50%)")

    # ── Log-price for train ───────────────────────────────────────────────────
    log_price_train = pd.DataFrame({
        'log_price': np.log1p(train['price'])
    }, index=train.index)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(config.DATA_DIR, exist_ok=True)

    train_brand_feats.to_parquet(config.BRAND_FEATS_TRAIN)
    test_brand_feats.to_parquet(config.BRAND_FEATS_TEST)

    train[['item_name']].to_parquet(config.ITEM_NAME_TRAIN)
    test[['item_name']].to_parquet(config.ITEM_NAME_TEST)

    train[['ipq_value']].to_parquet(config.IPQ_TRAIN)
    test[['ipq_value']].to_parquet(config.IPQ_TEST)

    log_price_train.to_parquet(config.LOG_PRICE_TRAIN)

    with open(config.BRAND_VOCAB_PKL, 'wb') as f:
        pickle.dump(brand_vocab, f)
    with open(config.BRAND_STATS_PKL, 'wb') as f:
        pickle.dump(brand_stats, f)

    # Also save sample_ids for alignment in later steps
    train[['sample_id']].to_parquet(os.path.join(config.DATA_DIR, "sample_id_train.parquet"))
    test[['sample_id']].to_parquet(os.path.join(config.DATA_DIR, "sample_id_test.parquet"))

    log.info("── Saved ──────────────────────────────────────────")
    log.info(f"  {config.BRAND_FEATS_TRAIN}")
    log.info(f"  {config.BRAND_FEATS_TEST}")
    log.info(f"  {config.ITEM_NAME_TRAIN}  ← Step 2 embeds this, not full catalog")
    log.info(f"  {config.ITEM_NAME_TEST}")
    log.info(f"  {config.IPQ_TRAIN} / {config.IPQ_TEST}")
    log.info(f"  {config.LOG_PRICE_TRAIN}")
    log.info(f"  {config.BRAND_VOCAB_PKL}")
    log.info(f"  {config.BRAND_STATS_PKL}")


if __name__ == "__main__":
    main()
