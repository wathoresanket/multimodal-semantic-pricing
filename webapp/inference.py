"""
Inference Pipeline — Loads all models and runs end-to-end prediction.

Singleton pattern: call `get_predictor()` once at startup.
"""
import sys, os, re, io, pickle
import numpy as np

# ── Numpy 1.x ↔ 2.x pickle compatibility ────────────────────────────────
# Pickles saved with numpy 2.x reference numpy._core which doesn't exist in 1.x
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
from PIL import Image
import requests

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from steps._model import GatedMultimodalMLP

# ── Reuse feature extraction logic from 01_extract_features.py ──────────────

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


def normalize_brand_token(token: str) -> str:
    token = token.lower()
    token = re.sub(r"'s$|'s$", '', token)
    token = re.sub(r"'", '', token)
    token = re.sub(r'[^a-z0-9]', '', token)
    return token


def extract_ipq(text: str) -> float:
    text = str(text)
    m = re.search(r'Value:\s*([\d\.]+)', text)
    if m:
        return float(m.group(1))
    m2 = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
    if m2:
        val = float(m2.group(1))
        return val if val > 0 else 1.0
    return 1.0


def extract_item_name(text: str) -> str:
    if not isinstance(text, str):
        return ''
    m = re.search(
        r'[Ii]tem\s+[Nn]ame\s*:\s*(.+?)(?:\n|\\n|Bullet|Value:|Unit:|$)',
        text
    )
    if m:
        return m.group(1).strip()
    return ' '.join(text.split()[:10])


def extract_brand(item_name: str, unigram_vocab: set, bigram_vocab: set) -> str:
    if not isinstance(item_name, str) or item_name.strip() == '':
        return 'unknown'
    tokens = item_name.split()
    norm_tokens = [normalize_brand_token(t) for t in tokens]
    if len(norm_tokens) >= 2:
        bigram = norm_tokens[0] + ' ' + norm_tokens[1]
        if bigram in bigram_vocab:
            return bigram
    if norm_tokens and norm_tokens[0] in unigram_vocab:
        return norm_tokens[0]
    m = re.search(r'\b(?:by|from)\s+(\w+)', item_name.lower())
    if m:
        candidate = normalize_brand_token(m.group(1))
        if candidate in unigram_vocab:
            return candidate
    return 'unknown'


def clean_catalog_content(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'(Item Name:|Bullet Point \d+:|Product Description:|Value:|Unit:)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class PricePredictor:
    """Loads all models and provides end-to-end prediction."""

    def __init__(self):
        print("[PricePredictor] Loading models... (this takes ~30s)")

        # ── Load brand vocab & stats ─────────────────────────────
        with open(config.BRAND_VOCAB_PKL, 'rb') as f:
            self.brand_vocab = pickle.load(f)
        with open(config.BRAND_STATS_PKL, 'rb') as f:
            self.brand_stats = pickle.load(f)

        # Separate unigram and bigram vocabs
        self.unigram_vocab = {b for b in self.brand_vocab if ' ' not in b}
        self.bigram_vocab = {b for b in self.brand_vocab if ' ' in b}

        # Brand lookup table
        self.brand_lookup = self.brand_stats.set_index('brand')[[
            'brand_count', 'brand_freq_log', 'brand_smooth_mean',
            'brand_std', 'brand_premium'
        ]]
        self.global_mean_price = self.brand_stats['brand_smooth_mean'].mean()

        # ── Load meta scaler ─────────────────────────────────────
        with open(config.META_SCALER, 'rb') as f:
            self.meta_scaler = pickle.load(f)

        # ── Load text model (DeBERTa-v3-large) ──────────────────
        from transformers import AutoModel, AutoTokenizer
        TEXT_MODEL = "microsoft/deberta-v3-large"
        print(f"[PricePredictor] Loading text model: {TEXT_MODEL}")
        self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=False)
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL).to(config.DEVICE)
        self.text_model.eval()

        # ── Load vision model (SigLIP-2 SO400M) ─────────────────
        from transformers import AutoImageProcessor, SiglipVisionModel
        VISION_MODEL = "google/siglip2-so400m-patch14-384"
        print(f"[PricePredictor] Loading vision model: {VISION_MODEL}")
        self.img_processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
        self.img_model = SiglipVisionModel.from_pretrained(VISION_MODEL).to(config.DEVICE)
        self.img_model.eval()

        # ── Load price prediction model ──────────────────────────
        text_dim = 1024  # DeBERTa-v3-large hidden size
        img_dim = 1152   # SigLIP-2 SO400M pooler output
        self.price_model = GatedMultimodalMLP(
            text_dim=text_dim,
            img_dim=img_dim,
            meta_dim=config.META_DIM,
        ).to(config.DEVICE)
        self.price_model.load_state_dict(
            torch.load(config.MODEL_CKPT, map_location=config.DEVICE)
        )
        self.price_model.eval()

        print("[PricePredictor] All models loaded ✓")

    def _embed_text(self, text: str) -> np.ndarray:
        """Generate text embedding from raw catalog content."""
        cleaned = clean_catalog_content(text)
        inputs = self.text_tokenizer(
            [cleaned], padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.text_model(**inputs)
        return out.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Generate image embedding from PIL Image."""
        with torch.no_grad():
            inputs = self.img_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
            out = self.img_model(**inputs)
        return out.pooler_output.cpu().numpy().flatten()

    def _download_image(self, url: str) -> Image.Image:
        """Download image from URL, fallback to dummy."""
        if not url or not url.startswith('http'):
            return Image.new("RGB", (384, 384), (200, 200, 200))
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"[PricePredictor] Image download failed: {e}")
            return Image.new("RGB", (384, 384), (200, 200, 200))

    def _extract_meta(self, text: str) -> np.ndarray:
        """Extract 7 meta features from text."""
        item_name = extract_item_name(text)
        brand = extract_brand(item_name, self.unigram_vocab, self.bigram_vocab)
        ipq = extract_ipq(text)

        # Brand features  
        if brand in self.brand_lookup.index:
            row = self.brand_lookup.loc[brand]
            brand_feats = [
                row['brand_count'], row['brand_freq_log'],
                row['brand_smooth_mean'], row['brand_std'],
                row['brand_premium'], 1.0  # is_known_brand
            ]
        else:
            brand_feats = [0.0, 0.0, self.global_mean_price, 0.0, 1.0, 0.0]

        meta = np.array([ipq] + brand_feats, dtype=np.float32).reshape(1, -1)
        return self.meta_scaler.transform(meta).flatten()

    def predict(self, text: str, image_url: str = "", image_base64: str = "") -> dict:
        """Full pipeline: text + image → predicted price."""
        # Embeddings
        text_emb = self._embed_text(text)
        
        image = None
        if image_base64:
            try:
                import base64
                img_b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
                image_data = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                print(f"[PricePredictor] Warning: Failed to parse base64 image: {e}")
                image = self._download_image(image_url)
        else:
            image = self._download_image(image_url)
            
        img_emb = self._embed_image(image)
        meta = self._extract_meta(text)

        # Price prediction
        t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        i = torch.tensor(img_emb, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        m = torch.tensor(meta, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            log_price = self.price_model(t, i, m).item()

        price = float(np.expm1(log_price))
        price = max(price, 0.10)

        # Confidence range (~20% band)
        low = price * 0.80
        high = price * 1.20

        return {
            "predicted_price": round(price, 2),
            "log_price": round(log_price, 4),
            "confidence_low": round(low, 2),
            "confidence_high": round(high, 2),
            "text_embedding": text_emb,
            "image_embedding": img_emb,
            "image": image,
        }

    def predict_with_custom_text(self, text: str, image: Image.Image, img_emb: np.ndarray = None) -> float:
        """Predict price with custom text but reuse image embedding for SHAP speedup."""
        text_emb = self._embed_text(text)
        if img_emb is None:
            img_emb = self._embed_image(image)
        meta = self._extract_meta(text)

        t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        i = torch.tensor(img_emb, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        m = torch.tensor(meta, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            log_price = self.price_model(t, i, m).item()

        return float(np.expm1(log_price))


# ── Singleton ────────────────────────────────────────────────────────────────

_predictor = None

def get_predictor() -> PricePredictor:
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor()
    return _predictor
