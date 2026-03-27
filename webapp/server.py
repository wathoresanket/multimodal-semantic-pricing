"""
FastAPI Server — All API endpoints for the Price Predictor Web App.

Run:
    cd /Data1/sanket/amazon-ml-challenge-2025
    python webapp/server.py
"""
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Price Predictor", version="1.0")

# ── Lazy-loaded singletons ──────────────────────────────────────────────────

predictor = None
index = None


def get_services():
    global predictor, index
    if predictor is None:
        from webapp.inference import get_predictor
        predictor = get_predictor()
    if index is None:
        from webapp.indexer import get_index
        index = get_index()
    return predictor, index


# ── Request Models ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    image_url: str = ""
    image_base64: str = ""


class DealCheckRequest(BaseModel):
    amazon_url: str = ""
    text: str = ""
    image_url: str = ""
    image_base64: str = ""
    listed_price: Optional[float] = None


class ExplainRequest(BaseModel):
    text: str
    image_url: str = ""
    image_base64: str = ""


# ── API Endpoints ───────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_product(req: DealCheckRequest):
    """Master endpoint: predictions, deal check, similar products, and explainability."""
    pred, idx = get_services()
    from webapp.explainer import compute_text_shap, compute_gradcam

    title = req.text
    image_url = req.image_url
    image_base64 = req.image_base64
    listed_price = req.listed_price

    # ── 1. Scrape Amazon if URL provided ──────────────────────────
    bullets = ""
    if req.amazon_url:
        from webapp.scraper import scrape_amazon_product
        scraped = scrape_amazon_product(req.amazon_url)
        if scraped['success']:
            title = scraped['title'] or title
            image_url = scraped['image_url'] or image_url
            bullets = scraped.get('bullets', '')
            if scraped['price'] is not None:
                listed_price = scraped['price']
        elif not title:
            return {
                "success": False,
                "error": scraped['error'] or "Could not scrape Amazon page. Please enter details manually.",
                "scraped": scraped,
            }

    if not title:
        raise HTTPException(400, "Product text is required")

    # ── 2. Predict Fair Price ─────────────────────────────────────
    input_text = f"Item Name: {title}"
    if bullets:
        input_text += f"\nBullet Points: {bullets}"
        
    result = pred.predict(input_text, image_url, image_base64)
    fair_price = result['predicted_price']

    # ── 3. Compute Deal Score ─────────────────────────────────────
    diff_pct = 0
    verdict = "N/A"
    verdict_class = "unknown"

    if listed_price and listed_price > 0:
        if fair_price > 0:
            diff_pct = ((fair_price - listed_price) / fair_price) * 100
            
        if diff_pct > 20:
            verdict = "🟢 Great Deal!"
            verdict_class = "great-deal"
        elif diff_pct > 5:
            verdict = "🟢 Good Deal"
            verdict_class = "good-deal"
        elif diff_pct > -5:
            verdict = "🟡 Fair Price"
            verdict_class = "fair-price"
        elif diff_pct > -20:
            verdict = "🟠 Slightly Overpriced"
            verdict_class = "overpriced"
        else:
            verdict = "🔴 Overpriced"
            verdict_class = "very-overpriced"

    # ── 4. Retrieve Similar Products ──────────────────────────────
    similar = idx.find_similar(result['text_embedding'], result['image_embedding'], k=6)

    # ── 5. Explainability (SHAP + GradCAM) ────────────────────────
    word_importances = compute_text_shap(
        pred, input_text, result['image'],
        result['image_embedding'], fair_price
    )

    gradcam_base64 = ""
    if image_url or image_base64:
        try:
            gradcam_base64 = compute_gradcam(pred, result['image'])
        except Exception as e:
            print(f"[GradCAM] Error: {e}")

    return {
        "success": True,
        "product_title": title,
        "product_image": image_url or image_base64,
        "listed_price": round(listed_price, 2) if listed_price else None,
        "predicted_fair_price": fair_price,
        "deal_score": round(diff_pct, 1),
        "verdict": verdict,
        "verdict_class": verdict_class,
        "confidence_low": result['confidence_low'],
        "confidence_high": result['confidence_high'],
        "similar_products": similar,
        "word_importances": word_importances,
        "gradcam_image": gradcam_base64,
    }


# ── Serve Static Files ──────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
