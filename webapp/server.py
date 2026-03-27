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
    """Master endpoint: predictions and semantic search."""
    pred, idx = get_services()

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

    # ── 3. Retrieve Similar Products ──────────────────────────────
    similar = idx.find_similar(result['text_embedding'], result['image_embedding'], k=12)

    return {
        "success": True,
        "product_title": title,
        "product_image": image_url or image_base64,
        "predicted_fair_price": fair_price,
        "similar_products": similar
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
