# Multimodal Semantic Pricing & Retrieval 🚀

An AI-powered engine combining State-of-the-Art Vision and Language Models to predict fair market prices for products and retrieve semantically similar items across large e-commerce catalogs. Built for the **Amazon ML Challenge 2025**.

![Dashboard Preview](https://via.placeholder.com/800x400.png?text=Multimodal+Semantic+Search+Dashboard)

## 🌟 Key Features

*   **Multimodal Price Prediction:** Fuses text (titles, descriptions, bullet points) and image (product photos) embeddings, combined with extracted tabular metadata, to regress a highly accurate fair market `log_price`.
*   **Semantic Vector Search:** Indexes embeddings using **FAISS** to instantly retrieve the most semantically related products from the training corpus when given a novel item.
*   **Premium Web Interface:** A fast, responsive, "Light Mode" dashboard built with Vanilla JavaScript and CSS, offering a minimalist yet powerful user experience.
*   **Live Amazon Scraping:** Paste any Amazon URL to automatically pull text/image features and run them through the AI engine locally.

## 🧠 Technology Stack

### Machine Learning & AI
*   **Text Backbone:** [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) (1024-dim embeddings from product descriptions).
*   **Vision Backbone:** [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) (1152-dim embeddings from high-res product images).
*   **Fusion Architecture:** PyTorch *Gated Multimodal MLP* with Deep Residual Blocks and skip connections to learn complex pricing interactions.
*   **Vector Retrieval:** [FAISS](https://github.com/facebookresearch/faiss) L2 Index for lightning-fast k-NN semantic search across thousands of products.
*   **Feature Engineering:** Custom regex and frequency-based Brand extraction and Item-Per-Quantity (IPQ) parsing.

### Backend Application
*   **Framework:** [FastAPI](https://fastapi.tiangolo.com/) & Uvicorn (serving requests concurrently and managing heavy PyTorch models in memory).
*   **Scraping:** Built-in web scraper using `requests` and `BeautifulSoup` to bypass simple blocks and extract Amazon product schemas.
*   **Data Handling:** `pandas`, `numpy`, and Parquet for rapid I/O.

### Frontend
*   **Structure:** HTML5, modern CSS3 (Custom Properties, Flexbox, Grid).
*   **Logic:** Asynchronous Vanilla JavaScript (No heavy frontend frameworks required!).

## 🏗️ Architecture Design

The core Model (in `steps/_model.py`) relies on a **Late Gated Fusion** technique:
1. Extract `[1024]` Text vector.
2. Extract `[1152]` Image vector.
3. Extract `[7]` Tabular Meta vector (IPQ + 6 historical brand pricing stats).
4. Project Text & Image to `[256]` each, and Meta to `[64]`.
5. Concatenate to a `[576]` vector.
6. Pass through deep Residual Blocks (with Dropout and LayerNorm) to predict Price.

Simultaneously, the raw concatenated Text & Image vectors mapped into normalized space are indexed in FAISS, allowing Cosine Similarity/Inner Product search over the catalog.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ and pip installed. A GPU is recommended for fast model inference but not strictly required.

```bash
# Clone the repository
git clone https://github.com/wathoresanket/multimodal-semantic-pricing.git
cd multimodal-semantic-pricing

# Install required packages
pip install -r requirements.txt
```

*(Note: Data files and PyTorch `.pt` checkpoints are ignored via `.gitignore` and must be downloaded from the competition drive and placed in `data/` and `checkpoints/` before running).*

### Running the Infrastructure

1.  **Extract Data & Train Models:** (Sequential Pipeline)
    ```bash
    bash run.sh all
    ```
    *This will extract features, generate DeBERTa and SigLIP embeddings, train the PyTorch model, and build the FAISS index.*

2.  **Launch the Web Application:**
    Once models are trained and checkpoints reside in `/checkpoints/`, launch the FastAPI server:
    ```bash
    python webapp/server.py
    ```

3.  **Access the Dashboard:**
    Open `http://localhost:8000` in your web browser!

## 📜 License
This project was developed for the Amazon ML Challenge 2025.
