/* ═══════════════════════════════════════════════════════════════════════════
   PriceVision AI — Unified Dashboard Logic
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Utility Functions ───────────────────────────────────────────────────

function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    btn.disabled = loading;
    text.style.display = loading ? 'none' : 'inline';
    loader.style.display = loading ? 'inline-block' : 'none';
    
    // Toggle global loading state
    document.getElementById('loading-state').style.display = loading ? 'flex' : 'none';
    if (loading) {
        document.getElementById('results-dashboard').style.display = 'none';
    }
}

function formatPrice(price) {
    if (price === null || price === undefined) return 'N/A';
    return '$' + Number(price).toFixed(2);
}

// ── Master Analyze Endpoint ─────────────────────────────────────────────

async function getBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

async function analyzeProduct() {
    const amazonUrl = document.getElementById('input-url').value.trim();
    const text = document.getElementById('input-text').value.trim();
    const imageUrl = document.getElementById('input-image').value.trim();
    
    let imageBase64 = "";
    const fileInput = document.getElementById('input-image-file');
    if (fileInput.files && fileInput.files[0]) {
        imageBase64 = await getBase64(fileInput.files[0]);
    }

    if (!amazonUrl && !text) {
        showToast('Please enter an Amazon URL or product details manually');
        return;
    }

    setLoading(amazonUrl ? 'btn-analyze' : 'btn-analyze-manual', true);

    try {
        const reqData = {
            amazon_url: amazonUrl,
            text: text,
            image_url: imageUrl,
            image_base64: imageBase64,
            listed_price: null,
        };

        const resp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqData),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${resp.status})`);
        }

        const data = await resp.json();

        if (!data.success) {
            showToast(data.error || 'Could not analyze product');
            return;
        }

        renderDashboard(data);

    } catch (err) {
        showToast(err.message);
    } finally {
        setLoading(amazonUrl ? 'btn-analyze' : 'btn-analyze-manual', false);
    }
}

// ── Render Dashboard ────────────────────────────────────────────────────

function renderDashboard(data) {
    // 1. Overview Section
    document.getElementById('res-title').textContent = data.product_title || "Unknown Product Details";
    
    const imgEl = document.getElementById('res-image');
    if (data.product_image) {
        imgEl.src = data.product_image;
        imgEl.style.display = 'block';
    } else {
        imgEl.style.display = 'none';
    }

    document.getElementById('res-fair').textContent = formatPrice(data.predicted_fair_price);

    // 2. Similar Products
    renderSimilarProducts('res-similar', data.similar_products);

    // Show Dashboard
    document.getElementById('results-dashboard').style.display = 'block';
}

// ── Render Helpers ──────────────────────────────────────────────────────

function renderSimilarProducts(containerId, products) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (!products || products.length === 0) return;

    products.forEach(p => {
        const card = document.createElement('div');
        card.className = 'similar-card';

        const imgSrc = p.image_url || '';
        const imgHtml = imgSrc 
            ? `<img class="similar-card-img" src="${escapeHtml(imgSrc)}" alt="Product" loading="lazy" onerror="this.style.display='none'">`
            : `<div class="similar-card-img" style="background:rgba(139,92,246,0.05);display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:24px">📦</div>`;

        card.innerHTML = `
            ${imgHtml}
            <div class="similar-card-body">
                <div class="similar-card-name">${escapeHtml(p.name)}</div>
                <div class="similar-card-footer">
                    <span class="similar-card-price">${formatPrice(p.price)}</span>
                    <span class="similar-card-sim">${p.similarity}% match</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

function renderShapWords(containerId, importances) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (!importances || importances.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No word importances available</p>';
        return;
    }

    const maxImp = Math.max(...importances.map(w => Math.abs(w.importance)));

    importances.forEach(w => {
        const span = document.createElement('span');
        span.className = 'shap-word';

        const intensity = maxImp > 0 ? Math.abs(w.importance) / maxImp : 0;
        const alpha = Math.min(0.8, intensity * 0.8 + 0.1);

        if (w.importance > 0) {
            span.style.background = `rgba(239, 68, 68, ${alpha})`;
            span.style.color = intensity > 0.3 ? '#fff' : '#fca5a5';
        } else {
            span.style.background = `rgba(59, 130, 246, ${alpha})`;
            span.style.color = intensity > 0.3 ? '#fff' : '#93c5fd';
        }

        const sign = w.importance > 0 ? '+' : '';
        span.innerHTML = `${escapeHtml(w.word)}<span class="tooltip">${sign}$${w.importance.toFixed(2)}</span>`;
        container.appendChild(span);
    });
}

function renderShapBars(containerId, importances) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (!importances || importances.length === 0) return;

    const top = importances.slice(0, 5); // Show top 5 to save space in unified view
    const maxImp = Math.max(...top.map(w => Math.abs(w.importance)));

    top.forEach(w => {
        const row = document.createElement('div');
        row.className = 'shap-bar-row';

        const widthPct = maxImp > 0 ? (Math.abs(w.importance) / maxImp * 50) : 0;
        const isPositive = w.importance > 0;
        const sign = isPositive ? '+' : '';

        row.innerHTML = `
            <span class="shap-bar-label">${escapeHtml(w.word)}</span>
            <div class="shap-bar-track">
                <div class="shap-bar-fill ${isPositive ? 'positive' : 'negative'}" 
                     style="width:${widthPct}%"></div>
            </div>
            <span class="shap-bar-value">${sign}$${w.importance.toFixed(2)}</span>
        `;
        container.appendChild(row);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
