"""
Amazon Product Scraper — Extract product info from Amazon URLs.

Uses cloudscraper to bypass bot protection.
Extracts title, image, price, and bullet points.
"""
import re
import cloudscraper
from bs4 import BeautifulSoup


def scrape_amazon_product(url: str) -> dict:
    """
    Scrape product title, image URL, price, and bullets from an Amazon product page.
    
    Returns: {title, image_url, price, bullets, success, error}
    """
    result = {
        "title": "",
        "image_url": "",
        "price": None,
        "bullets": "",
        "success": False,
        "error": None,
    }

    try:
        scraper = cloudscraper.create_scraper(browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        })
        resp = scraper.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'lxml')

        # ── Title ─────────────────────────────────────────────
        title_el = soup.find('span', id='productTitle')
        if title_el:
            result['title'] = title_el.get_text(strip=True)
        else:
            # Fallback
            title_el = soup.find('h1', id='title')
            if title_el:
                result['title'] = title_el.get_text(strip=True)

        # ── Image ─────────────────────────────────────────────
        img_el = soup.find('img', id='landingImage')
        if img_el and img_el.get('src'):
            result['image_url'] = img_el['src']
        else:
            img_el = soup.find('img', id='imgBlkFront')
            if img_el and img_el.get('src'):
                result['image_url'] = img_el['src']
            else:
                # Try data-old-hires or data-a-dynamic-image
                for img in soup.find_all('img'):
                    src = img.get('data-old-hires', '') or img.get('src', '')
                    if 'images-amazon' in src and 'icon' not in src.lower():
                        result['image_url'] = src
                        break

        # ── Price ─────────────────────────────────────────────
        price_el = soup.find('span', class_='a-price-whole')
        if price_el:
            price_text = price_el.get_text(strip=True).replace(',', '').rstrip('.')
            fraction_el = soup.find('span', class_='a-price-fraction')
            if fraction_el:
                price_text += '.' + fraction_el.get_text(strip=True)
            try:
                result['price'] = float(price_text)
            except ValueError:
                pass

        if not result['price']:
            # Try alternative price selectors
            for selector in ['#priceblock_ourprice', '#priceblock_dealprice',
                           '.a-price .a-offscreen', '#price_inside_buybox']:
                el = soup.select_one(selector)
                if el:
                    match = re.search(r'[\d,.]+', el.get_text())
                    if match:
                        try:
                            result['price'] = float(match.group().replace(',', ''))
                            break
                        except ValueError:
                            continue

        # ── Bullets & Description ─────────────────────────────
        bullets = []
        feature_bullets = soup.find(id='feature-bullets')
        if feature_bullets:
            for li in feature_bullets.find_all('li'):
                text = li.get_text(strip=True)
                if text and 'make sure this fits' not in text.lower() and text != '›See more product details':
                    bullets.append(text)
        
        if not bullets:
            desc_el = soup.find(id='productDescription')
            if desc_el:
                bullets.append(desc_el.get_text(strip=True))

        result['bullets'] = " ".join(bullets)

        result['success'] = bool(result['title'])
        if not result['success']:
            result['error'] = "Could not extract product information. Amazon may have blocked the request."

    except Exception as e:
        result['error'] = f"Scraping error: {str(e)}"

    return result

