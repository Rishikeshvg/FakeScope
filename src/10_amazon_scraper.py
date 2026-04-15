import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random

def get_amazon_reviews(asin, pages=2):
    reviews = []
    # Updated 2026 headers to look like a modern Windows 11 Chrome browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/',
        'Device-Memory': '8'
    }

    print(f"🚀 Starting extraction for ASIN: {asin}")

    for page in range(1, pages + 1):
        # Try the most stable review URL format for 2026
        url = f"https://www.amazon.in/product-reviews/{asin}/?pageNumber={page}&sortBy=recent"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 404:
                print(f"❌ Page {page}: 404 Not Found. ASIN might be invalid or variation-locked.")
                break
            elif response.status_code != 200:
                print(f"❌ Page {page}: Status {response.status_code} (Likely a bot-block)")
                break
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Use 'select' with current 2026 data-hooks
            review_elements = soup.select('div[data-hook="review"]')

            if not review_elements:
                print(f"⚠️ Page {page}: No review elements found. Checking if page layout changed...")
                # Fallback: Sometimes reviews are in a different list ID
                review_elements = soup.select('.a-section.review')

            for el in review_elements:
                try:
                    # Extract review body
                    body_el = el.select_one('span[data-hook="review-body"]')
                    text = body_el.text.strip() if body_el else ""
                    
                    # Extract verified status
                    verified_el = el.select_one('span[data-hook="avp-verified-purchase"]')
                    status = "Verified" if verified_el else "Unverified"
                    
                    if text:
                        reviews.append({
                            'text': text.replace('Read more', '').strip(),
                            'status': status
                        })
                except Exception:
                    continue
            
            print(f"✅ Page {page}: Found {len(review_elements)} review blocks.")
            time.sleep(random.uniform(2, 5)) # Random delay is key to not get blocked

        except Exception as e:
            print(f"⚠️ Connection Error on page {page}: {e}")
            break

    return reviews

# --- EXECUTION ---
# Using a high-volume ASIN for stability
current_asin = "B0B8S6K3R5" 
data = get_amazon_reviews(current_asin)

if data:
    df = pd.DataFrame(data)
    output_path = os.path.join('data', 'raw', 'scraped_reviews.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n🔥 SUCCESS! Saved {len(df)} reviews to {output_path}")
else:
    print("\n❌ Extraction failed. Try a different ASIN like B07HGH8D7W.")