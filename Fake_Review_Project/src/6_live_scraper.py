import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def scrape_flipkart_reviews(product_url, pages=2):
    reviews = []
    # More detailed headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }

    print(f"Targeting: {product_url}")

    for i in range(1, pages + 1):
        url = f"{product_url}&page={i}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"❌ Page {i}: HTTP Error {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # We try multiple common Flipkart review classes
            # Flipkart often changes these: 't-ZTKy' was old, 'ZmyHeS' is common now
            potential_classes = ['t-ZTKy', 'ZmyHeS', '_1AtVfX']
            page_reviews = []

            for cls in potential_classes:
                containers = soup.find_all('div', {'class': cls})
                if containers:
                    for c in containers:
                        text = c.get_text(strip=True).replace('READ MORE', '')
                        if text: page_reviews.append(text)
                    break # Stop if we found reviews with one class

            reviews.extend(page_reviews)
            print(f"✅ Page {i}: Found {len(page_reviews)} reviews.")
            
            # Random delay to look human
            time.sleep(3) 

        except Exception as e:
            print(f"❌ Error on page {i}: {e}")

    return reviews

# Let's try a slightly different URL structure if the first one fails
target_url = "https://www.flipkart.com/apple-iphone-15-black-128-gb/product-reviews/itm6ac6485515ae4?pid=MOBGTAGPTJBZBCYZ"

scraped_data = scrape_flipkart_reviews(target_url, pages=2)

if scraped_data:
    df = pd.DataFrame(scraped_data, columns=['text'])
    os.makedirs('data/raw', exist_ok=True) # Ensure folder exists
    output_path = os.path.join('data', 'raw', 'scraped_reviews.csv')
    df.to_csv(output_path, index=False)
    print(f"\n🔥 SUCCESS! Saved {len(df)} live reviews to {output_path}")
else:
    print("\n❌ Still 0 reviews. Flipkart might be blocking this specific IP or the classes changed again.")