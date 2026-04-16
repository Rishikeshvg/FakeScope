import csv
import random
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def scrape_meesho_reviews(url, min_reviews=15):
    output_path = os.path.join('data', 'raw', 'scraped_reviews.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print(f"🚀 Opening Meesho...")
        driver.get(url)
        time.sleep(5) 

        reviews = []
        
        # We'll scroll down to the reviews section
        driver.execute_script("window.scrollTo(0, 1500);")
        time.sleep(2)

        print("🔍 Extracting Reviews...")
        
        # Meesho reviews are usually in spans or paragraphs within a specific container
        # We will look for all common text containers
        potential_elements = driver.find_elements(By.TAG_NAME, "p")
        potential_elements += driver.find_elements(By.TAG_NAME, "span")

        for el in potential_elements:
            val = el.text.strip()
            # Logic: A real Meesho review is usually a short sentence (20-300 chars)
            # We filter out UI words like 'Free Delivery', 'Add to Cart', etc.
            if 20 < len(val) < 400:
                noise_words = ['delivery', 'cart', 'buy now', 'off', 'ratings', 'reviews', 'meesho']
                if not any(x in val.lower() for x in noise_words):
                    if val not in reviews:
                        reviews.append(val)
            
            if len(reviews) >= min_reviews:
                break

        # 3. SAVE RESULTS
        if reviews:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['text'])
                for r in reviews:
                    writer.writerow([r.replace('\n', ' ')])
            print(f"✅ SUCCESS! Extracted {len(reviews)} reviews from Meesho.")
        else:
            print("❌ Still 0 reviews. Try scrolling further down manually.")

    finally:
        print("Closing browser in 5 seconds...")
        time.sleep(5)
        driver.quit()

if __name__ == "__main__":
    # A popular Bluetooth speaker on Meesho (High review count)
    target_url = "https://www.meesho.com/boAt-Stone-200-5W-Portable-Bluetooth-Speaker/p/1v2k3z" 
    scrape_meesho_reviews(target_url)