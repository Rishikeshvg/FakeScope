from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os

# 1. Setup Chrome Options (Stealth Mode)
chrome_options = Options()
# chrome_options.add_argument("--headless") # Uncomment this to hide the browser window
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# 2. Initialize Driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

reviews = []
# Updated URL to the general iPhone 15 review page
target_url = "https://www.flipkart.com/apple-iphone-15-black-128-gb/product-reviews/itm6ac6485515ae4?pid=MOBGTAGPTJBZBCYZ"

try:
    print("Opening Browser...")
    driver.get(target_url)
    time.sleep(5) # Give it time to load

    # Scrape 2 pages
    for page in range(1, 3):
        print(f"Scraping Page {page}...")
        
        # Find all review text elements
        # 'ZmyHeS' is the current class for review text as of early 2026
        elements = driver.find_elements(By.CLASS_NAME, "ZmyHeS")
        
        for el in elements:
            clean_text = el.text.replace("READ MORE", "").strip()
            if clean_text:
                reviews.append(clean_text)
        
        # Try to find the "Next" button and click it
        try:
            next_button = driver.find_element(By.XPATH, "//span[text()='Next']")
            next_button.click()
            time.sleep(4)
        except:
            print("No more pages or blocked.")
            break

    # 3. Save Data
    if reviews:
        df = pd.DataFrame(reviews, columns=['text'])
        output_path = os.path.join('data', 'raw', 'scraped_reviews.csv')
        df.to_csv(output_path, index=False)
        print(f"✅ Success! Saved {len(df)} reviews to {output_path}")
    else:
        print("❌ No reviews found. Flipkart's layout might have changed.")

finally:
    driver.quit()