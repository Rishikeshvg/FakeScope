from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_amazon_reviews(url, max_pages=3):
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    reviews = []

    try:
        # Build review URL keeping same domain as input
        if '/dp/' in url:
            asin = url.split('/dp/')[1].split('/')[0].split('?')[0]
            if 'amazon.in' in url:
                review_url = f"https://www.amazon.in/product-reviews/{asin}/?reviewerType=all_reviews&sortBy=recent"
            else:
                review_url = f"https://www.amazon.com/product-reviews/{asin}/?reviewerType=all_reviews&sortBy=recent"
        else:
            review_url = url

        print(f"Opening: {review_url}")
        driver.get(review_url)
        time.sleep(5)

        for page in range(1, max_pages + 1):
            print(f"Scraping page {page}...")

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-hook="review-body"]'))
                )
            except:
                print("Reviews not found on this page.")
                break

            elements = driver.find_elements(By.CSS_SELECTOR, '[data-hook="review-body"]')
            for el in elements:
                text = el.text.replace("Read more", "").strip()
                if text and len(text) > 5:
                    reviews.append(text)

            print(f"Found {len(elements)} reviews on page {page}")

            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, 'li.a-last a')
                next_btn.click()
                time.sleep(3)
            except:
                print("No more pages.")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()

    return reviews


if __name__ == "__main__":
    url = input("Paste Amazon product URL: ")
    reviews = get_amazon_reviews(url)
    print(f"\nTotal reviews scraped: {len(reviews)}")
    for i, r in enumerate(reviews[:3]):
        print(f"\nReview {i+1}: {r[:100]}")