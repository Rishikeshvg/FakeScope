import os
import pandas as pd
from bs4 import BeautifulSoup

# CHANGE THIS to your actual HTML filename
html_path = os.path.join("data", "raw", "amazon_reviews.html")

output_path = os.path.join("data", "processed", "extracted_reviews.csv")

try:
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "lxml")

    reviews = []

    # Amazon review text selector
    review_blocks = soup.find_all("span", {"data-hook": "review-body"})

    for block in review_blocks:
        text = block.get_text(strip=True)
        if text:
            reviews.append(text)

    print("Total reviews extracted:", len(reviews))

    df = pd.DataFrame({"review_text": reviews})
    df.to_csv(output_path, index=False)

    print("Saved to:", output_path)

except Exception as e:
    print("Error:", e)
