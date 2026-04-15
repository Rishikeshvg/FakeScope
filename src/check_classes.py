from bs4 import BeautifulSoup
import os

html_file = os.path.join('data', 'raw', 'flipkart_reviews.html')

with open(html_file, 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

# This prints the first 500 characters of all 'div' text to help us find the review
for div in soup.find_all('div'):
    text = div.get_text().strip()
    # If the text is long, it's likely a review
    if len(text) > 50:
        print(f"Class: {div.get('class')} | Text: {text[:50]}...")