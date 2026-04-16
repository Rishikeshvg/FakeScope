from bs4 import BeautifulSoup
import pandas as pd
import os

html_file = os.path.join('data', 'raw', 'flipkart_reviews.html')

if not os.path.exists(html_file):
    print(f"❌ Error: {html_file} not found!")
else:
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    reviews = []
    
    # Flipkart reviews are almost always inside a div.
    # We look for divs that DON'T have many nested tags and have significant text.
    for div in soup.find_all('div'):
        # Get the text, remove the 'READ MORE' link text
        text = div.get_text(separator=" ").replace('READ MORE', '').strip()
        
        # LOGIC: If a div has 100-1000 characters and is not just a menu, it's a review.
        if 50 < len(text) < 1500:
            # Avoid duplicates and menu items
            if text not in reviews and "Sign In" not in text and "Cart" not in text:
                reviews.append(text)

    if reviews:
        # Create the dataframe
        df = pd.DataFrame(reviews, columns=['text'])
        
        # Clean up: Remove very short entries that might be headers
        df = df[df['text'].str.split().str.len() > 5]
        
        output_path = os.path.join('data', 'raw', 'scraped_reviews.csv')
        df.to_csv(output_path, index=False)
        
        print(f"✅ Success! Found {len(df)} reviews in the hairdryer file.")
        print("\n--- Sample Scraped Review ---")
        print(df['text'].iloc[0][:100] + "...")
    else:
        print("❌ Still no reviews found. The HTML might be empty or restricted.")