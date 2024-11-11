import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = "https://clevershop.mk/product-category/mobilni-laptopi-i-tableti/"
product_data = []

def scrape_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        product_containers = soup.select('.product')

        # Loop through each product container on the page
        for product in product_containers:
            title_tag = product.select_one('.wd-entities-title')
            title = title_tag.get_text(strip=True) if title_tag else None

            # Product URL
            product_url_tag = product.select_one('a[href]')
            product_url = product_url_tag["href"] if product_url_tag else None

            # Price Information
            price_tags = product.select('.woocommerce-Price-amount')
            regular_price = price_tags[0].get_text(strip=True) if price_tags else None
            discount_price = price_tags[1].get_text(strip=True) if len(price_tags) > 1 else None

            # Add to Cart URL
            add_to_cart_tag = product.select_one("a.add_to_cart_button")
            add_to_cart_url = add_to_cart_tag["href"] if add_to_cart_tag else None

            # Append product info to list
            product_data.append({
                "Title": title,
                "Regular Price": regular_price,
                "Discount Price": discount_price,
                "Product URL": product_url,
                "Add to Cart URL": add_to_cart_url
            })


# Iterate through pages
for page_num in range(1, 6):  # Adjust range based on total pages available
    page_url = f"{base_url}page/{page_num}/"
    print(f"Scraping {page_url}")
    scrape_page(page_url)

# Create a DataFrame
df = pd.DataFrame(product_data)

# Save the DataFrame to a CSV file
df.to_csv("products.csv", index=False)
print("Data saved to products.csv")
