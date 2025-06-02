import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client, Client # Ensure Client is imported
import joblib
from dotenv import load_dotenv
import re
import time # For potential retries or delays if needed

def preprocess_text_backend(text):
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if len(word) > 2 and not word.isdigit()])
    return text

def train_and_save_model():
    load_dotenv()
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Supabase URL or Key not found in .env file.")
        return

    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("Fetching all products for TF-IDF model training (with pagination)...")
    all_products_data = []
    offset = 0
    page_size = 1000 # Fetch in chunks of 1000 (Supabase's typical page limit)
    max_retries = 3 # Optional: for handling transient network issues

    while True:
        current_retry = 0
        while current_retry < max_retries:
            try:
                print(f"Fetching products from offset {offset}, limit {page_size} (Attempt {current_retry + 1})...")
                response = (
                    supabase_client.table("Products")
                    .select("Product_Id, Product_Name, Product_Description")
                    .range(offset, offset + page_size - 1) # Use .range() for pagination
                    .execute()
                )

                if hasattr(response, 'data') and response.data is not None: # Check if data is not None
                    print(f"Fetched {len(response.data)} products in this batch.")
                    all_products_data.extend(response.data)
                    if len(response.data) < page_size:
                        # Fetched less than page_size, so it's the last page or an empty page
                        print("Last page reached or empty page.")
                        break # Break from retry loop, then outer while loop will break
                    offset += page_size
                    break # Success for this page, break from retry loop
                elif hasattr(response, 'error') and response.error:
                    print(f"Supabase API Error on page (offset {offset}): {response.error.message}")
                    # Depending on the error, you might want to retry or stop
                    current_retry += 1
                    if current_retry >= max_retries:
                        print("Max retries reached for this page. Stopping.")
                        # Decide if you want to proceed with partially fetched data or exit
                        if not all_products_data: return # Exit if no data fetched at all
                        break # Break from retry, outer loop will assess
                    print(f"Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    print(f"No data and no error on page (offset {offset}). Assuming end of data.")
                    break # Break from retry, outer loop will assess

            except Exception as e:
                print(f"General Error during fetching products at offset {offset}: {e}")
                import traceback
                traceback.print_exc()
                current_retry += 1
                if current_retry >= max_retries:
                    print("Max retries reached due to general error. Stopping.")
                    if not all_products_data: return
                    break
                print(f"Retrying in 3 seconds...")
                time.sleep(3)
        
        # Check if we should break the outer while loop
        if (hasattr(response, 'data') and response.data is not None and len(response.data) < page_size) or \
           not (hasattr(response, 'data') and response.data is not None) or \
           current_retry >= max_retries:
            break


    if not all_products_data:
        print("No products were successfully fetched after all attempts.")
        return

    products_data = all_products_data
    print(f"--- Fetched a total of {len(products_data)} products using pagination. ---")

    df = pd.DataFrame(products_data)
    # Ensure Product_Id is integer if it's not already (can happen if some are None from DB)
    df['Product_Id'] = pd.to_numeric(df['Product_Id'], errors='coerce').fillna(0).astype(int)

    # Filter out rows where Product_Id became 0 due to conversion error (if original was bad)
    df = df[df['Product_Id'] != 0]


    df['corpus_text'] = (df['Product_Name'].fillna('') + " " + df['Product_Description'].fillna('')).apply(preprocess_text_backend)

    print(f"Training TF-IDF on {len(df)} products...")
    if df.empty:
        print("DataFrame is empty after processing. Cannot train TF-IDF model.")
        return
        
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.90,
        min_df=3, # If you have few products now, you might need to lower this to 1 or 2
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(df['corpus_text'])

    # Ensure no NaN/None product IDs before saving
    product_ids_for_matrix = df['Product_Id'].dropna().astype(int).tolist()
    if len(product_ids_for_matrix) != tfidf_matrix.shape[0]:
        print(f"Warning: Mismatch between product IDs ({len(product_ids_for_matrix)}) and TF-IDF matrix rows ({tfidf_matrix.shape[0]}).")
        # This can happen if some products had no corpus_text after preprocessing.
        # You might need a more robust way to align them or filter df before fit_transform.
        # For now, this might lead to issues in main.py when looking up by ID.
        # A safer approach: Re-index product_ids_for_matrix based on the df that was used for fit_transform.
        # However, if df['corpus_text'] had empty strings, fit_transform might handle it differently.
        # Let's assume for now df used for fit_transform is the one whose IDs we need.
        if df.shape[0] == tfidf_matrix.shape[0]:
             product_ids_for_matrix = df['Product_Id'].tolist() # Re-align based on the df used
        else:
            print("CRITICAL WARNING: Cannot safely align product IDs with TF-IDF matrix. Model saving might be incorrect.")


    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')
    joblib.dump(product_ids_for_matrix, 'product_ids_for_matrix.joblib')

    print("TF-IDF model, matrix, and product IDs saved successfully!")
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"Number of Product IDs saved: {len(product_ids_for_matrix)}")


if __name__ == "__main__":
    train_and_save_model()