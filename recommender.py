import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json

def load_data():
    """
    Load and preprocess the Instacart dataset.
    This function loads the necessary CSV files from the Instacart dataset,
    merges them, and prepares a transaction list suitable for market basket analysis.
    """
    try:
        print("Loading data...")
        # Load the datasets
        order_products_prior = pd.read_csv('./instacart-market-basket-analysis/order_products__prior.csv')
        products = pd.read_csv('./instacart-market-basket-analysis/products.csv')
        orders = pd.read_csv('./instacart-market-basket-analysis/orders.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: Make sure the Kaggle dataset is downloaded and extracted in a folder named 'instacart-market-basket-analysis' in the same directory as the script.")
        return None, None

    print("Preprocessing data...")
    # Reduce memory usage by downcasting types where possible
    order_products_prior['order_id'] = order_products_prior['order_id'].astype('uint32')
    order_products_prior['product_id'] = order_products_prior['product_id'].astype('uint16')
    order_products_prior['add_to_cart_order'] = order_products_prior['add_to_cart_order'].astype('uint8')
    order_products_prior['reordered'] = order_products_prior['reordered'].astype('uint8')
    orders['order_id'] = orders['order_id'].astype('uint32')
    orders['user_id'] = orders['user_id'].astype('uint32')
    products['product_id'] = products['product_id'].astype('uint16')
    
    # Merge the datasets to get product names
    order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')
    
    # Create a list of transactions (each transaction is a list of products in an order)
    transactions = order_products_prior.groupby('order_id')['product_name'].apply(list).tolist()
    print("Data preprocessing complete.")

    return transactions, products['product_name'].dropna().unique().tolist()

def get_recommendations(transactions):
    """
    Generate association rules using the Apriori algorithm with a memory-efficient sparse matrix.
    """
    print("Encoding transactions for Apriori using a sparse matrix...")
    # One-hot encode the data using a sparse format to save memory
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    
    # Convert the sparse matrix into a sparse DataFrame
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    print("Transaction encoding complete.")

    print("Running Apriori algorithm...")
    # Run Apriori on the sparse DataFrame
    frequent_itemsets = apriori(df, min_support=0.001, use_colnames=True)
    print(f"Apriori complete. Found {len(frequent_itemsets)} frequent itemsets.")

    print("Generating association rules...")
    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    print("Association rule generation complete.")

    # Convert antecedents and consequents to lists for JSON serialization
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))

    return rules

def main():
    """
    Main function to run the recommendation engine.
    """
    transactions, product_list = load_data()
    if transactions:
        rules = get_recommendations(transactions)

        # Save the rules and product list to JSON files for the web app
        print("Saving rules and product list to JSON files...")
        rules.to_json('recommendation_rules.json', orient='records')
        with open('product_list.json', 'w') as f:
            json.dump(product_list, f)
        print("Files saved successfully. You can now run the web interface.")

if __name__ == '__main__':
    main()

