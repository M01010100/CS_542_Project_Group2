
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd 
from dataHelper import load_data, sort_member_numbers, print_member_items

RULES = None  

def apply_apriori(items_dict, min_support=0.01):
    transactions = list(items_dict.values())
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    return frequent_itemsets, rules

def organize_results(frequent_itemsets, rules, top_n=10):
    #Parameters:
    #- frequent_itemsets: DataFrame of frequent itemsets
    #  rules: DataFrame of association rules
    #- top_n: Number of top results to return
    #Returns:
    #- Dictionary containing organized results
    organized = {
        'top_frequent_items': frequent_itemsets.nlargest(top_n, 'support')[['itemsets', 'support']],
        'top_confidence_rules': rules.nlargest(top_n, 'confidence')[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
        'top_lift_rules': rules.nlargest(top_n, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
        'summary_stats': {
            'total_itemsets': len(frequent_itemsets),
            'total_rules': len(rules),
            'avg_confidence': rules['confidence'].mean() if len(rules) > 0 else 0,
            'avg_lift': rules['lift'].mean() if len(rules) > 0 else 0
        }
    }
    return organized

def analyze_results(organized_results):
    print("\n" + "="*80)
    print("MARKET BASKET ANALYSIS RESULTS")
    print("="*80)
    
    # Summary Statistics
    print("\n SUMMARY STATISTICS:")
    print("-" * 80)
    stats = organized_results['summary_stats']
    print(f"Total Frequent Itemsets Found: {stats['total_itemsets']}")
    print(f"Total Association Rules Found: {stats['total_rules']}")
    print(f"Average Confidence: {stats['avg_confidence']:.3f}")
    print(f"Average Lift: {stats['avg_lift']:.3f}")
    
    # Top Frequent Items
    print("\n TOP FREQUENT ITEMSETS:")
    print("-" * 80)
    print(organized_results['top_frequent_items'].to_string(index=False))
    
    # Top Confidence Rules
    print("\n TOP RULES BY CONFIDENCE:")
    print("-" * 80)
    print("(If customer buys X, they will likely buy Y)")
    for idx, row in organized_results['top_confidence_rules'].iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        print(f"\n{antecedents} → {consequents}")
        print(f"  Confidence: {row['confidence']:.2%} | Support: {row['support']:.3f} | Lift: {row['lift']:.2f}")
    
    # Top Lift Rules
    print("\n TOP RULES BY LIFT:")
    print("-" * 80)
    print("(Strongest product associations)")
    for idx, row in organized_results['top_lift_rules'].iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        print(f"\n{antecedents} → {consequents}")
        print(f"  Lift: {row['lift']:.2f} | Confidence: {row['confidence']:.2%} | Support: {row['support']:.3f}")
    
    print("\n" + "="*80)
    print("\nINTERPRETATION GUIDE:")
    print("-" * 80)
    print("• Support: How frequently the itemset appears in transactions")
    print("• Confidence: Probability of buying Y given that X is bought")
    print("• Lift > 1: Items are likely to be bought together")
    print("• Lift = 1: No association between items")
    print("• Lift < 1: Items are unlikely to be bought together")
    print("="*80 + "\n")

def predict_for_customer(customer_items, rules, top_n=5):
    #Parameters:
    #- customer_items: List of items the customer already has
    #- rules: DataFrame of association rules
    #- top_n: Number of recommendations to return
    
    #Returns:
    #- List of recommended items with confidence scores
    recommendations = []
    customer_set = set(customer_items)
    
    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        
        # If customer has all antecedent items and doesn't have consequent
        if antecedents.issubset(customer_set) and not consequents.intersection(customer_set):
            for item in consequents:
                recommendations.append({
                    'item': item,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'based_on': list(antecedents)
                })
    
    # Sort by confidence and return top N
    recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    return recommendations[:top_n]

def interactive_predict_for_customer():
    customer_items = input("Enter items the customer has (comma-separated): ")
    customer_items = [item.strip() for item in customer_items.split(',')]

    rules = RULES
    top_n  = int(input("Enter number of recommendations to return (default 5): ") or 5)

    recommendations = []
    customer_set = set(customer_items)
    
    for idx, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        
        # If customer has all antecedent items and doesn't have consequent
        if antecedents.issubset(customer_set) and not consequents.intersection(customer_set):
            for item in consequents:
                recommendations.append({
                    'item': item,
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'based_on': list(antecedents)
                })
    
    # Sort by confidence and return top N
    recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    return recommendations[:top_n]

def run_apriori():
    global RULES
    data = load_data('data/Groceries_dataset.csv')
    print(" Dataset Preview:")
    print(data.head())
    print(f"\nTotal transactions: {len(data)}")
    print(f"Unique members: {data['Member_number'].nunique()}")
    
    print("\n" + "="*80)
    print("Sample Member Shopping Baskets:")
    print_member_items(data, 10)
    
    # Apply Apriori algorithm
    print("\n Running Apriori algorithm...")
    items_dict = sort_member_numbers(data)
    frequent_itemsets, rules = apply_apriori(items_dict, min_support=0.01)
    RULES = rules  
    # Organize and analyze results
    organized = organize_results(frequent_itemsets, rules, top_n=10)
    analyze_results(organized)
    
    # Example: Predict for a sample customer
    print("\n" + "="*80)
    print(" CUSTOMER RECOMMENDATION EXAMPLE:")
    print("="*80)
    sample_customer_items = ['frankfurter' , 'sausage']
    print(f"Customer has: {sample_customer_items}")
    predictions = predict_for_customer(sample_customer_items, rules, top_n=5)
    
    if predictions:
        print("\nRecommended items:")
        for i, pred in enumerate(predictions, 1):
            print(f"\n{i}. {pred['item']}")
            print(f"   Confidence: {pred['confidence']:.2%}")
            print(f"   Lift: {pred['lift']:.2f}")
            print(f"   Based on: {', '.join(pred['based_on'])}")
    else:
        print("\nNo recommendations found for this basket.")
    print("="*80 + "\n")

