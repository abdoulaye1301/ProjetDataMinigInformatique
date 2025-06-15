from mlxtend.frequent_patterns import fpgrowth, association_rules

def run_fpgrowth(df):
    basket = df.groupby(['CustomerID', 'Description'])['Description'].count().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = fpgrowth(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
