from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(df):
    basket = (
        df.groupby(["CustomerID", "Description"])["Description"]
        .count()
        .unstack()
        .fillna(0)
    )
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(basket, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    return rules[["antecedents", "consequents", "lift"]]


def get_recommendations(product_name, rules, top_n=5):
    product_rules = rules[rules["antecedents"].apply(lambda x: product_name in list(x))]
    product_rules = product_rules.sort_values(by="lift", ascending=False)
    recommendations = []
    for _, row in product_rules.iterrows():
        for item in row["consequents"]:
            if item != product_name:
                recommendations.append(item)
    return list(dict.fromkeys(recommendations))[:top_n]
