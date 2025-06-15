from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px

# Entraînement + visualisation
def run_kmeans(df):
    features = df.groupby("CustomerID")[["Quantity", "TotalAmount"]].sum()
    kmeans = KMeans(n_clusters=3, random_state=42)
    features['Cluster'] = kmeans.fit_predict(features)
    fig = px.scatter(features, x="Quantity", y="TotalAmount", color=features['Cluster'].astype(str))
    return features.reset_index(), fig

# Prédiction sur de nouveaux clients
def predict_cluster(kmeans_model, quantity, amount):
    X_new = pd.DataFrame([[quantity, amount]], columns=["Quantity", "TotalAmount"])
    cluster = kmeans_model.predict(X_new)[0]
    return int(cluster)