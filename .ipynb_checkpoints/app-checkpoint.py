import streamlit as st
import pandas as pd
from models.apriori_model import run_apriori
from models.fpgrowth_model import run_fpgrowth
from models.kmeans_model import run_kmeans
from models.rfm_model import run_rfm
from utils.preprocessing import preprocess_data

st.set_page_config(page_title="Data Mining E-commerce", layout="wide")
st.title("🛍️ Plateforme d'analyse Data Mining pour E-commerce")

st.sidebar.header("📁 Charger les données")
uploaded_file = st.sidebar.file_uploader(
    "Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    st.sidebar.header("📊 Choix du modèle")
    model_choice = st.sidebar.selectbox(
        "Sélectionnez le modèle :", ["Apriori", "FP-Growth", "K-means", "RFM"]
    )

    df_cleaned = preprocess_data(df)

    if model_choice == "Apriori":
        st.subheader("Règles d'association - APRIORI")
        rules = run_apriori(df_cleaned)
        st.dataframe(rules)

    elif model_choice == "FP-Growth":
        st.subheader("Règles d'association - FP-GROWTH")
        rules = run_fpgrowth(df_cleaned)
        st.dataframe(rules)

    elif model_choice == "K-means":
        st.subheader("Segmentation client - K-means")
        result_df, fig = run_kmeans(df_cleaned)
        st.plotly_chart(fig)
        st.dataframe(result_df)

    elif model_choice == "RFM":
        st.subheader("Segmentation RFM")
        rfm_df = run_rfm(df_cleaned)
        st.dataframe(rfm_df)

else:
    st.info("Veuillez charger un fichier de données pour démarrer l'analyse.")
