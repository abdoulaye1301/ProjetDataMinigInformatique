import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models.apriori_model import run_apriori, get_recommendations
from models.kmeans_model import run_kmeans, predict_cluster
from models.rfm_model import run_rfm
from utils.preprocessing import preprocess_data

st.set_page_config(page_title="Plateforme Data Mining", layout="wide")
st.title("🛍️ Analyse Data Mining - E-commerce")

# Choix de l’onglet
menu = st.sidebar.radio(
    "Navigation", ["Exploration des données", "Modélisation", "Prédiction"]
)

uploaded_file = st.sidebar.file_uploader(
    "📁 Charger un fichier CSV ou Excel", type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(uploaded_file)

    df = preprocess_data(df)

    if menu == "Exploration des données":
        st.header("📊 Analyse exploratoire")
        st.write("Aperçu des données :")
        st.dataframe(df.head())
        df2 = df.copy()
        ## Recuperation des variables quantitatives
        varQuant = ["UnitPrice", "Quantity"]
        ## Recuperation des variables qualitatives
        varQual = ["Country", "Description", "StockCode"]

        choix = st.sidebar.selectbox("Type statistique", ("Description", "Graphique"))
        # Statistique descriptive
        if choix == "Description":
            descr = st.sidebar.selectbox(
                "Description des variables",
                ("Variables quanitatives", "Variables qualitatives"),
            )
            if descr == "Variables quanitatives":
                st.text("Description des variables quantitatives")
                st.write(df2[varQuant].describe())
            elif descr == "Variables qualitatives":
                st.text("Description des variables qualitatives")
                st.write(df2[varQual].describe())

        # Les graphiques
        elif choix == "Graphique":
            statist = st.sidebar.selectbox(
                "Graphiques", ("Diagramme en barre", "Histogramme", "Boxplot")
            )

            # Visualisation des variables quantitatives
            if statist == "Histogramme":
                st.text("Représantation graphique des variables quantitatives")
                var = st.sidebar.selectbox("Choisire la variable", varQuant)
                fig, ax = plt.subplots()
                sns.histplot(df[var], bins=30, ax=ax)
                plt.title(f"Histogramme de {var}")
                plt.xlabel(var)
                st.pyplot(fig)

                # Visualisation des variables qualitatives
            elif statist == "Diagramme en barre":
                st.text("Représantation graphique des variables qualitatives")
                var = st.sidebar.selectbox("Choisire la variable", varQual)
                if var == "Country":
                    # Nombre de commandes par pays
                    st.subheader("🌍 Transactions par pays")
                    country_counts = df["Country"].value_counts().head(10)
                    st.bar_chart(country_counts)
                else:
                    don, ax = plt.subplots()
                    ax = sns.countplot(data=df2, x=var, palette="Set2")  # color=colbar)
                    plt.title(f"Diagramme en barre de {var}")
                    plt.ylabel("Frequence")
                    plt.xlabel(var)
                    st.pyplot(don)

                # Vsualisation des boxplot après nétoyage
            elif statist == "Boxplot":
                st.text("Observation des outliers")
                # Nettoyage des outliers
                for var in varQuant:
                    Q1 = df2[var].quantile(0.25)
                    Q3 = df2[var].quantile(0.75)
                    IQR = Q3 - Q1
                    min = Q1 - 1.5 * IQR
                    max = Q3 + 1.5 * IQR
                    df2.loc[df2[var] < min, var] = min
                    df2.loc[df2[var] > max, var] = max

                var = st.sidebar.selectbox("Choisire la variable", varQuant)

                don, ax = plt.subplots()
                ax = sns.boxplot(data=df2, x=var, color="blue")
                plt.title(f"Boxplot de {var}")
                plt.xlabel(var)
                st.pyplot(don)

    elif menu == "Modélisation":
        st.header("📊 Analyse des modèles")
        model_choice = st.sidebar.selectbox(
            "Sélectionnez un modèle :", ["Apriori", "K-means", "RFM"]
        )

        if model_choice == "Apriori":
            st.subheader("Règles d'association - Apriori")
            rules = run_apriori(df)
            st.dataframe(rules.head(5))

            st.subheader("🔁 Recommandation basée sur un produit")
            all_products = df["Description"].dropna().unique()
            product_selected = st.sidebar.selectbox(
                "Sélectionnez un produit :", sorted(all_products)
            )
            recommended = get_recommendations(product_selected, rules)
            st.write(f"Produits recommandés pour : **{product_selected}**")
            if recommended:
                st.success(recommended)
            else:
                st.warning("Aucune recommandation trouvée.")

        elif model_choice == "K-means":
            st.subheader("Segmentation client - K-means")
            result_df, fig = run_kmeans(df)
            st.plotly_chart(fig)
            st.dataframe(result_df)

        elif model_choice == "RFM":
            st.subheader("Segmentation RFM")
            rfm_df = run_rfm(df)
            st.dataframe(rfm_df)

    elif menu == "Prédiction":
        choix_predi = st.sidebar.selectbox(
            "Sélectionnez un modèle de prédiction :", ["K-means", "RFM"]
        )
        if choix_predi == "RFM":
            # Préduction du segment
            st.header("🔮 Prédiction de segment client (RFM)")
            rfm_result = run_rfm(df)
            # st.subheader("Segmentation RFM calculée")
            # st.dataframe(rfm_result.head())

            client_id = st.sidebar.selectbox(
                "Choisissez un client :", rfm_result["CustomerID"].unique()
            )
            st.write(rfm_result[rfm_result["CustomerID"] == client_id])
        elif choix_predi == "K-means":
            # Prédiction de la classe avec K-Means
            st.subheader("📍 Prédire le segment d’un nouveau client (K-means)")
            _, _, kmeans_model = run_kmeans(df)
            qty = st.number_input("Quantité totale achetée", min_value=0)
            amt = st.number_input("Montant total dépensé", min_value=0.0)
            if st.button("Prédire le cluster"):
                cluster = predict_cluster(kmeans_model, qty, amt)
                st.success(f"Ce client appartiendrait au cluster : {cluster}")
else:
    st.info("Veuillez charger un fichier pour commencer.")
