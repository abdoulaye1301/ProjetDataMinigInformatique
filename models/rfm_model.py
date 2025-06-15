import pandas as pd


def run_rfm(df):

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalAmount": lambda x: x.sum(),
        }
    )
    rfm.columns = ["Recence", "Frequence", "Montant"]
    quartiles = (
        rfm[["Recence", "Frequence", "Montant"]].quantile([0.25, 0.5, 0.75]).to_dict()
    )

    # Fonction pour definir les côtes de la recences
    def r_score(x):
        if x <= quartiles["Recence"][0.25]:
            return 4
        elif quartiles["Recence"][0.25] < x <= quartiles["Recence"][0.5]:
            return 3
        elif quartiles["Recence"][0.5] < x <= quartiles["Recence"][0.75]:
            return 2
        else:
            return 1

    # Fonction pour definir les côtes du montant, et frequence
    def fm_score(x, col):
        if x <= quartiles[col][0.25]:
            return 1
        elif quartiles[col][0.25] < x <= quartiles[col][0.5]:
            return 2
        elif quartiles[col][0.5] < x <= quartiles[col][0.75]:
            return 3
        else:
            return 4

    rfm["R"] = rfm["Recence"].apply(lambda x: r_score(x))
    rfm["F"] = rfm["Frequence"].apply(lambda x: fm_score(x, "Frequence"))
    rfm["M"] = rfm["Montant"].apply(lambda x: fm_score(x, "Montant"))
    # Concatenation des scores RFM
    rfm["RFM_Score"] = rfm["R"].map(str) + rfm["F"].map(str) + rfm["M"].map(str)
    code_segt = {
        # Définition de la carte de segmentation(voir le dessin ci-dessus) à travers un dictionnaire
        r"11": "Clients en hibernation",
        r"1[2-3]": "Clients à risque",
        r"14": "Clients à ne pas perdre",
        r"21": "Clients presqu'endormis",
        r"22": "Clients à suivre",
        r"[2-3][3-4]": "Clients loyaux",
        r"31": "Clients prometteurs",
        r"41": "Nouveaux clients",
        r"[3-4]2": "Clients potentiellement loyaux",
        r"4[3-4]": "Très bons clients",
    }

    # Ajout de la colonne "Segment" au dataframe rfm
    rfm["Segment"] = rfm["R"].map(str) + rfm["F"].map(str)
    rfm["Segment"] = rfm["Segment"].replace(code_segt, regex=True)
    return rfm.reset_index()
