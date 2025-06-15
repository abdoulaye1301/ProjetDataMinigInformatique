import pandas as pd


def preprocess_data(df):
    df = df.dropna(subset=["CustomerID", "Quantity", "UnitPrice", "InvoiceDate"])
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df
