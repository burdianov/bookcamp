import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

df = pd.read_csv("../data/telco-customer-churn.csv")

df.head(5)
df.head().T

df.dtypes

total_charges = pd.to_numeric(df.TotalCharges, errors="coerce")

df[total_charges.isnull()][["customerID", "TotalCharges"]]

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")
df.TotalCharges = df.TotalCharges.fillna(0)

df.columns = df.columns.str.lower().str.replace(" ", "_")
string_columns = list(df.dtypes[df.dtypes == "object"].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

df.churn = (df.churn == "yes").astype(int)

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

y_train = df_train.churn.values
y_val = df_val.churn.values

del df_train["churn"]
del df_val["churn"]

df_train_full.isnull().sum()
df_train_full.churn.value_counts()

global_mean = df_train_full.churn.mean()
round(global_mean, 3)

categorical = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]

numerical = [
    "tenure",
    "monthlycharges",
    "totalcharges",
]
df_train_full[categorical].nunique()

female_mean = df_train_full[df_train_full.gender == "female"].churn.mean()
male_mean = df_train_full[df_train_full.gender == "male"].churn.mean()

partner_yes = df_train_full[df_train_full.partner == "yes"].churn.mean()
partner_no = df_train_full[df_train_full.partner == "no"].churn.mean()

global_mean = df_train_full.churn.mean()

df_group = df_train_full.groupby(by="gender").churn.agg(["mean"])
df_group["diff"] = df_group["mean"] - global_mean
df_group["risk"] = df_group["mean"] / global_mean

for col in categorical:
    df_group = df_train_full.groupby(by=col).churn.agg(["mean"])
    df_group["diff"] = df_group["mean"] - global_mean
    df_group["rate"] = df_group["mean"] / global_mean
    display(df_group)


def calculate_mi(series):
    return mutual_info_score(series, df_train_full.churn)


df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name="MI")
