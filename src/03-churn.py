import math

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

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

df_train_full[numerical].corrwith(df_train_full.churn)

train_dict = df_train[categorical + numerical].to_dict(orient="records")

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

X_train[0]

dv.get_feature_names_out()

model = LogisticRegression(solver="liblinear", random_state=1)
model.fit(X_train, y_train)

val_dict = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)
y_pred = model.predict_proba(X_val)[:, 1]
churn = y_pred >= 0.5

(y_val == churn).mean()

dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))

## Using the model
customer = {
    "customerid": "8879-zkjof",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85,
    "totalcharges": 3320.75,
}

X_test = dv.transform([customer])
model.predict_proba(X_test)
model.predict_proba(X_test)[0, 1]

customer = {
    "gender": "female",
    "seniorcitizen": 1,
    "partner": "no",
    "dependents": "no",
    "phoneservice": "yes",
    "multiplelines": "yes",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 85.7,
    "totalcharges": 85.7,
}
X_test = dv.transform([customer])
model.predict_proba(X_test)

# compute the accuracy
y_pred = model.predict_proba(X_val)[:, 1]
churn = y_pred >= 0.5
(churn == y_val).mean()  # accuracy

thresholds = np.linspace(0, 1, 11)
for t in thresholds:
    churn = y_pred >= t
    acc = accuracy_score(y_val, churn)
    print("%0.2f %0.3f" % (t, acc))

thresholds = np.linspace(0, 1, 21)
accuracies = []
for t in thresholds:
    acc = accuracy_score(y_val, y_pred >= t)
    accuracies.append(acc)
    print("%0.2f %0.3f" % (t, acc))

plt.plot(thresholds, accuracies)

size_val = len(y_val)
baseline = np.repeat(False, size_val)

accuracy_score(baseline, y_val)
