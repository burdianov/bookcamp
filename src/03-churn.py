import math

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mutual_info_score,
    accuracy_score,
    roc_curve,
    auc,
    roc_auc_score,
)
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

t = 0.5
predict_churn = y_pred >= t
predict_no_churn = y_pred < t

actual_churn = y_val == 1
actual_no_churn = y_val == 0

true_positive = (predict_churn & actual_churn).sum()
false_positive = (predict_churn & actual_no_churn).sum()

false_negative = (predict_no_churn & actual_churn).sum()
true_negative = (predict_no_churn & actual_no_churn).sum()

confusion_table = np.array(
    [[true_negative, false_positive], [false_negative, true_positive]]
)

confusion_table_fractions = confusion_table / confusion_table.sum()

accuracy = (true_negative + true_positive) / (
    true_negative + true_positive + false_negative + false_positive
)

precision = true_positive / (true_positive + false_positive)

recall = true_positive / (true_positive + false_negative)

false_positive_rate = false_positive / (false_positive + true_negative)
# the lower, the better

true_positive_rate = true_positive / (true_positive + false_negative)
# true_positive_rate == recall; the higher, the better

scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    tp = ((y_pred >= t) & (y_val == 1)).sum()
    fp = ((y_pred >= t) & (y_val == 0)).sum()
    fn = ((y_pred < t) & (y_val == 1)).sum()
    tn = ((y_pred < t) & (y_val == 0)).sum()
    scores.append((t, tp, fp, fn, tn))

df_scores = pd.DataFrame(scores)
df_scores.columns = ["threshold", "tp", "fp", "fn", "tn"]
df_scores[::10]

df_scores["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)
df_scores[::10]

plt.plot(df_scores.threshold, df_scores.tpr, label="TPR")
plt.plot(df_scores.threshold, df_scores.fpr, label="FPR")
plt.legend()

np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []
    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        tp = ((y_pred >= t) & (y_val == 1)).sum()
        fp = ((y_pred >= t) & (y_val == 0)).sum()
        fn = ((y_pred < t) & (y_val == 1)).sum()
        tn = ((y_pred < t) & (y_val == 0)).sum()
        scores.append((t, tp, fp, fn, tn))

    df_scores = pd.DataFrame(scores)
    df_scores.columns = ["threshold", "tp", "fp", "fn", "tn"]
    df_scores["tpr"] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores["fpr"] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores


df_rand = tpr_fpr_dataframe(y_val, y_rand)

plt.plot(df_rand.threshold, df_rand.tpr, label="TPR")
plt.plot(df_rand.threshold, df_rand.fpr, label="FPR")
plt.legend()

num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_pred_ideal = np.linspace(0, 1, num_neg + num_pos)
df_ideal = tpr_fpr_dataframe(y_ideal, y_pred_ideal)

plt.plot(df_ideal.threshold, df_ideal.tpr, label="TPR")
plt.plot(df_ideal.threshold, df_ideal.fpr, label="FPR")
plt.legend()

plt.figure(figsize=(5, 5))
plt.plot(df_scores.fpr, df_scores.tpr, label="Model")
plt.plot(df_rand.fpr, df_rand.tpr, label="Random")
plt.plot(df_ideal.fpr, df_ideal.tpr, label="Ideal")
plt.legend()

plt.figure(figsize=(5, 5))
plt.plot(df_scores.fpr, df_scores.tpr)
plt.plot([0, 1], [0, 1])

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])

auc(df_scores.fpr, df_scores.tpr)
roc_auc_score(y_val, y_pred)


def train(df, y):
    cat = df[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)
    X = dv.transform(cat)
    model = LogisticRegression(solver="liblinear")
    model.fit(X, y)
    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient="records")
    X = dv.transform(cat)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


kfold = KFold(n_splits=10, shuffle=True, random_state=1)
aucs = []
for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    df_val = df_train_full.iloc[val_idx]
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)
    auc = roc_auc_score(y_val, y_pred)
    aucs.append(auc)


def train(df, y, C):
    cat = df[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)
    X = dv.transform(cat)
    model = LogisticRegression(solver="liblinear", C=C)
    model.fit(X, y)
    return dv, model


nfolds = 5
kfold = KFold(n_splits=nfolds, shuffle=True, random_state=1)
for C in [0.001, 0.01, 0.1, 0.5, 1, 10]:
    aucs = []
    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]
        y_train = df_train.churn.values
        y_val = df_val.churn.values
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
    print("C=%s, auc = %0.3f ± %0.3f" % (C, np.mean(aucs), np.std(aucs)))

y_train = df_train_full.churn.values
y_test = df_test.churn.values
dv, model = train(df_train_full, y_train, C=0.5)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
print("auc = %.3f" % auc)
