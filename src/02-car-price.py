import pandas as pd  # for numerical operations
import numpy as np  # for tabular data
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/cardataset.csv")
df.head()

# Data analysis and preprocessing

## Lowercase all the column names and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(" ", "_")

## Select only the columns with string values
string_columns = list(df.dtypes[df.dtypes == "object"].index)

## Lowercase and replace spaces with underscores for values in all string columns of the DataFrame
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

df.head()

## Plot the target variables histogram (distribution of prices)
sns.histplot(df.msrp, bins=40)

## Zoom in
sns.histplot(df.msrp[df.msrp < 100_000])

## Apply the log on the prices Ynew = log(y + 1)
log_price = np.log1p(df.msrp)
sns.histplot(log_price)

## Check the missing values
df.isnull().sum()

# Validation framework

## Split data into training, validation and test sets

n = len(df)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

np.random.seed(2)
idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train : n_train + n_val].copy()
df_test = df_shuffled.iloc[n_train + n_val :].copy()

## Apply the log transformation
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

## Optional: remove the target variable from the dataframe to avoid using it accidentally later
del df_train["msrp"]
del df_val["msrp"]
del df_test["msrp"]

<<<<<<< HEAD
# Linear regression
# y ≈ g(X)
# y: vector of predicted values
# X: matrix of features and observations


def g(xi):
    pass


w0 = 7.17
# [w1 w2 w3 ]
w = [0.01, 0.04, 0.002]
n = 3


def linear_regression(xi):
    result = w0
    for j in range(n):
        result = result + xi[j] * w[j]
    return result


x = [1, 2, 3]
w = [4, 5, 6]

matrix = [[2, 2, 2], [3, 3, 3]]

p = np.dot(x, w)
com = np.dot(matrix, x)


def dot(xi, w):
    n = len(w)
    result = 0.0
    for i in range(n):
        result += xi[i] * w[i]
    return result


pdot = dot(x, w)


def linear_regression(xi):
    xi = [1] + xi
    return dot(xi, w)


w = [w0] + w
=======
base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]

df_num = df_train[base]

df_num.isna().count()

df_num.fillna(0)

X_train = df_num.values

a = np.array([1, 2, 3, 4])
b = np.array([5, 9, 14, 4])

mm = b == a

feature = "num_doors_%s" % 3
>>>>>>> cc3f880eca83fd879baffb002d5b0125f2d4d2ea
