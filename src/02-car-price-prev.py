import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading and preparing data

df = pd.read_csv("../data/cardataset.csv")
df.head()

df.columns = df.columns.str.lower().str.replace(" ", "_")
string_columns = list(df.dtypes[df.dtypes == "object"].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

df.head()

# Target variable analysis

sns.displot(df.msrp, kde=False)
sns.displot(df.msrp[df.msrp < 100_000], kde=False)

log_price = np.log1p(df.msrp)
sns.displot(log_price, kde=False)

# Checking the missing values

df.isnull().sum()

# Split data into training, validation and test set

n = len(df)

n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

np.random.seed(42)
idx = np.arange(n)
np.random.shuffle(idx)

df_shuffled = df.iloc[idx]

df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train : n_train + n_val].copy()
df_test = df_shuffled.iloc[n_train + n_val :].copy()

# Apply the log transformation

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

# Remove the target variable from the dataframe

del df_train["msrp"]
del df_val["msrp"]
del df_test["msrp"]


# Linear regression implemented with NumPy
def linear_regression(X, y):
    # adding the dummy column
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    # normal equation formula
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]
df_num = df_train[base]

df_num = df_num.fillna(0)

X_train = df_num.values

w_0, w = linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)

sns.distplot(y_pred, label="prediction")
sns.distplot(y_train, label="target")
plt.legend()


# RMSE (Root Mean Square Error) implementation
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error**2).mean()
    return np.sqrt(mse)


rmse(y_train, y_pred)

df_num = df_val[base]
df_num = df_num.fillna(0)
X_val = df_num.values

y_pred = w_0 + X_val.dot(w)

rmse(y_val, y_pred)


# convert a dataframe into a matrix
def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df["age"] = 2017 - df.year
    features.append("age")

    # apply one-hot encoding
    for v in [2, 3, 4]:
        feature = "num_doors_%s" % v
        value = (df["number_of_doors"] == v).astype(int)
        df[feature] = value
        features.append(feature)

    for v in ["chevrolet", "ford", "volkswagen", "toyota", "dodge"]:
        feature = "is_make_%s" % v
        df[feature] = (df["make"] == v).astype(int)
        features.append(feature)

    for v in [
        "regular_unleaded",
        "premium_unleaded_(required)",
        "premium_unleaded_(recommended)",
        "flex-fuel_(unleaded/e85)",
    ]:  # A
        feature = "is_type_%s" % v
        df[feature] = (df["engine_fuel_type"] == v).astype(int)
        features.append(feature)
    for v in ["automatic", "manual", "automated_manual"]:  # B
        feature = "is_transmission_%s" % v
        df[feature] = (df["transmission_type"] == v).astype(int)
        features.append(feature)
    for v in [
        "front_wheel_drive",
        "rear_wheel_drive",
        "all_wheel_drive",
        "four_wheel_drive",
    ]:  # C
        feature = "is_driven_wheels_%s" % v
        df[feature] = (df["driven_wheels"] == v).astype(int)
        features.append(feature)
    for v in [
        "crossover",
        "flex_fuel",
        "luxury",
        "luxury,performance",
        "hatchback",
    ]:  # D
        feature = "is_mc_%s" % v
        df[feature] = (df["market_category"] == v).astype(int)
        features.append(feature)
    for v in ["compact", "midsize", "large"]:  # E
        feature = "is_size_%s" % v
        df[feature] = (df["vehicle_size"] == v).astype(int)
        features.append(feature)
    for v in ["sedan", "4dr_suv", "coupe", "convertible", "4dr_hatchback"]:  # F
        feature = "is_style_%s" % v
        df[feature] = (df["vehicle_style"] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


df_train["age"] = 2017 - df_train.year

X_train = prepare_X(df_train)
w_0, w = linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print("validation:", rmse(y_val, y_pred))

sns.distplot(y_pred, label="prediction")
sns.distplot(y_val, label="target")
plt.legend()

X_train = prepare_X(df_train)
w_0, w = linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print("validation:", rmse(y_val, y_pred))


def linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


for r in [0, 0.001, 0.01, 0.1, 1, 10]:
    w_0, w = linear_regression_reg(X_train, y_train, r=r)
    print("%5s, %.2f, %.2f, %.2f" % (r, w_0, w[13], w[21]))

X_train = prepare_X(df_train)
w_0, w = linear_regression_reg(X_train, y_train, r=0.001)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print("validation:", rmse(y_val, y_pred))

X_train = prepare_X(df_train)
X_val = prepare_X(df_val)
for r in [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    print("%6s" % r, rmse(y_val, y_pred))

X_train = prepare_X(df_train)
w_0, w = linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print("validation:", rmse(y_val, y_pred))

X_test = prepare_X(df_test)
y_pred = w_0 + X_test.dot(w)
print("test:", rmse(y_test, y_pred))

# using the model

ad = {
    "city_mpg": 18,
    "driven_wheels": "all_wheel_drive",
    "engine_cylinders": 6.0,
    "engine_fuel_type": "regular_unleaded",
    "engine_hp": 268.0,
    "highway_mpg": 25,
    "make": "toyota",
    "market_category": "crossover,performance",
    "model": "venza",
    "number_of_doors": 4.0,
    "popularity": 2031,
    "transmission_type": "automatic",
    "vehicle_size": "midsize",
    "vehicle_style": "wagon",
    "year": 2013,
}

df_test = pd.DataFrame([ad])
X_test = prepare_X(df_test)

y_pred = w_0 + X_test.dot(w)

suggestion = np.expm1(y_pred)
