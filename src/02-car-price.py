import pandas as pd  # for numerical operations
import numpy as np  # for tabular data
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/cardataset.csv")
df.head()

# Lowercase all the column names and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Select only the columns with string values
string_columns = list(df.dtypes[df.dtypes == "object"].index)

# Lowercase and replace spaces with underscores for values in all string columns of the DataFrame
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

df.head()

# Plot the target variables histogram (distribution of prices)
sns.histplot(df.msrp, bins=40)

# Zoom in
sns.histplot(df.msrp[df.msrp < 100_000])

# Apply the log on the prices Ynew = log(y + 1)
log_price = np.log1p(df.msrp)
sns.histplot(log_price)

# Check the missing values
df.isnull().sum()
