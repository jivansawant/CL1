

## DMV Practical 8 — Data Cleaning and Preparation

### 1. CORE TAKEAWAY

**The aim is to clean telecom customer data, handle missing values and outliers, prepare categorical and numerical features, and split the data into training and testing sets for customer churn analysis.**

### 2. MINIMALIST PYTHON CODE

**Notebook Cell 1 — Python Code**

I've corrected the column-name issue by standardizing the column headers immediately after loading the CSV. This ensures that `Churn` becomes `churn` and `customerID` becomes `customerid` regardless of their original capitalization or surrounding spaces.

Copy and run this entire cell from top to bottom.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("telecom_customer_churn.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Remove duplicates and standardize text
df = df.drop_duplicates()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip().str.lower()

# 3. Convert columns to correct data types
cols = ["seniorcitizen", "tenure", "monthlycharges", "totalcharges"]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Remove rows with missing target
df = df.dropna(subset=["churn"])

# 5. Handle missing values
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    if col not in ["customerid", "churn"] and df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# 6. Handle outliers using IQR
for col in ["monthlycharges", "totalcharges"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 7. Feature engineering
df["tenuregroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 48, float("inf")],
    labels=["new", "medium", "long"]
)

# 8. Export cleaned dataset
df.to_csv("Telecom_Customer_Churn_Cleaned.csv", index=False)

# 9. Separate features and target
X = df.drop(columns=["customerid", "churn"])
X = pd.get_dummies(X, drop_first=True, dtype=int)
y = df["churn"].map({"yes": 1, "no": 0})

# 10. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 11. Scale numerical features
scale_cols = ["tenure", "monthlycharges", "totalcharges"]
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# 12. Export encoded dataset
encoded = X.copy()
encoded["churn"] = y
encoded.to_csv("Telecom_Customer_Churn_Encoded.csv", index=False)

# Results
print("\nCleaned dataset shape:", df.shape)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("\nChurn distribution:\n", df["churn"].value_counts())
print("\nMissing values after cleaning:", df.isnull().sum().sum())
print("\nBoth datasets exported successfully!")
```

**Important:** Run the entire cell, not only the section that previously failed. If you previously ran different versions of the practical, restart the notebook kernel and execute this cell from the beginning.

---

### 3. WHY CODE WALKTHROUGH

Put the following explanation in a separate **Markdown cell**.

#### Step 1: Load and explore the data

```python
df = pd.read_csv("telecom_customer_churn.csv")
df.columns = df.columns.str.strip().str.lower()
```

The first line loads the CSV file into a DataFrame.

The second line standardizes the column names:
- `strip()` removes surrounding spaces.
- `lower()` converts names to lowercase.

For example, `Churn`, `CHURN`, and ` churn ` all become `churn`.

**This is the key correction that prevents your previous `KeyError`.**

#### Step 2: Remove duplicates and standardize text

```python
df = df.drop_duplicates()
```

Removes repeated rows so duplicate records do not distort the analysis.

We then remove unnecessary spaces and standardize text to lowercase. For example, `Yes` becomes `yes`.

#### Step 3: Convert data types

```python
df[col] = pd.to_numeric(df[col], errors="coerce")
```

Converts numerical columns into numeric data types.

`errors="coerce"` converts invalid values into missing values (`NaN`), which we can handle in the next step.

#### Step 4: Handle missing values

```python
df = df.dropna(subset=["churn"])
```

Removes rows where the target variable is missing. Without a known churn outcome, we cannot use that row as a labeled training example.

For other numerical columns, we use the **median**. For categorical columns, we use the **mode**.

#### Step 5: Handle outliers

We calculate the first quartile, third quartile, and interquartile range:

\[
IQR = Q3-Q1
\]

We then use `clip()` to restrict unusually low or high values to the calculated boundaries.

This example applies the method to monthly and total charges. Outliers can be genuine customer values, so clipping is a modeling choice rather than proof that the data was incorrect.

#### Step 6: Feature engineering

```python
df["tenuregroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 48, float("inf")],
    labels=["new", "medium", "long"]
)
```

Creates a new feature called `tenuregroup` based on how long the customer has stayed.

This can help us investigate whether newer customers have different churn behavior from long-term customers.

#### Step 7: Encode categorical features

```python
X = pd.get_dummies(X, drop_first=True, dtype=int)
```

Converts categories into numerical indicator columns so machine-learning algorithms can process them.

We also convert the target into binary values:

- `yes` → `1`
- `no` → `0`

#### Step 8: Split the data

```python
train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Divides the dataset into:
- 80% training data.
- 20% testing data.

`random_state=42` makes the split reproducible. `stratify=y` helps maintain the proportion of churn and non-churn customers in both sets.

#### Step 9: Scale numerical features

```python
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])
```

Standardizes the selected numerical features.

We fit the scaler on the training set only, then apply the same transformation to the testing set. This avoids using test-set statistics to calculate the scaling parameters.

#### Step 10: Export the results

The practical creates two files:

| File | Contents |
|---|---|
| `Telecom_Customer_Churn_Cleaned.csv` | Cleaned data with readable categories |
| `Telecom_Customer_Churn_Encoded.csv` | Numerically encoded features and churn target |

The training and testing sets are available as Python variables for further modeling.

**Technical note:** For a production machine-learning workflow, imputation and outlier boundaries should also be learned from training data only. This simplified lab version performs those cleaning operations before the split.

---

### 4. VIVA CONCEPT CRASH COURSE

| Concept | Simple explanation |
|---|---|
| Data cleaning | Correcting missing, duplicate, or inconsistent data |
| Data preparation | Making data suitable for analysis or machine learning |
| Missing values | Data entries that are unavailable |
| Median | Middle value of sorted numerical data |
| Mode | Most frequently occurring value |
| Outlier | A value unusually far from the rest of the data |
| IQR | Difference between the 75th and 25th percentiles |
| Feature engineering | Creating useful features from existing columns |
| Encoding | Converting categorical features into numerical representations |
| Standardization | Rescaling numerical data using its mean and standard deviation |
| Data leakage | Unintentionally allowing information from testing data to influence training |
| Training set | Data used to train a model |
| Testing set | Data reserved to evaluate a trained model |

### 5. TOP 3 EXPECTED VIVA QUESTIONS

**Q1. Why do we use the median to fill missing numerical values?**

**Answer:** The median is less affected by extreme values than the mean, making it useful for numerical data containing outliers.

**Q2. Why do we use `pd.get_dummies()`?**

**Answer:** It converts categorical features into numerical indicator columns that machine-learning algorithms can process.

**Q3. Why do we split the dataset into training and testing sets?**

**Answer:** The training set is used to learn patterns, while the testing set evaluates performance on unseen data.

---



