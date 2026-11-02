# DMV Practical 9 — Data Wrangling

## 1. CORE TAKEAWAY

**Data wrangling is the process of cleaning, transforming, combining, filtering, and summarizing raw data so that it is ready for analysis.** In this practical, we'll prepare real estate data, compare average property prices by neighborhood and property type, calculate price per square foot, and identify potential outliers.

**Important correction:** Your original code has a column-name mismatch. The cleaning step converts `Sale Price ($)` into `sale_price`, not `sale_price_`. Similarly, `Area (sq ft)` becomes `area_sq_ft`. I'll use consistent column names throughout the code so these errors don't occur.

## 2. MINIMALIST PYTHON CODE

### Notebook Cell 1 — Python Code

This version keeps all seven required operations, avoids unnecessary functions, and uses consistent column names throughout.

**Dataset:** Save your file as `RealEstate_Prices.csv` in the same folder as your notebook.

```python
import pandas as pd

# 1. Load dataset
df = pd.read_csv("RealEstate_Prices.csv")

# 2. Clean column names
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^\w]+", "_", regex=True)
              .str.strip("_"))

print("Original Data:")
print(df.head())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Ensure numeric columns have numeric data types
cols = ["property_id", "bedrooms", "bathrooms", "area_sq_ft",
        "year_built", "sale_price", "sale_year"]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows missing essential information
df = df.dropna(subset=["property_id", "sale_price", "area_sq_ft", "sale_year"])

# 3. Handle remaining missing values
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any() and not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate records
df = df.drop_duplicates()

# Standardize text
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip().str.lower()

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# 4. Merge neighborhood information (example data)
neighborhood_data = pd.DataFrame({
    "neighborhood": ["downtown", "suburb", "riverside",
                     "hillview", "lakeside"],
    "avg_income": [75000, 65000, 60000, 70000, 72000]
})

df = pd.merge(df, neighborhood_data, on="neighborhood", how="left")

# 5. Filter properties sold from 2021 to 2023
filtered_data = df[df["sale_year"].between(2021, 2023)]
print("\nProperties Sold from 2021 to 2023:")
print(filtered_data.head())

# 6. Encode categorical columns
df_encoded = pd.get_dummies(
    df, columns=["property_type", "neighborhood"], dtype=int
)

# 7. Aggregate average prices
print("\nAverage Price by Neighborhood:")
print(df.groupby("neighborhood")["sale_price"].mean())

print("\nAverage Price by Property Type:")
print(df.groupby("property_type")["sale_price"].mean())

# Price per square foot
df["price_per_sq_ft"] = df["sale_price"] / df["area_sq_ft"]

print("\nPrice per Square Foot:")
print(df[["property_id", "price_per_sq_ft"]].head())

# 8. Identify outliers using IQR
q1 = df["sale_price"].quantile(0.25)
q3 = df["sale_price"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("\nPotential Sale Price Outliers:")
print(df[~df["sale_price"].between(lower, upper)][
    ["property_id", "sale_price"]
])

# Remove outliers from sale price and area
q1_area = df["area_sq_ft"].quantile(0.25)
q3_area = df["area_sq_ft"].quantile(0.75)
iqr_area = q3_area - q1_area

lower_area = q1_area - 1.5 * iqr_area
upper_area = q3_area + 1.5 * iqr_area

df_clean = df[
    df["sale_price"].between(lower, upper) &
    df["area_sq_ft"].between(lower_area, upper_area)
].copy()

print("\nShape Before Outlier Removal:", df.shape)
print("Shape After Outlier Removal:", df_clean.shape)

# 9. Export results
df_clean.to_csv("RealEstate_Cleaned.csv", index=False)
df_encoded.to_csv("RealEstate_Encoded.csv", index=False)

print("\nDatasets exported successfully!")
```

**Note about the merge:** The neighborhood income values are illustrative example data, not verified real-world statistics. The merge demonstrates how additional neighborhood information can be combined with the property dataset. In an actual analysis, use a reliable external dataset.

---

## 3. WHY CODE WALKTHROUGH

Copy this explanation into a separate **Markdown cell** in your notebook.

### Step 1: Load and inspect the dataset

```python
df = pd.read_csv("RealEstate_Prices.csv")
```

Loads the CSV file into a Pandas DataFrame.

We inspect `df.head()`, `df.shape`, and `df.isnull().sum()` to understand the data before modifying it.

- `head()` displays the first five rows.
- `shape` shows the number of rows and columns.
- `isnull().sum()` counts missing values in each column.

### Step 2: Clean column names

```python
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^\w]+", "_", regex=True)
              .str.strip("_"))
```

This standardizes column names so they are easier to reference in Python.

For example:

| Original column | Cleaned column |
|---|---|
| `Property ID` | `property_id` |
| `Property Type` | `property_type` |
| `Area (sq ft)` | `area_sq_ft` |
| `Sale Price ($)` | `sale_price` |
| `Sale Year` | `sale_year` |

This step is particularly important because inconsistent column names can cause `KeyError` exceptions.

### Step 3: Convert data types and handle missing values

```python
df[col] = pd.to_numeric(df[col], errors="coerce")
```

Converts the selected columns into numerical values. Invalid entries become missing values.

We remove rows missing essential information such as property ID, sale price, area, or sale year.

For other numerical columns, we fill missing values with the median. For categorical columns, we use the most frequent value when available.

We also remove duplicate records using:

```python
df = df.drop_duplicates()
```

**Why?** Missing values, incorrect data types, and duplicate records can distort statistical summaries.

### Step 4: Merge neighborhood information

```python
df = pd.merge(df, neighborhood_data, on="neighborhood", how="left")
```

A merge combines two DataFrames using a common column.

Here, `neighborhood` connects the property records to the neighborhood income information.

- `on="neighborhood"` specifies the matching column.
- `how="left"` preserves all property records, even if matching neighborhood information is unavailable.

This operation is useful when combining property prices with external information such as neighborhood income, population, or access to amenities.

### Step 5: Filter and subset the data

```python
filtered_data = df[df["sale_year"].between(2021, 2023)]
```

Selects properties sold between 2021 and 2023, including both years.

Filtering helps us focus on a particular time period or group of properties without modifying the original DataFrame.

### Step 6: Encode categorical variables

```python
df_encoded = pd.get_dummies(
    df, columns=["property_type", "neighborhood"], dtype=int
)
```

Converts categorical columns into numerical indicator columns.

For example, a `property_type` column containing `villa` and `apartment` can become:

| property_type_villa | property_type_apartment |
|---:|---:|
| 1 | 0 |
| 0 | 1 |

This representation is useful for numerical analysis and machine-learning algorithms.

### Step 7: Aggregate property prices

```python
df.groupby("neighborhood")["sale_price"].mean()
```

Groups properties by neighborhood and calculates the average sale price for each group.

Similarly:

```python
df.groupby("property_type")["sale_price"].mean()
```

Calculates the average sale price for each property type.

These summaries help us explore whether prices differ across neighborhoods and property types.

We also calculate:

```python
df["price_per_sq_ft"] = df["sale_price"] / df["area_sq_ft"]
```

This creates a new feature representing the sale price per square foot, making properties of different sizes easier to compare.

### Step 8: Detect and handle outliers

We calculate the interquartile range:

\[
IQR = Q3-Q1
\]

The potential outlier boundaries are:

\[
\text{Lower bound} = Q1-1.5(IQR)
\]

\[
\text{Upper bound} = Q3+1.5(IQR)
\]

Values outside these boundaries are flagged as potential outliers.

The code applies this method to both sale price and property area. It then creates `df_clean`, which excludes properties outside either range.

**Remember:** An unusually expensive property may be completely legitimate. Outlier removal should depend on the purpose of the analysis, rather than automatically treating every extreme value as an error.

### Step 9: Export the results

```python
df_clean.to_csv("RealEstate_Cleaned.csv", index=False)
```

Exports the cleaned dataset.

The code also exports an encoded dataset for subsequent numerical analysis.

You should obtain:
- `RealEstate_Cleaned.csv`
- `RealEstate_Encoded.csv`

The cleaned file contains the properties retained after outlier removal. The encoded file in this version is created before outlier removal, so it retains the original rows used for encoding.

---

## 4. VIVA CONCEPT CRASH COURSE

| Concept | Simple explanation |
|---|---|
| Data wrangling | Transforming raw data into a usable form |
| Data cleaning | Correcting missing, duplicate, or inconsistent data |
| Data merging | Combining datasets using a common column |
| Filtering | Selecting rows that satisfy a condition |
| Subsetting | Selecting particular rows or columns |
| Encoding | Converting categorical data into numerical representations |
| Aggregation | Summarizing data using operations such as mean, sum, or count |
| Feature engineering | Creating useful new columns from existing data |
| Outlier | An observation that differs substantially from the rest |
| IQR | Difference between the 75th and 25th percentiles |
| `groupby()` | Groups rows according to a specified category |
| `pd.merge()` | Combines DataFrames using matching values |
| `pd.get_dummies()` | Creates indicator columns for categorical variables |
| `to_csv()` | Exports a DataFrame to a CSV file |

### Important distinction: Data cleaning vs. data wrangling

**Data cleaning** focuses on correcting problems in the data, such as missing values, duplicates, and inconsistent formatting.

**Data wrangling** is broader. It includes cleaning, merging, filtering, encoding, transforming, and aggregating data.

Therefore, data cleaning is one part of data wrangling.

---

## 5. TOP 3 EXPECTED VIVA QUESTIONS

**Q1. What is data wrangling, and why is it necessary?**

**Answer:** Data wrangling transforms raw data into a structured, consistent, and usable format for analysis or modeling.

**Q2. What is the difference between `merge()` and `groupby()` in Pandas?**

**Answer:** `merge()` combines datasets using matching columns, whereas `groupby()` groups observations and calculates summaries for each group.

**Q3. How does the IQR method identify outliers?**

**Answer:** It calculates the first and third quartiles, finds their difference, and flags observations below \(Q1-1.5(IQR)\) or above \(Q3+1.5(IQR)\) as potential outliers.

---

### Quick revision

Remember the sequence:

**Load → Clean column names → Handle missing values → Remove duplicates → Merge → Filter → Encode → Aggregate → Calculate derived metrics → Detect outliers → Export.**

**Exam tip:** Your sample contains only five properties. With such a small dataset, IQR-based outlier removal can be unstable and may remove legitimate observations. Explain the method confidently, but remember that the decision to remove outliers should be justified by the data and analysis objective.