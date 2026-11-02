# CL-1 Lab Assignment 11 (Group B): Data Aggregation

## 1. Core concept

**Data aggregation** means grouping data and calculating summary values, such as totals, averages, or counts.

For this practical, we need to analyze retail sales and answer these questions:

- Which region has the highest total sales?
- What percentage of total sales comes from each region?
- How do product categories perform in different regions?
- How can we compare sales using bar charts?

Your original code is mostly correct, but it repeats the same aggregation several times and uses different plotting approaches. We can simplify it by using **Pandas for calculations and plotting**, which keeps the code shorter and easier to remember.

The sample dataset contains these columns:

| Column | Meaning |
|---|---|
| `Transaction_Date` | Date of the transaction |
| `Region` | Region where the sale occurred |
| `Product_Category` | Category of the product |
| `Quantity` | Number of items sold |
| `Sales_Amount` | Amount of the sale |

**Important:** The sample contains only five records. The code will work with a larger dataset too, provided it has the expected columns.

## 2. Complete Python code — One Cell

Copy the entire code into **one Jupyter Notebook cell** and run it.

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("Retail_Sales_Data.csv")
df.columns = df.columns.str.strip()

# 2. Explore the dataset
print("First 5 Records:")
display(df.head())

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
display(df.dtypes.to_frame("Data Type"))

print("\nMissing Values:")
display(df.isnull().sum().to_frame("Missing Values"))

print("\nSummary Statistics:")
display(df.describe(include="all").T)

# 3. Prepare the required columns
df["Sales_Amount"] = pd.to_numeric(df["Sales_Amount"], errors="coerce")
df = df.dropna(subset=["Region", "Sales_Amount"])

# 4. Calculate total sales by region
region_sales = df.groupby("Region")["Sales_Amount"].sum().sort_values(ascending=False)

print("\nTotal Sales by Region:")
display(region_sales.to_frame("Total Sales"))

# 5. Bar chart of total sales by region
region_sales.plot(kind="bar", title="Total Sales by Region", figsize=(8, 5))
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. Pie chart of sales distribution
region_sales.plot(kind="pie", autopct="%1.1f%%", title="Sales Distribution by Region", figsize=(7, 7))
plt.ylabel("")
plt.tight_layout()
plt.show()

# 7. Identify the top-performing regions
print("\nTop-Performing Regions:")
display(region_sales.head().to_frame("Total Sales"))

print("Highest-Sales Region:", region_sales.idxmax())
print("Highest Total Sales:", region_sales.max())

# 8. Calculate sales by region and product category
region_category_sales = df.groupby(
    ["Region", "Product_Category"]
)["Sales_Amount"].sum().unstack(fill_value=0)

print("\nSales by Region and Product Category:")
display(region_category_sales)

# 9. Stacked bar chart
region_category_sales.plot(
    kind="bar", stacked=True,
    title="Sales by Region and Product Category",
    figsize=(9, 5)
)
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 10. Grouped bar chart
region_category_sales.plot(
    kind="bar",
    title="Grouped Sales Comparison by Region and Product Category",
    figsize=(9, 5)
)
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

---

## 3. Code breakdown — Detailed explanation

### Step 1: Import libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
```

- **Pandas (`pd`)** handles loading, cleaning, grouping, and summarizing the data.
- **Matplotlib (`plt`)** helps configure and display the charts.
- We do not need NumPy or Seaborn for this practical, so we have removed them.

### Step 2: Load the dataset

```python
df = pd.read_csv("Retail_Sales_Data.csv")
df.columns = df.columns.str.strip()
```

- `pd.read_csv()` reads the CSV file into a DataFrame called `df`.
- `df.columns` accesses the column names.
- `.str.strip()` removes unwanted spaces from the beginning or end of each column name.

For example, a column named `" Region "` becomes `"Region"`.

**Remember:** The CSV file must be in the notebook's current working directory, or you must provide its correct file path.

### Step 3: Explore the dataset

**Display the first five rows:**

```python
display(df.head())
```

`head()` displays the first five records by default. It helps us understand what the dataset contains.

**Check the dataset's dimensions:**

```python
print("Dataset Shape:", df.shape)
```

`shape` returns a tuple:

`(number_of_rows, number_of_columns)`

For example, `(100, 5)` means 100 rows and 5 columns.

**Display column names:**

```python
print(df.columns.tolist())
```

This shows the available columns as a Python list. It helps us check the names before performing calculations.

**Check data types:**

```python
display(df.dtypes.to_frame("Data Type"))
```

This tells us whether each column contains integers, floating-point numbers, dates stored as text, or other data types.

For example, `Sales_Amount` should be numeric so that we can calculate its total.

**Check missing values:**

```python
display(df.isnull().sum().to_frame("Missing Values"))
```

Here is how the expression works:

1. `df.isnull()` identifies missing values.
2. `.sum()` counts missing values in each column.
3. `.to_frame()` displays the result as a table.

**Display summary statistics:**

```python
display(df.describe(include="all").T)
```

- `describe()` generates summary statistics.
- For numerical columns, these can include the count, mean, standard deviation, minimum, maximum, and quartiles.
- For categorical columns, the output can include the number of non-missing values, unique values, most frequent value, and its frequency.
- `.T` transposes the output so that each original column appears as a row, making it easier to read.

### Step 4: Prepare the sales data

```python
df["Sales_Amount"] = pd.to_numeric(df["Sales_Amount"], errors="coerce")
df = df.dropna(subset=["Region", "Sales_Amount"])
```

**First line:**

`pd.to_numeric()` converts `Sales_Amount` into a numeric column.

The parameter `errors="coerce"` converts values that cannot be interpreted as numbers into `NaN`, which represents a missing value.

**Second line:**

`dropna(subset=["Region", "Sales_Amount"])` removes records that have a missing region or sales amount.

This is important because:
- A missing region prevents us from assigning the sale to a region.
- A missing sales amount prevents us from calculating that transaction's contribution to total sales.

We do not remove records simply because `Quantity` is missing, because this practical calculates sales using `Sales_Amount`, not quantity.

### Step 5: Group sales by region

```python
region_sales = df.groupby("Region")["Sales_Amount"].sum().sort_values(ascending=False)
```

**This is the most important line in the practical.**

Let's break it into four parts.

**1. `groupby("Region")`**

Groups all transactions belonging to the same region.

For example:

| Region | Sales_Amount |
|---|---:|
| South | 1000 |
| West | 2000 |
| South | 1500 |
| West | 500 |

**2. `["Sales_Amount"]`**

Selects the sales amount column to be aggregated.

**3. `.sum()`**

Adds the sales amounts within each region.

The result becomes:

| Region | Total Sales |
|---|---:|
| South | 2500 |
| West | 2500 |

**4. `.sort_values(ascending=False)`**

Sorts the totals from highest to lowest.

This makes it easier to identify the best-performing regions.

Finally:

```python
display(region_sales.to_frame("Total Sales"))
```

Converts the resulting Series into a DataFrame with the column name `Total Sales`.

**Viva point:** `groupby()` divides data into groups, and `sum()` calculates the total for each group.

### Step 6: Create a bar chart

```python
region_sales.plot(kind="bar", title="Total Sales by Region", figsize=(8, 5))
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

We use Pandas' built-in plotting method instead of manually creating bars with Matplotlib.

- `kind="bar"` creates a bar chart.
- `title` specifies the chart title.
- `figsize=(8, 5)` sets the figure size.
- `xlabel()` labels the horizontal axis.
- `ylabel()` labels the vertical axis.
- `xticks(rotation=0)` keeps region names horizontal.
- `tight_layout()` adjusts spacing to prevent labels from being cut off.
- `show()` displays the chart.

**How to interpret it:** Each bar represents a region. A taller bar indicates a larger total sales amount.

### Step 7: Create a pie chart

```python
region_sales.plot(kind="pie", autopct="%1.1f%%", title="Sales Distribution by Region", figsize=(7, 7))
plt.ylabel("")
plt.tight_layout()
plt.show()
```

This chart shows each region's contribution to the total sales.

- `kind="pie"` creates a pie chart.
- `autopct="%1.1f%%"` displays each slice's percentage to one decimal place.
- `plt.ylabel("")` removes the default y-axis label.

The percentage for a region is calculated as:

\[
\text{Sales Percentage} =
\frac{\text{Sales of Region}}{\text{Total Sales of All Regions}}
\times 100
\]

For example, if a region generates ₹25,000 in sales out of ₹100,000 total sales:

\[
\frac{25,000}{100,000}\times100=25\%
\]

The pie chart would display that region as 25% of total sales.

**Remember:** A bar chart is generally easier for comparing the exact performance of many regions, while a pie chart emphasizes each region's share of the total.

### Step 8: Identify the top-performing regions

```python
display(region_sales.head().to_frame("Total Sales"))
```

Since `region_sales` is already sorted in descending order, `head()` displays the five highest-sales regions by default.

```python
print("Highest-Sales Region:", region_sales.idxmax())
print("Highest Total Sales:", region_sales.max())
```

- `idxmax()` returns the index label—in this case, the region—with the highest sales.
- `max()` returns the highest sales value.

For example, if West has the largest total, `idxmax()` returns `West`.

**Important:** If multiple regions have exactly the same highest total, `idxmax()` returns the first one encountered in the Series. It does not list all tied regions.

### Step 9: Aggregate sales by region and product category

```python
region_category_sales = df.groupby(
    ["Region", "Product_Category"]
)["Sales_Amount"].sum().unstack(fill_value=0)
```

This extends the earlier aggregation by grouping on **two columns instead of one**.

**1. `groupby(["Region", "Product_Category"])`**

Creates groups based on both the region and the product category.

For example:
- North + Grocery
- North + Furniture
- South + Grocery
- South + Electronics

Each combination is treated as a separate group.

**2. `["Sales_Amount"].sum()`**

Adds the sales amounts for each region-category combination.

**3. `.unstack(fill_value=0)`**

Moves the product category from the row index into columns.

The result might look like this:

| Region | Electronics | Furniture | Grocery |
|---|---:|---:|---:|
| East | 4000 | 2500 | 1500 |
| North | 3000 | 3500 | 2000 |
| South | 2500 | 1500 | 4000 |

These numbers are illustrative, not actual results from your file.

`fill_value=0` fills missing region-category combinations with zero. For example, if there are no Electronics sales in South, that cell displays `0`.

**Viva point:** `unstack()` reshapes grouped data so that one level of the index becomes columns. This makes the result suitable for comparing categories across regions.

### Step 10: Create a stacked bar chart

```python
region_category_sales.plot(
    kind="bar", stacked=True,
    title="Sales by Region and Product Category",
    figsize=(9, 5)
)
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

The key parameter is:

```python
stacked=True
```

It stacks the product-category bars on top of one another for each region.

**How to interpret it:**
- Each complete bar represents a region.
- Each colored section represents a product category.
- The total bar height represents total sales for that region.
- The size of each section shows how much that category contributes to the region's total.

A stacked bar chart is useful when you want to compare both total regional sales and the composition of those sales.

### Step 11: Create a grouped bar chart

```python
region_category_sales.plot(
    kind="bar",
    title="Grouped Sales Comparison by Region and Product Category",
    figsize=(9, 5)
)
plt.xlabel("Region")
plt.ylabel("Total Sales Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

Here, we omit `stacked=True`. Pandas therefore draws the bars side by side for each region.

**How to interpret it:**
- Each group of bars represents a region.
- Each bar within the group represents a product category.
- The bar heights allow you to compare a particular category across regions.

For example, you can compare Grocery sales in North, South, East, and West.

**Difference between the two charts:**

| Stacked bar chart | Grouped bar chart |
|---|---|
| Bars are stacked on top of each other. | Bars appear side by side. |
| Useful for understanding total sales and their composition. | Useful for comparing individual categories. |
| Each full bar represents the region's combined sales. | Each category has its own separate bar. |

---

## 4. Viva concepts to remember

| Concept | Meaning |
|---|---|
| Data aggregation | Summarizing data using totals, averages, counts, and similar calculations. |
| `groupby()` | Divides records into groups based on one or more columns. |
| `sum()` | Calculates the total of values in each group. |
| `sort_values()` | Sorts values in ascending or descending order. |
| `idxmax()` | Returns the index label corresponding to the maximum value. |
| `max()` | Returns the maximum value. |
| `unstack()` | Converts one index level into columns. |
| `fill_value=0` | Fills missing combinations with zero during reshaping. |
| Bar chart | Compares values across categories. |
| Pie chart | Shows the percentage contribution of each category to a total. |
| Stacked bar chart | Shows totals and their component categories. |
| Grouped bar chart | Compares individual categories side by side. |
| `describe()` | Generates summary statistics. |
| `isnull().sum()` | Counts missing values in each column. |

### Three likely viva questions

**Q1. What is data aggregation?**

Data aggregation is the process of grouping data and calculating summary values such as sum, mean, count, minimum, or maximum. In this practical, we use aggregation to calculate total sales by region.

**Q2. What is the difference between `groupby()` and `unstack()`?**

`groupby()` divides data into groups for calculations. `unstack()` reshapes a level of the resulting index into columns. Here, `groupby()` calculates sales for each region and product category, while `unstack()` places product categories into separate columns.

**Q3. Why do we use stacked and grouped bar charts?**

A stacked bar chart shows each region's total sales and the contribution of each product category. A grouped bar chart places categories side by side, making it easier to compare individual category sales across regions.

---

## 5. Quick revision sequence

Memorize the practical in this order:

1. **Load:** `pd.read_csv()`
2. **Explore:** `head()`, `shape`, `dtypes`, `isnull().sum()`, `describe()`
3. **Prepare:** Convert sales to numeric and remove records with missing region or sales amount.
4. **Aggregate:** `groupby("Region")["Sales_Amount"].sum()`
5. **Visualize:** Bar chart and pie chart using `.plot()`
6. **Find the top region:** `idxmax()` and `max()`
7. **Aggregate two columns:** `groupby(["Region", "Product_Category"])["Sales_Amount"].sum()`
8. **Reshape:** `unstack(fill_value=0)`
9. **Compare categories:** Stacked and grouped bar charts.

**One important point:** The highest-sales region is not necessarily the most profitable region. This dataset contains sales amounts, not costs or profit margins, so this practical identifies the region with the highest **total sales**, not the highest profit.