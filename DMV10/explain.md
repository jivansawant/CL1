# CL-1 Lab Assignment 4 (Group B): Data Visualization Using Matplotlib

## 1. Core concept

The aim of this practical is to visualize **Air Quality Index (AQI) trends and pollutant levels over time** using Python.

We will use only **Pandas and Matplotlib**. We can simplify your original code by removing automatic column detection, repeated conditions, and unnecessary plots.

Our dataset contains five columns:

| Column | Meaning |
|---|---|
| `Date` | Date of measurement |
| `PM2.5` | Fine particulate matter concentration |
| `PM10` | Coarse particulate matter concentration |
| `CO` | Carbon monoxide level |
| `AQI` | Air Quality Index |

We will create five useful visualizations:
1. Line plot of AQI over time.
2. Line plots of pollutant levels.
3. Bar chart of average monthly AQI.
4. Box plot showing the distribution of AQI.
5. Scatter plots showing the relationship between AQI and pollutants.

## 2. Complete Python code — One Cell

Copy the entire code into **one Jupyter Notebook cell**.

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and explore the dataset
df = pd.read_csv("City_Air_Quality.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

print("First 5 Records:")
display(df.head())

print("Dataset Shape:", df.shape)
print("\nMissing Values:")
display(df.isnull().sum())

print("\nSummary Statistics:")
display(df.describe())

# 2. AQI trend over time
plt.figure(figsize=(8, 4))
plt.plot(df["Date"], df["AQI"], marker="o")
plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Pollutant levels over time
for col in ["PM2.5", "PM10", "CO"]:
    plt.figure(figsize=(8, 4))
    plt.plot(df["Date"], df[col], marker="o", label=col)
    plt.title(col + " Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. Average monthly AQI
df["Month"] = df["Date"].dt.to_period("M").astype(str)
monthly_aqi = df.groupby("Month")["AQI"].mean()

monthly_aqi.plot(kind="bar", figsize=(8, 4), title="Average Monthly AQI")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. AQI distribution using a box plot
plt.figure(figsize=(6, 4))
plt.boxplot(df["AQI"].dropna())
plt.title("AQI Distribution")
plt.ylabel("AQI")
plt.show()

# 6. Relationship between AQI and pollutants
for col in ["PM2.5", "PM10", "CO"]:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[col], df["AQI"])
    plt.title("AQI vs " + col)
    plt.xlabel(col)
    plt.ylabel("AQI")
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

- **Pandas:** Loads and processes the dataset.
- **Matplotlib:** Creates graphs and charts.

We do not need Seaborn or NumPy for this practical.

### Step 2: Load and prepare the data

```python
df = pd.read_csv("City_Air_Quality.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
```

**`pd.read_csv()`** reads the CSV file and stores it in a DataFrame named `df`.

**`pd.to_datetime()`** converts the Date column into a proper date format. This allows Python to recognize dates and extract information such as the month or year.

**`sort_values("Date")`** arranges the records from earliest to latest, ensuring that the line chart displays the trend in chronological order.

### Step 3: Explore the dataset

```python
display(df.head())
print(df.shape)
display(df.isnull().sum())
display(df.describe())
```

- `head()` displays the first five records.
- `shape` gives the number of rows and columns.
- `isnull().sum()` counts missing values in each column.
- `describe()` calculates numerical summary statistics, including mean, minimum, maximum, and quartiles.

These commands help us understand the data before visualizing it.

### Step 4: Plot the AQI trend

```python
plt.figure(figsize=(8, 4))
plt.plot(df["Date"], df["AQI"], marker="o")
plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

This graph shows how AQI changes with time.

| Command | Purpose |
|---|---|
| `plt.figure()` | Creates a new figure. |
| `plt.plot()` | Creates a line plot. |
| `marker="o"` | Places a circle at each data point. |
| `plt.title()` | Adds a title. |
| `plt.xlabel()` | Labels the x-axis. |
| `plt.ylabel()` | Labels the y-axis. |
| `plt.xticks(rotation=45)` | Rotates date labels to improve readability. |
| `plt.tight_layout()` | Adjusts spacing around the graph. |
| `plt.show()` | Displays the graph. |

**Interpretation:** Rising AQI values indicate worsening air quality, while falling values indicate improving air quality, assuming the same AQI scale is used throughout.

### Step 5: Plot individual pollutant levels

```python
for col in ["PM2.5", "PM10", "CO"]:
    plt.figure(figsize=(8, 4))
    plt.plot(df["Date"], df[col], marker="o", label=col)
    plt.title(col + " Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

Instead of writing three separate sets of plotting commands, we use a **for loop**.

The loop selects each pollutant in turn:
- `PM2.5`
- `PM10`
- `CO`

For every pollutant, it creates a separate line plot.

**Why separate graphs?** Pollutants may use different measurement scales. Separate plots make their individual trends easier to read.

`plt.legend()` displays the pollutant label on the graph.

### Step 6: Compare average monthly AQI

```python
df["Month"] = df["Date"].dt.to_period("M").astype(str)
monthly_aqi = df.groupby("Month")["AQI"].mean()
```

The first line extracts the month and year from each date.

For example:

`2025-01-01` becomes `2025-01`.

The second line groups the records by month and calculates the **average AQI for each month**.

We then plot the results:

```python
monthly_aqi.plot(kind="bar", figsize=(8, 4), title="Average Monthly AQI")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

- `kind="bar"` creates a bar chart.
- Each bar represents one month.
- The height of the bar represents that month's average AQI.

**Important:** Your sample shows only January 2025. If the full dataset contains only January, the chart will have just one bar. Multiple months are needed to compare monthly trends.

### Step 7: Analyze AQI distribution using a box plot

```python
plt.figure(figsize=(6, 4))
plt.boxplot(df["AQI"].dropna())
plt.title("AQI Distribution")
plt.ylabel("AQI")
plt.show()
```

A box plot summarizes how AQI values are distributed.

It displays:
- **Median:** The middle AQI value.
- **Box:** The middle 50% of the observations.
- **Whiskers:** The range of values within the box-plot rule.
- **Potential outliers:** Values shown individually beyond the whiskers.

`dropna()` excludes missing AQI values from the plot.

**Why use a box plot?** It helps us understand the spread of AQI values and identify unusually high or low observations. A potential outlier is not necessarily an error.

### Step 8: Explore the relationship between AQI and pollutants

```python
for col in ["PM2.5", "PM10", "CO"]:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[col], df["AQI"])
    plt.title("AQI vs " + col)
    plt.xlabel(col)
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()
```

This loop creates three scatter plots.

- The **x-axis** represents the pollutant level.
- The **y-axis** represents AQI.
- Each dot represents one observation.

**Interpretation:**
- An upward pattern suggests that higher pollutant levels tend to occur alongside higher AQI values.
- A downward pattern suggests an inverse relationship.
- A scattered pattern suggests a weaker or more complicated relationship.

A scatter plot can reveal an association, but it does not prove that one pollutant caused the AQI change.

---

## 4. Viva concepts to remember

| Concept | Meaning |
|---|---|
| Data visualization | Representing data graphically to identify patterns and trends. |
| Matplotlib | A Python library for creating visualizations. |
| Line plot | Shows changes in values over time or another ordered variable. |
| Bar chart | Compares values across categories or periods. |
| Box plot | Summarizes a distribution and helps identify potential outliers. |
| Scatter plot | Shows the relationship between two numerical variables. |
| `plot()` | Creates a line plot by default. |
| `scatter()` | Creates a scatter plot. |
| `boxplot()` | Creates a box plot. |
| `groupby()` | Groups records to calculate summary statistics. |
| `mean()` | Calculates the arithmetic average. |
| `to_datetime()` | Converts values into date/time data. |
| `sort_values()` | Sorts rows by the specified column. |
| `tight_layout()` | Adjusts spacing to help prevent labels from being cut off. |

### Three likely viva questions

**Q1. Why do we use a line plot for AQI?**

A line plot shows how AQI changes over time, making trends, rises, and falls easier to identify.

**Q2. What is the difference between a line plot and a scatter plot?**

A line plot connects data points to show a trend, while a scatter plot displays individual points to examine the relationship between two numerical variables.

**Q3. Why do we use a box plot?**

A box plot summarizes the distribution of AQI values using the median, quartiles, whiskers, and potential outliers.

---

## 5. Quick revision sequence

Remember these steps for the practical exam:

1. **Import:** `pandas` and `matplotlib.pyplot`.
2. **Load:** `pd.read_csv()`.
3. **Explore:** `head()`, `shape`, `isnull().sum()`, `describe()`.
4. **Prepare dates:** `pd.to_datetime()` and `sort_values()`.
5. **AQI trend:** `plt.plot()`.
6. **Pollutant trends:** A `for` loop with `plt.plot()`.
7. **Monthly comparison:** `groupby()` + `mean()` + bar chart.
8. **Distribution:** `plt.boxplot()`.
9. **Relationship:** `plt.scatter()`.

**One technical note:** This simplified code assumes the CSV has the exact column names shown in your sample. It does not automatically repair missing or invalid dates, and it does not fill missing pollutant values. If your actual dataset contains those problems, additional cleaning may be needed before plotting.