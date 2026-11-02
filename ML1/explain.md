# ML Practical 1: Feature Transformation Using PCA

## 1. CORE TAKEAWAY

**Principal Component Analysis (PCA) reduces the number of numerical features in a dataset while preserving as much information about variation in the data as possible.** In this practical, we standardize the wine measurements, transform the original 13 features into 2 principal components, and visualize whether wine samples from different `Customer_Segment` classes form distinct groups.

**First, understand the dataset:** Your `Wine.csv` contains 13 wine measurements and one target column, `Customer_Segment`. The sample shows three segment labels in the original wine dataset, so we'll use those labels to color the graph.

One important clarification: your problem statement mentions distinguishing red and white wine, but your provided dataset uses `Customer_Segment` and appears to be the classic wine classification dataset with three wine classes. **We should follow your actual dataset rather than incorrectly label its classes as red and white wine.**

---

## 2. MINIMALIST PYTHON CODE

**Markdown Cell 1 — Complete practical code**

Copy this code into a single Jupyter Notebook code cell. It includes the essential PCA operations, a small output summary, and one visualization.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("Wine.csv")

# Separate features and target
X = df.drop(columns=["Customer_Segment"])
y = df["Customer_Segment"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce 13 features to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the reduced data
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Customer_Segment"] = y.values

# Display results
print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_pca.shape[1])
print("Variance explained:", pca.explained_variance_ratio_.round(3))
print("Total variance retained:",
      round(pca.explained_variance_ratio_.sum() * 100, 2), "%")

# Visualize the principal components
for segment in sorted(y.unique()):
    data = pca_df[pca_df["Customer_Segment"] == segment]
    plt.scatter(data["PC1"], data["PC2"], label=f"Segment {segment}")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Wine Dataset")
plt.legend()
plt.show()
```

### What this code accomplishes

| Operation | Result |
|---|---|
| Load dataset | Reads `Wine.csv` |
| Separate features and target | Keeps the 13 measurements separate from the segment labels |
| Standardization | Puts measurements on a comparable scale |
| PCA transformation | Converts 13 features into 2 principal components |
| Variance calculation | Shows how much variation the two components preserve |
| Scatter plot | Helps visualize whether different wine segments separate |

**Why are we using only two components?**

Two components allow us to visualize the data on a simple 2D graph. This does not guarantee that all the original information is preserved. We measure how much information about variation remains using the explained variance ratio.

---

## 3. WHY CODE WALKTHROUGH

**Markdown Cell 2 — Explanation and concepts**

### Step 1: Import libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

- `pandas`: Loads and manages the dataset.
- `matplotlib`: Creates the scatter plot.
- `StandardScaler`: Standardizes the measurements.
- `PCA`: Performs dimensionality reduction.

We don't need NumPy separately because pandas and scikit-learn handle the operations required for this practical.

### Step 2: Load the dataset

```python
df = pd.read_csv("Wine.csv")
```

This loads the CSV file into a DataFrame called `df`.

Your dataset contains measurements such as:
- Alcohol
- Malic acid
- Ash
- Magnesium
- Flavanoids
- Color intensity
- Proline

There are 13 numerical features and one target column.

**Why do we load the dataset first?**

PCA needs numerical data to analyze relationships and variation between features. We first load the data so that we can select the measurements we want to transform.

### Step 3: Separate features and target

```python
X = df.drop(columns=["Customer_Segment"])
y = df["Customer_Segment"]
```

This is a very important step.

**`X` contains the input features.** These are the 13 wine measurements that PCA will analyze.

**`y` contains the target labels.** These indicate which wine segment each sample belongs to.

Why keep them separate?

PCA is an unsupervised dimensionality-reduction technique. It finds directions of maximum variation using the input features, without using the target labels to decide those directions.

We keep `y` so that we can color the graph according to the known segments afterward.

**Remember:** The target column must not be included in the PCA input for this practical.

### Step 4: Standardize the features

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This is one of the most important steps in PCA.

Consider two features:

- Alcohol might have values around 14.
- Proline might have values around 1,000.

PCA looks for directions that capture the greatest variation in the data. If features have very different scales, features with large numerical variations can disproportionately influence the result.

Standardization transforms each feature so that it has approximately:
- Mean = 0
- Standard deviation = 1

The formula, written in plain text:

**Standardized value = (Original value − Feature mean) / Feature standard deviation**

What do the two functions do?

- `fit()`: Learns the mean and standard deviation of each feature.
- `transform()`: Uses those values to standardize the measurements.

`fit_transform()` performs both operations together.

Why do we need this?

It allows PCA to analyze the measurements on comparable scales rather than allowing a feature's numerical magnitude to dominate simply because of its units.

### Step 5: Apply PCA

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

This is the main operation of the practical.

Originally, each wine sample has 13 numerical features.

After PCA, each sample has only 2 new features:
- PC1
- PC2

These are called **principal components**.

What is a principal component?

A principal component is a new variable formed by combining the original features using calculated weights.

For example, PC1 may combine information from alcohol, flavanoids, proline, and other measurements. Its exact weights are determined from the dataset by PCA.

- **PC1** captures the greatest possible variation along a single direction.
- **PC2** captures the greatest possible variation remaining in a direction perpendicular to PC1.

The components are calculated from the standardized measurements.

**Why is this useful?**

Instead of working with 13 measurements, we can represent each sample using just two numbers. This makes the data easier to visualize and may simplify later analysis.

However, the two components might not preserve all the original information.

### Step 6: Store the transformed data

```python
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Customer_Segment"] = y.values
```

The PCA transformation returns an array of numerical values.

We convert it into a DataFrame to make the results easier to understand.

We then add the original segment labels.

For example, the resulting structure looks like this:

| PC1 | PC2 | Customer_Segment |
|---:|---:|---:|
| -1.2 | 0.8 | 1 |
| 0.5 | -1.1 | 2 |
| 1.4 | 0.2 | 3 |

These are illustrative values, not actual results from your dataset.

Notice that the segment labels remain unchanged. PCA transforms the features, not the labels.

### Step 7: Display the variance explained

```python
print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_pca.shape[1])
print("Variance explained:", pca.explained_variance_ratio_.round(3))
print("Total variance retained:",
      round(pca.explained_variance_ratio_.sum() * 100, 2), "%")
```

This tells us whether reducing the dataset to two components retains a substantial amount of variation.

**Explained variance ratio** measures the proportion of the dataset's total variation captured by each component.

For example, suppose the output is:

```text
Variance explained: [0.362 0.192]
Total variance retained: 55.4 %
```

These values are hypothetical and are included only to demonstrate how to interpret the output.

They would mean:
- PC1 explains 36.2% of the total variation.
- PC2 explains 19.2%.
- Together, they explain 55.4%.

The remaining 44.6% of the original variation is not represented by these two components.

The formula is:

**Total variance retained = (Variance explained by PC1 + Variance explained by PC2) × 100**

A higher percentage means the two components preserve more of the original variation.

Important distinction: PCA preserves variation, not necessarily every piece of information that is useful for classification.

### Step 8: Visualize the principal components

```python
for segment in sorted(y.unique()):
    data = pca_df[pca_df["Customer_Segment"] == segment]
    plt.scatter(data["PC1"], data["PC2"], label=f"Segment {segment}")
```

This code draws one group of points for each segment.

- The horizontal axis represents PC1.
- The vertical axis represents PC2.
- Each point represents one wine sample.
- Different colors distinguish different wine segments.

The purpose is to inspect whether the samples from different segments form distinct groups.

Then:

```python
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Wine Dataset")
plt.legend()
plt.show()
```

These commands label the axes, give the graph a title, display the legend, and show the plot.

**How should you interpret the graph?**

- If different segments form clearly separated groups, the first two components capture variation that helps distinguish those segments visually.
- If groups overlap, the first two components do not completely separate the segments.
- Overlap does not necessarily mean PCA failed. The discarded components may contain additional useful information, and PCA is not designed specifically to maximize class separation.

---

## 4. VIVA CONCEPT CRASH COURSE

This is the theory you should understand before your practical exam.

### A. What is dimensionality reduction?

**Dimensionality reduction means reducing the number of features used to represent a dataset while trying to preserve its important information.**

Imagine that each wine sample is described by 13 measurements.

Analyzing all 13 measurements can be difficult, especially when you want to visualize the data.

Dimensionality reduction creates a smaller representation of the data.

In this practical:

- Original dimensions: 13 features.
- Reduced dimensions: 2 principal components.
- Purpose: Simplify the data and visualize the wine segments.

### B. What is PCA?

**PCA stands for Principal Component Analysis.**

It is an unsupervised dimensionality-reduction technique that transforms correlated or related numerical features into new variables called principal components.

PCA works by finding directions in the data that capture as much variation as possible.

Think of it this way:

You have 13 measurements for every wine sample. PCA combines information from these measurements to create two new variables that capture as much of the overall variation as possible.

Those variables are PC1 and PC2.

### C. How does PCA work?

For your practical, remember these five steps:

1. **Select features:** Remove the target column from the input data.
2. **Standardize:** Bring the numerical features onto comparable scales.
3. **Calculate principal components:** Find directions that capture the greatest variation.
4. **Reduce dimensions:** Retain only the first two components.
5. **Visualize:** Plot the components and color the samples according to their original segments.

If an examiner asks you to explain the practical in one answer, these five steps are sufficient.

### D. What are PC1 and PC2?

| Property | PC1 | PC2 |
|---|---|---|
| Full name | First Principal Component | Second Principal Component |
| Purpose | Captures the maximum possible variation in one direction | Captures the maximum remaining variation in a perpendicular direction |
| Relationship | First component | Orthogonal to PC1 |
| In your practical | X-axis | Y-axis |

Principal components are calculated as combinations of the original standardized features.

Their weights are called **loadings**.

For example, a component can be represented conceptually as:

PC1 = Weight 1 × Alcohol + Weight 2 × Malic Acid + Weight 3 × Ash + ...

The actual calculation uses all the selected features and their calculated weights.

You don't need to memorize the individual weights for this practical.

### E. Why is standardization necessary before PCA?

This is a very likely viva question.

Consider these measurements:

- Alcohol: approximately 14
- Magnesium: approximately 100
- Proline: approximately 1,000

These features use different numerical scales.

PCA identifies directions of maximum variance. Without standardization, features with large numerical variations may dominate the components.

Standardization gives each feature a mean of approximately zero and a standard deviation of approximately one.

**Viva answer:** We standardize the features because PCA is sensitive to differences in scale. Standardization prevents features with large numerical magnitudes from disproportionately influencing the results.

One nuance: Standardization is not mandatory for every possible PCA application, but it is appropriate for this dataset because the measurements have different scales.

### F. What is explained variance?

Explained variance tells us how much of the dataset's variation a principal component captures.

Suppose, for illustration:

- PC1 captures 40%.
- PC2 captures 25%.

Together, the two components capture 65% of the total variation.

This means the two-dimensional representation preserves 65% of the variation measured by PCA.

**Formula in plain text:**

Total explained variance = Explained variance of PC1 + Explained variance of PC2.

The explained variance ratio is available in:

```python
pca.explained_variance_ratio_
```

For your two-component model, this returns two values.

For example, `[0.40, 0.25]` would mean that PC1 explains 40% and PC2 explains 25% of the total variance.

### G. Why use PCA instead of simply selecting two original features?

Suppose you select only Alcohol and Magnesium from the original dataset.

You would discard the other 11 measurements without considering whether they contain useful information.

PCA instead combines information from the original features to construct new components.

This can preserve more overall variation than arbitrarily choosing two original features.

However, PCA is not guaranteed to retain every feature's useful information, and it does not guarantee the best classification accuracy.

### H. Is PCA supervised or unsupervised?

**PCA is an unsupervised learning technique.**

It does not use class labels to calculate the principal components.

In your code:

```python
X_pca = pca.fit_transform(X_scaled)
```

PCA operates only on `X_scaled`.

The target labels are used afterward to color the scatter plot.

This distinction is important because PCA tries to capture overall variation, not specifically the variation that separates wine classes.

### I. Why do we use a scatter plot?

A scatter plot helps us visualize the transformed data.

Each point represents one wine sample.

- PC1 gives the horizontal coordinate.
- PC2 gives the vertical coordinate.
- Color identifies the original wine segment.

If the groups are separated, the two components provide a useful visual representation of differences between the segments.

If the groups overlap, two components may not be sufficient to distinguish every sample.

**Remember:** PCA does not itself classify wine samples. It transforms their features.

### J. Advantages and limitations of PCA

**Advantages**
- Reduces the number of features.
- Can make data easier to visualize.
- Can reduce redundancy when features are correlated.
- May reduce computational requirements for subsequent analysis.

**Limitations**
- Some information is lost when components are discarded.
- Principal components can be difficult to interpret because they combine original features.
- PCA does not use class labels to maximize class separation.
- Standardization is often important when features have different scales.

### K. What output should you expect?

Your dataset has 13 original numerical features, so the following output should be consistent with the provided sample:

```text
Original number of features: 13
Reduced number of features: 2
```

The explained variance percentages and scatter plot depend on the complete dataset.

Your final graph will have:
- PC1 on the horizontal axis.
- PC2 on the vertical axis.
- Separate colors for the available `Customer_Segment` classes.

You should inspect the actual graph and variance output rather than memorizing invented numerical results.

---

## 5. TOP 3 EXPECTED VIVA QUESTIONS

### Q1. What is PCA, and why is it used?

**Answer:** PCA stands for Principal Component Analysis. It is an unsupervised dimensionality-reduction technique that transforms the original features into principal components while preserving as much variation as possible.

### Q2. Why do we standardize the data before applying PCA?

**Answer:** PCA is sensitive to feature scales. Standardization brings the features to comparable scales so that large numerical values do not disproportionately influence the principal components.

### Q3. What are PC1 and PC2?

**Answer:** PC1 captures the maximum possible variation along one direction. PC2 captures the maximum remaining variation in a direction perpendicular to PC1. We use them to represent the original high-dimensional data in two dimensions.

---

## Final Revision: Memorize These 7 Points

| Question | Answer to remember |
|---|---|
| What is PCA? | Principal Component Analysis |
| What is its purpose? | Dimensionality reduction |
| Is PCA supervised? | No, it is unsupervised |
| Why standardize? | To make feature scales comparable |
| What is PC1? | The direction capturing the maximum possible variance |
| What is explained variance? | The proportion of total variation captured by a component |
| Why use a scatter plot? | To visualize the transformed samples and inspect group separation |

**The entire practical in one sentence:**

We load the wine dataset, separate its measurements from the target labels, standardize the measurements, apply PCA to reduce 13 features to 2 principal components, measure the variance retained, and visualize the wine segments using a scatter plot.

**One final exam tip:** Your practical title mentions red and white wine, but your actual dataset uses `Customer_Segment` and appears to represent three wine classes. If the examiner expects red-versus-white classification, confirm which dataset is prescribed. PCA itself does not classify wine; it only transforms the features.