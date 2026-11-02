# Practical 4: K-Means Clustering on the Iris Dataset

## 1. CORE TAKEAWAY

**K-Means groups similar flowers into clusters without using their species labels. The elbow method helps us choose the number of clusters (`k`) by finding the point where increasing `k` stops reducing the clustering error as dramatically.**

Your original code is much more complicated than necessary for this practical. It includes correlation heatmaps, pair plots, several clustering scores, automatic elbow detection, centroid analysis, and species comparisons. **You do not need all of that to demonstrate K-Means and the elbow method.**

We'll use the **built-in Iris dataset from scikit-learn**, so you don't need `Iris.csv`.

## 2. MINIMALIST PYTHON CODE

Run this as one complete practical in Jupyter Notebook.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load built-in Iris dataset
iris = load_iris()
X = iris.data

print("Dataset shape:", X.shape)
print("Number of flowers:", len(X))
print("Features:", iris.feature_names)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

# Plot elbow graph
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.show()

# Final K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

print("Cluster sizes:")
for i in range(3):
    print("Cluster", i, ":", sum(clusters == i), "flowers")

# Visualize clusters using petal measurements
plt.scatter(X[:, 2], X[:, 3], c=clusters)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-Means Clustering of Iris Dataset")
plt.show()
```

**Expected result:** The dataset contains 150 flowers and four features. The elbow graph should show a noticeable bend around `k = 3`, although the exact choice of elbow can be subjective. The final model groups the flowers into three clusters.

---

## 3. WHY CODE WALKTHROUGH

### Step 1: Load the dataset

```python
iris = load_iris()
X = iris.data
```

- `load_iris()` loads the dataset already available in scikit-learn.
- `iris.data` contains the measurements of the flowers.
- `X` contains four features for each flower:
  1. Sepal length
  2. Sepal width
  3. Petal length
  4. Petal width

There are 150 flowers, each with four measurements.

Therefore, the shape of `X` is `(150, 4)`.

**Important:** The built-in dataset also contains the actual species labels, but we don't use them to train K-Means. This is an unsupervised learning algorithm.

### Step 2: Standardize the data

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The four features have different ranges. Standardization puts them on a comparable scale so features with larger numerical values do not dominate the distance calculations simply because of their units or magnitude.

You don't need to memorize the formula. Remember:

**StandardScaler makes the features comparable before K-Means calculates distances.**

### Step 3: Apply the elbow method

```python
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)
```

Let's break it down.

**What is `k`?**

`k` is the number of clusters we want K-Means to create.

- `k = 1`: one cluster containing all flowers.
- `k = 2`: two clusters.
- `k = 3`: three clusters.
- And so on, up to ten clusters.

**What is WCSS?**

WCSS stands for *Within-Cluster Sum of Squares*. It measures how far the data points are from their assigned cluster centres, using squared distances.

- Lower WCSS means the points are closer to their cluster centres.
- Increasing the number of clusters generally decreases WCSS.
- If we keep increasing the number of clusters, however, we eventually get diminishing benefits.

`model.inertia_` gives us the WCSS calculated by scikit-learn.

**What does `wcss.append(model.inertia_)` do?**

It saves the WCSS value for each value of `k`, so we can plot them afterward.

### Step 4: Plot the elbow graph

```python
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.xticks(range(1, 11))
plt.show()
```

Read the graph as follows:

- **X-axis:** Number of clusters (`k`).
- **Y-axis:** WCSS.
- **Elbow point:** The point where the curve starts flattening noticeably.

For example, WCSS may decrease substantially when moving from one to two clusters and from two to three clusters. After that, the improvements may become smaller.

That bend suggests a reasonable number of clusters.

For this Iris practical, **three clusters is a reasonable choice**, though the elbow method is a guide rather than a guarantee that three is the objectively perfect answer.

### Step 5: Train the final K-Means model

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
```

Here:

- `n_clusters=3` tells K-Means to create three clusters.
- `fit_predict()` fits the model and assigns a cluster number to each flower.
- `clusters` stores the predicted cluster number for every flower.

For example, a cluster assignment could look like this:

```text
[1, 1, 1, 0, 0, 2, 2, ...]
```

These are **cluster IDs, not species names**. Cluster `0` does not automatically mean Iris-setosa, and cluster IDs can change between different model configurations.

### Step 6: Print cluster sizes

```python
for i in range(3):
    print("Cluster", i, ":", sum(clusters == i), "flowers")
```

This counts the flowers assigned to each cluster.

The sizes do not have to be equal. K-Means assigns flowers according to their measurements and the learned cluster centres.

### Step 7: Visualize the clusters

```python
plt.scatter(X[:, 2], X[:, 3], c=clusters)
```

The plot uses two features:

- `X[:, 2]`: petal length.
- `X[:, 3]`: petal width.
- `c=clusters`: colours points according to their assigned cluster.

We use the original measurements for easier interpretation. The clustering itself was performed using all four standardized features.

---

## 4. VIVA CONCEPT CRASH COURSE

### Q1. What is clustering?

Clustering is an **unsupervised machine learning technique** that groups similar data points together.

Example: Grouping flowers based on their measurements without telling the algorithm which species each flower belongs to.

### Q2. What is K-Means?

K-Means is a clustering algorithm that divides data into `k` clusters.

Basic working:

1. Choose the number of clusters, `k`.
2. Initialize `k` cluster centres.
3. Assign each data point to its nearest centre.
4. Recalculate the centres based on their assigned points.
5. Repeat until the assignments or centres stop changing significantly, or the iteration limit is reached.

### Q3. What is the elbow method?

The elbow method helps select a suitable number of clusters by plotting WCSS against different values of `k`.

We look for a bend in the graph where adding more clusters gives diminishing reductions in WCSS.

### Q4. What is a centroid?

A centroid is the centre of a cluster, calculated as the mean of the points assigned to it.

K-Means repeatedly updates the centroids as it forms clusters.

### Q5. Why do we use StandardScaler?

Because features can have different numerical scales. StandardScaler puts them on a comparable scale so that large-valued features do not dominate distance calculations just because their values are larger.

### Q6. What is WCSS?

WCSS means *Within-Cluster Sum of Squares*. It measures the sum of squared distances between data points and their assigned cluster centres.

Lower WCSS generally indicates tighter clusters, but WCSS alone cannot tell us the ideal number of clusters because it usually decreases as `k` increases.

### Q7. Why is K-Means unsupervised learning?

Because K-Means does not require known species labels to create its clusters. It groups data based on feature values.

### Q8. What does `n_clusters=3` mean?

It instructs K-Means to create three clusters.

### Q9. What does `random_state=42` do?

It makes the random initialization reproducible, so repeated runs with the same settings can produce consistent results.

### Q10. What does `n_init=10` mean?

K-Means runs with ten different initial centroid configurations and retains the result with the lowest WCSS. This reduces the chance of getting a poor result from an unlucky initialization.

---

## 5. TOP 3 EXPECTED VIVA QUESTIONS

**1. What is the elbow method?**

It is a method for selecting the number of clusters by plotting WCSS against different values of `k` and looking for the point where the curve starts to flatten.

**2. Why is K-Means called an unsupervised algorithm?**

Because it forms clusters based on the data's features without requiring predefined class labels.

**3. Why do we use StandardScaler before K-Means?**

To make features comparable in scale, preventing features with larger numerical values from dominating the distance calculations.

---

### Last-minute exam revision

Memorize this sequence:

**Load Iris → Standardize features → Calculate WCSS for different `k` values → Plot the elbow graph → Choose `k = 3` → Train K-Means → Visualize clusters.**

One important distinction: the elbow method suggests a suitable value of `k`; it does not prove that the clusters perfectly match the three actual Iris species.