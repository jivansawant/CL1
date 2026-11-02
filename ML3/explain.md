# ML Practical 3: Classification Using Support Vector Machines (SVM)

## 1. CORE TAKEAWAY

**Support Vector Machine (SVM) is a machine learning algorithm that learns to distinguish between classes. In this practical, we use SVM to recognize handwritten digits from 0 to 9 by learning patterns in their pixel values.**

Don't worry about kernels, `C`, `gamma`, or the complicated code you were given. We'll start with the basic idea of classification, build a simple working program, and then understand SVM using everyday examples.

Your original code includes cross-validation, four kernel comparisons, parameter tuning, several performance metrics, and multiple graphs. **You do not need all of that to understand and demonstrate the basic SVM practical.** We'll keep the essential operations and one useful visualization.

## 2. MINIMALIST PYTHON CODE

### Markdown Cell 1 — Complete practical code

We'll use scikit-learn's built-in handwritten digits dataset, so you don't need to download a separate file.

Copy this into one Jupyter Notebook code cell:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load handwritten digit data
digits = load_digits()
X = digits.data
y = digits.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train the SVM model
model = make_pipeline(StandardScaler(), SVC())
model.fit(X_train, y_train)

# Predict the digits
y_pred = model.predict(X_test)

# Display results
print("Number of images:", len(X))
print("Number of features:", X.shape[1])
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Display a sample handwritten digit
plt.imshow(digits.images[0], cmap="gray_r")
plt.title(f"Actual digit: {y[0]}")
plt.axis("off")
plt.show()
```

### What this code does

1. Loads images of handwritten digits from 0 to 9.
2. Converts each image into numerical features.
3. Splits the data into training and testing sets.
4. Standardizes the features.
5. Trains an SVM classifier.
6. Predicts the digits in the test set.
7. Displays the accuracy, confusion matrix, and an example image.

That's the complete basic practical. Let's understand what each step means.

---

## 3. WHY CODE WALKTHROUGH

### Step 1: Import the libraries

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
```

Let's understand only what each library does.

| Library or function | Purpose |
|---|---|
| `matplotlib.pyplot` | Displays a digit image |
| `load_digits` | Loads the built-in handwritten digits dataset |
| `train_test_split` | Divides the data into training and testing sets |
| `StandardScaler` | Standardizes the numerical features |
| `make_pipeline` | Connects preprocessing and model training |
| `SVC` | Creates an SVM classification model |
| `accuracy_score` | Measures the percentage of correct predictions |
| `confusion_matrix` | Shows which digits were predicted correctly or incorrectly |

**For the exam:** The most important function here is `SVC()`, because it creates the Support Vector Machine classifier.

### Step 2: Load the dataset

```python
digits = load_digits()
X = digits.data
y = digits.target
```

The dataset contains images of handwritten digits from 0 to 9.

Each image is an 8 × 8 grid of pixels.

For example, imagine a handwritten digit 7. Some pixels are dark, some are light, and together they form the shape of 7.

The computer doesn't initially understand that shape as a human does. It receives numerical pixel values.

An 8 × 8 image contains:

8 × 8 = 64 pixels.

That is why each image is represented using **64 numerical features**.

Now consider these two variables:

**`X = digits.data`**

Contains the pixel values for every image.

Each row represents one image, and each column represents one pixel position.

**`y = digits.target`**

Contains the correct digit labels: 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9.

For example:

| Pixel 1 | Pixel 2 | Pixel 3 | ... | Pixel 64 | Target |
|---:|---:|---:|---|---:|---:|
| 0 | 0 | 5 | ... | 0 | 7 |
| 0 | 2 | 8 | ... | 0 | 3 |

These are illustrative values, not actual dataset rows.

The model learns the relationship between the pixel values and their corresponding digit labels.

### Step 3: Divide the data into training and testing sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

This line divides our dataset into two parts.

**Training data:** Used to teach the model.

**Testing data:** Used to evaluate the trained model on images it did not use during training.

We use `test_size=0.2`, meaning 20% of the observations are reserved for testing and 80% for training.

Let's understand the other two parameters.

- `random_state=42`: Makes the random split reproducible.
- `stratify=y`: Helps preserve the proportion of each digit class in both sets.

Why is this important?

Suppose we trained the model and then tested it on the exact same images.

It might perform well on those images but struggle with images it hasn't seen.

Keeping test data separate gives us a more meaningful evaluation.

**Viva point:** We use the training set to fit the model and the test set to evaluate its performance on unseen observations.

### Step 4: Create the SVM model

```python
model = make_pipeline(StandardScaler(), SVC())
```

This is the most important line in the practical.

It creates two connected operations:

1. `StandardScaler()` standardizes the features.
2. `SVC()` creates the Support Vector Machine classifier.

The pipeline ensures that the features are standardized before being passed to SVM.

We don't need to memorize a long list of parameters.

For now, remember:

**`SVC()` = Support Vector Classification.**

When we use `SVC()` without specifying a kernel, scikit-learn uses the RBF kernel by default.

We'll explain what a kernel means in the next section.

### Step 5: Train the model

```python
model.fit(X_train, y_train)
```

This is where the model learns.

It receives:
- The pixel values from the training images.
- The correct digit labels for those images.

It then learns decision boundaries that help distinguish between the digit classes.

For example, it tries to learn patterns that help distinguish a handwritten 3 from a handwritten 8.

The model is not given a rule such as "if the image has two loops, predict 8." Instead, it learns from the numerical examples.

**Remember: `fit()` means train the model.**

### Step 6: Predict the digits

```python
y_pred = model.predict(X_test)
```

After training, we give the model the test images.

The model predicts a digit for each image.

For example:

| Actual digit | Predicted digit |
|---:|---:|
| 3 | 3 |
| 8 | 8 |
| 5 | 6 |
| 1 | 1 |

These are illustrative examples.

The first, second, and fourth predictions are correct. The third prediction is incorrect.

We use the predictions to calculate the model's performance.

**Remember: `predict()` means make predictions using the trained model.**

### Step 7: Calculate accuracy

```python
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
```

Accuracy measures how many predictions are correct out of all predictions.

The formula in plain text is:

**Accuracy = Number of correct predictions / Total number of predictions**

Suppose the model correctly identifies 170 out of 180 test images.

Accuracy = 170 / 180 = 0.9444, or approximately 94.44%.

This is only an example; your actual result will be produced by the code.

The function `round(..., 4)` displays the result to four decimal places.

### Step 8: Display the confusion matrix

```python
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

Accuracy gives us one overall number. A confusion matrix provides more detail.

It shows which digits the model predicted correctly and which digits it confused with other digits.

For example, the model might sometimes mistake a handwritten 9 for a 4.

In a confusion matrix:
- Rows represent the actual classes.
- Columns represent the predicted classes.
- Values along the main diagonal represent correct predictions.
- Values outside the main diagonal represent incorrect predictions.

Because our dataset contains ten classes, the confusion matrix has 10 rows and 10 columns.

For example, the cell in row 3 and column 3 represents the number of actual 3s correctly predicted as 3.

The cell in row 3 and column 8 represents the number of actual 3s incorrectly predicted as 8.

**For your practical, the confusion matrix helps demonstrate that the model handles all ten digit classes, not just one class.**

### Step 9: Display an example image

```python
plt.imshow(digits.images[0], cmap="gray_r")
plt.title(f"Actual digit: {y[0]}")
plt.axis("off")
plt.show()
```

This displays one of the handwritten digit images.

- `digits.images[0]` selects the first image.
- `cmap="gray_r"` displays it using reversed grayscale.
- `y[0]` retrieves its correct label.
- `plt.axis("off")` hides the coordinate axes.
- `plt.show()` displays the image.

Why do we need this?

It helps us connect the numerical data used by the model with the actual handwritten digits we want to recognize.

This visualization is optional for the algorithm itself, but useful for understanding the practical.

---

## 4. VIVA CONCEPT CRASH COURSE

Now let's learn SVM from the beginning. You don't need to study every mathematical detail to understand this practical.

### A. What is classification?

Classification is a type of supervised machine learning in which a model learns to assign observations to categories.

For example:

- Email → Spam or Not Spam.
- Image → Cat or Dog.
- Medical data → Class 0 or Class 1.
- Handwritten image → Digit 0, 1, 2, ..., 9.

In our practical, the classes are the ten numerical digits.

**Supervised learning** means that the training data includes the correct answers.

Our dataset contains both the pixel values (`X`) and the correct digit labels (`y`), so it is suitable for supervised learning.

### B. What is SVM?

SVM stands for **Support Vector Machine**.

It is a supervised machine learning algorithm used for classification and, in another form, regression.

The goal of SVM classification is to find a decision boundary that separates different classes.

Imagine that you have red and blue points drawn on a sheet of paper.

You want to draw a line that separates the red points from the blue points.

Many lines might separate them, but SVM looks for a boundary that maximizes the margin between the classes.

The **margin** is the distance between the decision boundary and the nearest training points from the different classes.

SVM aims to find a boundary with a large margin because this can help the model generalize to unseen data.

### C. What are support vectors?

Support vectors are the training observations that determine or constrain the position of the decision boundary.

Imagine a line separating two groups of points.

The points closest to that line are particularly important because they help determine where the line should be placed.

Those important points are called **support vectors**.

Why are they important?

SVM uses them to establish its decision boundary. Moving or changing these points can affect the boundary more directly than changing points far away from it.

**Viva answer:** Support vectors are the training samples closest to the decision boundary that help determine its position.

### D. What is a decision boundary?

A decision boundary is the boundary the model uses to distinguish between classes.

For two classes, it might be a straight line in a two-dimensional example.

With more features, it can be a higher-dimensional boundary.

In our practical, each image has 64 features, so the model works in a 64-dimensional feature space.

We cannot directly visualize that entire space on a normal 2D graph, but the model can still learn a decision boundary in it.

### E. What is a kernel in SVM?

This is the part of your original code that looks complicated, so let's simplify it.

**A kernel is a function that helps SVM model relationships between data points and construct decision boundaries.**

Think of kernels as different ways of allowing SVM to separate classes.

Some datasets can be separated well with a straight boundary. Others require a more flexible boundary.

A kernel lets SVM handle different types of relationships without explicitly requiring you to construct new features by hand.

Your original code compares four kernels.

You only need to understand their basic differences:

| Kernel | Simple explanation | When it can be useful |
|---|---|---|
| Linear | Uses a linear decision boundary | When classes can be separated reasonably well by a linear boundary |
| Polynomial | Can model curved relationships using polynomial interactions | When more complex polynomial-shaped boundaries are useful |
| RBF | Can model flexible, nonlinear boundaries | When the relationship between classes is more complicated |
| Sigmoid | Uses a sigmoid-shaped kernel function | In some problems where its resulting decision boundary is suitable |

**For this practical, you can start with the default RBF kernel.** You do not need to compare all four kernels to demonstrate basic SVM classification.

#### What is the RBF kernel?

RBF stands for **Radial Basis Function**.

It is a commonly used kernel that allows SVM to model nonlinear relationships.

For example, imagine that one class forms a circle surrounded by another class.

A single straight line cannot separate the inner circle from the surrounding group.

An RBF-based SVM can learn a more flexible boundary that may separate them.

The handwritten digits dataset contains patterns that may not be separable using a single straight boundary, which is why RBF is a reasonable starting choice.

You don't need to memorize the mathematical formula for the RBF kernel for this simplified practical.

### F. What are `C` and `gamma`?

Your original code contains parameters such as:

```python
SVC(kernel="rbf", C=1.0, gamma="scale")
```

You don't need to add these parameters to the simplified code, but you should know what they mean.

**1. What is `C`?**

`C` controls the trade-off between a wider margin and penalizing training errors.

- A smaller `C` allows more training errors in exchange for a stronger preference for a wider margin.
- A larger `C` penalizes training errors more strongly, which can produce a more complex boundary.

A very large `C` can increase the risk of overfitting.

**2. What is `gamma`?**

`gamma` controls how far the influence of individual training samples extends when using the RBF kernel.

- Smaller `gamma`: Each sample has a broader influence, often producing a smoother boundary.
- Larger `gamma`: Each sample has a more localized influence, potentially producing a more complicated boundary.

Very large values can lead to overfitting.

**What does `gamma="scale"` mean?**

It tells scikit-learn to calculate a default gamma value from the training data rather than requiring you to choose a number manually.

**For your exam:** Start with `SVC()` and let scikit-learn use its default settings. Learn about `C` and `gamma` conceptually rather than trying to tune them before you understand SVM itself.

### G. Why do we use `StandardScaler()`?

Standardization transforms numerical features so that they have comparable scales.

In this dataset, pixel values are already bounded and have a common interpretation, but their distributions can differ across pixel positions.

SVM is sensitive to feature scales, particularly when using distance-sensitive kernels such as RBF.

Standardization helps prevent differences in feature scale from disproportionately influencing the model.

The formula is:

Standardized value = (Original value − Mean) / Standard deviation

Our code uses:

```python
make_pipeline(StandardScaler(), SVC())
```

This automatically standardizes the training data before fitting SVM and applies the same learned transformation to test data.

The scaler is fitted only on the training data, which avoids using information from the test set during preprocessing.

### H. What is the difference between training and testing?

Training is when the model learns patterns from examples.

Testing is when we evaluate how well the trained model predicts observations that were not used to fit it.

We use an 80:20 split:

- 80% for training.
- 20% for testing.

Why not evaluate the model on the training data?

Because the model has already seen those observations. Its performance on training data can be overly optimistic.

The testing data provides a more meaningful estimate of performance on unseen examples.

### I. What is accuracy?

Accuracy measures the fraction of predictions that match the correct labels.

Formula:

Accuracy = Correct predictions / Total predictions

For example, if a model makes 175 correct predictions out of 180 test images:

Accuracy = 175 / 180 = 0.9722, or 97.22%.

That is an illustrative calculation, not your actual result.

Higher accuracy means more correct predictions on the evaluated dataset. However, accuracy alone does not explain which classes the model confuses.

### J. What is the difference between SVM and KNN?

This is another useful comparison for classification practicals.

| SVM | KNN |
|---|---|
| Learns a decision boundary during training | Stores training examples and uses nearby examples when predicting |
| Uses support vectors to define its boundary | Uses the nearest training examples |
| Can use kernels to model nonlinear relationships | Uses a distance measure to find neighbors |
| Requires choices such as the kernel and `C` | Requires a choice of the number of neighbors, `k` |

The main distinction is that SVM learns a separating boundary, while KNN classifies a new observation using the labels of its nearest neighbors.

### K. What are the limitations of SVM?

1. SVM can require substantial training time on very large datasets.
2. Choosing a suitable kernel and parameters can be difficult.
3. The model may be harder to interpret than a simple decision tree.
4. Performance depends on the data and the preprocessing used.
5. SVM does not automatically guarantee high accuracy on every classification problem.

For this practical, the built-in digits dataset is small enough that a basic SVM model is manageable.

---

## 5. TOP 3 EXPECTED VIVA QUESTIONS

### Q1. What is SVM, and how does it work?

**Answer:** SVM stands for Support Vector Machine. It is a supervised learning algorithm that classifies data by finding a decision boundary that separates classes while aiming to maximize the margin between them.

### Q2. What is a kernel in SVM? Name some types.

**Answer:** A kernel helps SVM model relationships between data points and construct decision boundaries. Common types are Linear, Polynomial, RBF, and Sigmoid.

### Q3. What are support vectors?

**Answer:** Support vectors are the training samples closest to the decision boundary that help determine its position.

---

## Last-Minute Revision Sheet

Before your practical, remember these points:

| Term | What to remember |
|---|---|
| Classification | Predicting a category or class |
| SVM | Support Vector Machine |
| Support vectors | Important training points that determine the boundary |
| Decision boundary | Separates different classes |
| Margin | Distance between the boundary and the nearest points |
| Kernel | Helps SVM model different types of relationships |
| RBF | A kernel that can model nonlinear boundaries |
| `C` | Controls the trade-off between margin width and training errors |
| `gamma` | Controls the influence range of individual samples for RBF |
| `fit()` | Trains the model |
| `predict()` | Predicts the classes |
| Accuracy | Fraction of correct predictions |
| Confusion matrix | Shows correct and incorrect predictions for each class |

### How to explain your practical to the examiner

If the examiner asks, "Explain your practical," you can say:

"We use the handwritten digits dataset, where each 8 × 8 image is represented by 64 numerical pixel features. We split the dataset into training and testing sets, standardize the features, and train an SVM classifier. The trained model predicts digits from 0 to 9. Finally, we evaluate its performance using accuracy and a confusion matrix."

### One important correction to your original code

Your original code uses `RANDOM_STATE` here:

```python
train_test_split(
    X, y, test_size=0.20,
    stratify=y, random_state=RANDOM_STATE
)
```

But `RANDOM_STATE` is never defined in the code you provided. Running that cell as written would produce a `NameError`.

Our simplified version uses `random_state=42` directly, so this particular error is avoided.

**My recommendation:** For now, practise only the simplified code and understand what `X`, `y`, `fit()`, `predict()`, accuracy, and the confusion matrix mean. Once those concepts are clear, learn the four kernel types and the basic meaning of `C` and `gamma`. You do not need to master parameter tuning before you can understand and perform this practical.