# Practical 7: Random Forest Classifier

## 1. CORE TAKEAWAY

**Random Forest is a machine learning algorithm that uses multiple decision trees to make a prediction.** In this practical, we use the Car Evaluation dataset to predict the car's acceptability (`unacc`, `acc`, `good`, or `vgood`) based on its features.

**Small correction:** Your dataset's target column is `decision`, not `safety`. The `safety` column is an input feature. So, the code below predicts the **car's decision/classification**, using safety as one of the inputs.

## 2. MINIMALIST PYTHON CODE

Save the complete dataset as `car.csv` in your working folder. The file should contain all the rows, not just the sample rows in your message.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("car.csv")

# Convert text values into numbers
encoder = LabelEncoder()

for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# Separate inputs and output
X = df.drop("decision", axis=1)
y = df["decision"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predicted classes:", y_pred[:10])
```

**Expected output:** The program prints the model's accuracy and the first ten predicted class labels in encoded numeric form. The exact accuracy depends on the dataset and split.

## 3. CODE WALKTHROUGH

### Step 1: Import libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

- `pandas`: reads the dataset.
- `train_test_split`: divides data into training and testing sets.
- `LabelEncoder`: converts text categories into numbers.
- `RandomForestClassifier`: creates the Random Forest model.
- `accuracy_score`: measures prediction accuracy.

### Step 2: Load the dataset

```python
df = pd.read_csv("car.csv")
```

This reads the CSV file into a DataFrame called `df`.

Your dataset has six input columns:

- `buying price`
- `maintenance cost`
- `number of persons`
- `lug_boot`
- `safety`
- `decision` — the target we want to predict.

### Step 3: Convert text into numbers

```python
encoder = LabelEncoder()

for col in df.columns:
    df[col] = encoder.fit_transform(df[col])
```

Machine learning models generally require numerical inputs. Your dataset contains text values such as `vhigh`, `small`, `low`, and `unacc`.

`LabelEncoder` assigns a number to each category.

For example, categories such as `low`, `med`, and `high` might be encoded as `1`, `2`, and `0`, respectively, depending on the categories present and their ordering.

**Important limitation:** This is a simple encoding method, but it can introduce misleading numerical relationships between categories. One-hot encoding is often preferable for categorical input features. For a basic practical, however, this version demonstrates the main Random Forest workflow.

### Step 4: Separate inputs and output

```python
X = df.drop("decision", axis=1)
y = df["decision"]
```

- `X` contains the six input columns except `decision`.
- `y` contains the answer the model must learn to predict.

In machine learning, these are called the **features (`X`)** and the **target (`y`)**.

### Step 5: Split the dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

This divides the dataset into:

- 80% training data.
- 20% testing data.

The model learns from the training data and is evaluated on the testing data.

`random_state=42` makes the split reproducible.

### Step 6: Create the Random Forest model

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

Random Forest builds multiple decision trees and combines their predictions.

- `n_estimators=100`: builds 100 trees.
- `random_state=42`: makes the random process reproducible.

**What is a decision tree?**

A decision tree makes predictions by applying rules to features. For example, it may use safety, buying price, and passenger capacity to help classify a car.

**Why use a forest?**

A single tree can make unreliable predictions. Combining many trees often produces more robust predictions.

### Step 7: Train the model

```python
model.fit(X_train, y_train)
```

The model learns patterns connecting the car features to the target classes.

### Step 8: Make predictions

```python
y_pred = model.predict(X_test)
```

The trained model predicts the class of each car in the test set.

### Step 9: Calculate accuracy

```python
print("Accuracy:", accuracy_score(y_test, y_pred))
```

Accuracy measures how many predictions were correct.

**Formula:**

Accuracy = Number of correct predictions / Total predictions

For example, if 300 out of 346 predictions are correct:

Accuracy = 300 / 346 = approximately 86.7%.

This is only an example, not the expected result of your code.

---

## 4. VIVA CONCEPTS

**Q1. What is Random Forest?**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

**Q2. What is a decision tree?**

A decision tree predicts an outcome by following rules based on input features.

**Q3. What is `n_estimators`?**

It specifies the number of trees in the Random Forest.

**Q4. What are features and target?**

Features are the input variables used for prediction. The target is the variable the model predicts.

**Q5. Why split the dataset?**

To train the model on one portion of the data and evaluate its performance on unseen data.

**Q6. What is accuracy?**

Accuracy is the proportion of predictions that match the correct labels.

## 5. TOP 3 EXPECTED VIVA QUESTIONS

**1. Why is Random Forest called an ensemble algorithm?**

Because it combines predictions from multiple decision trees.

**2. What is the purpose of `fit()` and `predict()`?**

`fit()` trains the model. `predict()` generates predictions using the trained model.

**3. What is the difference between training and testing data?**

Training data is used to teach the model. Testing data evaluates how well the model predicts unseen examples.

---

### Remember this practical in one line

**Load data → Encode categories → Separate X and y → Split data → Train Random Forest → Predict → Calculate accuracy.**

One final detail: the dataset you pasted contains only a small sample. Use the **complete Car Evaluation dataset** for meaningful training and testing results.