# Practical 6: Ensemble Learning on the Iris Dataset

Let's make this much easier than the code you provided. You do **not** need to understand every line of that long program to learn this practical.

We'll start with the basic idea, learn three important algorithms, and then write a short program that trains them and compares their performance.

## 1. CORE TAKEAWAY

**Ensemble learning combines multiple models or uses multiple learners together to make predictions.** In this practical, we will use AdaBoost, Gradient Boosting (GBM), and XGBoost to classify Iris flowers, then compare their results using accuracy, precision, recall, and F1-score.

## 2. UNDERSTAND ENSEMBLE LEARNING FROM ZERO

### First, what problem are we solving?

The Iris dataset contains measurements of flowers, such as:

- Sepal length
- Sepal width
- Petal length
- Petal width

Our task is to predict which of these three species a flower belongs to:

1. Iris-setosa
2. Iris-versicolor
3. Iris-virginica

This is a **classification problem** because we are predicting a category.

Imagine we ask one student to identify a flower. They might make a mistake. We could ask several students and combine their opinions to make a decision.

The idea of combining models is the basic intuition behind ensemble learning.

### What is ensemble learning?

Ensemble learning is a machine learning technique that combines multiple learners to produce predictions.

For example, imagine three models predict the species of a flower:

| Model | Prediction |
|---|---|
| Model 1 | Setosa |
| Model 2 | Versicolor |
| Model 3 | Setosa |

With majority voting, the final prediction is **Setosa**, because two out of three models predicted it.

That is the basic idea of voting.

However, your practical also includes three specific boosting algorithms. These are related to ensemble learning, but they work differently from simple voting.

## 3. UNDERSTAND THE THREE ALGORITHMS

Before the code, understand the difference between these three methods.

### A. AdaBoost — Adaptive Boosting

Imagine a student answering questions in a test.

- Initially, the student gets some questions wrong.
- The teacher pays more attention to the questions the student struggled with.
- The student tries to improve on those mistakes.
- The process repeats, combining the results of multiple learners.

AdaBoost works with a similar idea: **it builds a sequence of weak learners, giving more attention to examples that earlier learners classified incorrectly.**

A weak learner is a model that performs only a little better than a simple baseline on the task.

In our program, we use a small decision tree as the weak learner.

**Remember:** AdaBoost focuses more on examples that earlier learners get wrong.

### B. Gradient Boosting (GBM)

Gradient Boosting also builds models sequentially.

Imagine you make a prediction, compare it with the correct answer, and calculate how wrong it was. Then you build another model to help correct the remaining errors.

The process repeats, gradually improving the combined model.

**Remember:** Gradient Boosting adds models sequentially to reduce the errors of the combined model, using gradient-based optimization.

### C. XGBoost

XGBoost stands for **Extreme Gradient Boosting**.

It is an optimized implementation of gradient-boosted trees. It includes techniques to improve training efficiency and control model complexity.

For this practical, you can think of it as a more optimized gradient-boosting method.

**Remember:** XGBoost is an efficient, regularized implementation of gradient boosting.

### Quick comparison

| Algorithm | Main idea |
|---|---|
| AdaBoost | Gives more attention to examples earlier learners get wrong |
| Gradient Boosting (GBM) | Builds models sequentially to reduce remaining errors |
| XGBoost | An optimized implementation of gradient-boosted trees |

**Important difference:** AdaBoost, GBM, and XGBoost build ensembles sequentially. Voting combines predictions from separately trained models. Your assignment covers both approaches.

---

## 4. MINIMALIST PYTHON CODE

We'll use the **built-in Iris dataset**, so you don't need `iris.csv`.

The code performs these tasks:

1. Loads the data.
2. Splits it into training and testing data.
3. Trains AdaBoost, GBM, and XGBoost.
4. Compares their performance.
5. Uses hard voting and soft voting.

### Complete code

Run this in one Jupyter Notebook cell.

If XGBoost is not installed, run `pip install xgboost` in a separate notebook cell first.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create the three models
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

gbm = GradientBoostingClassifier(
    n_estimators=100, random_state=42
)

xgb = XGBClassifier(
    n_estimators=100,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=2
)

models = {
    "AdaBoost": ada,
    "GBM": gbm,
    "XGBoost": xgb
}

# 4. Train and evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, average="macro", zero_division=0),
        recall_score(y_test, pred, average="macro", zero_division=0),
        f1_score(y_test, pred, average="macro", zero_division=0)
    ])

# 5. Apply hard and soft voting
voting_models = {
    "Hard Voting": VotingClassifier(
        estimators=[("ada", ada), ("gbm", gbm), ("xgb", xgb)],
        voting="hard"
    ),
    "Soft Voting": VotingClassifier(
        estimators=[("ada", ada), ("gbm", gbm), ("xgb", xgb)],
        voting="soft"
    )
}

for name, model in voting_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, average="macro", zero_division=0),
        recall_score(y_test, pred, average="macro", zero_division=0),
        f1_score(y_test, pred, average="macro", zero_division=0)
    ])

# 6. Display the comparison
result_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-score"
])

print(result_df.round(3))
```

**What should you expect?** The program prints a comparison table with five rows: the three boosting models, hard voting, and soft voting. The scores depend on the trained models and test split. There is no guarantee that voting will outperform every individual model.

---

## 5. CODE WALKTHROUGH — EXPLAINED FROM THE BEGINNING

Let's understand why each part of the code is necessary.

### Step 1: Import the libraries

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

A **library** is a collection of ready-made code that we can use in our program.

You don't have to write the algorithms yourself because these libraries already provide their implementations.

| Import | Purpose |
|---|---|
| `pandas` | Displays the results as a table |
| `load_iris` | Loads the Iris dataset |
| `train_test_split` | Divides data into training and testing sets |
| `DecisionTreeClassifier` | Creates a decision tree |
| `AdaBoostClassifier` | Implements AdaBoost |
| `GradientBoostingClassifier` | Implements Gradient Boosting |
| `XGBClassifier` | Implements XGBoost classification |
| `VotingClassifier` | Combines models using voting |
| Evaluation functions | Calculate performance scores |

**Important:** XGBoost is an external library. If Python reports `ModuleNotFoundError: No module named 'xgboost'`, install it using `pip install xgboost`.

### Step 2: Load the dataset

```python
iris = load_iris()
X = iris.data
y = iris.target
```

Remember these two variables:

**`X` = input features**

The flower measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

**`y` = correct answers**

The species labels, encoded as numbers:

- `0` = Setosa
- `1` = Versicolor
- `2` = Virginica

The algorithm uses `X` to learn how to predict `y`.

This is supervised learning because we provide the correct labels during training.

### Step 3: Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Imagine you have 100 questions to practise.

You practise using 80 questions and then test yourself on the remaining 20.

Machine learning uses a similar approach.

For the Iris dataset:

- Total flowers: 150
- Training flowers: 120
- Testing flowers: 30

**Why do we split the data?**

We want to check how well the models predict examples that were not used to train them.

If we test the model on its training data, its performance may appear better than its ability to generalize to unseen examples.

Understanding these parameters:

- `test_size=0.2`: Reserves 20% of the data for testing.
- `random_state=42`: Makes the split reproducible.
- `stratify=y`: Helps preserve the proportions of the three species in both sets.

### Step 4: Create the AdaBoost model

```python
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
```

Let's understand each parameter.

**`DecisionTreeClassifier(max_depth=1)`**

This creates a very small decision tree, called a decision stump.

A decision tree makes predictions using rules based on feature values.

For example, it might learn a rule involving petal length to distinguish flower species.

A depth of one limits the tree to a very simple decision structure.

**`n_estimators=50`**

This tells AdaBoost to use up to 50 sequential learners.

Think of it as allowing the algorithm to build a sequence of small decision trees.

**`random_state=42`**

This helps make the model's random choices reproducible.

### Step 5: Create the Gradient Boosting model

```python
gbm = GradientBoostingClassifier(
    n_estimators=100, random_state=42
)
```

Here:

- `GradientBoostingClassifier` creates a Gradient Boosting classification model.
- `n_estimators=100` specifies 100 boosting stages.
- `random_state=42` helps make results reproducible.

Unlike the AdaBoost configuration, we haven't explicitly limited the depth of the trees in this model. It uses the estimator's default tree settings.

### Step 6: Create the XGBoost model

```python
xgb = XGBClassifier(
    n_estimators=100,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=2
)
```

Let's understand the parameters that matter most.

**`n_estimators=100`**

Specifies 100 boosting rounds.

**`objective="multi:softprob"`**

Tells XGBoost to handle a multiclass classification problem and produce class probabilities.

There are three flower species, so this is a multiclass problem.

**`eval_metric="mlogloss"`**

Specifies the multiclass log-loss metric used for evaluation by XGBoost during training. Lower log loss generally indicates better probability predictions.

**`n_jobs=2`**

Allows the model to use two CPU threads for supported operations.

You don't need to memorize every parameter. For the exam, understand that XGBoost builds a boosted tree model for classification.

### Step 7: Store the models together

```python
models = {
    "AdaBoost": ada,
    "GBM": gbm,
    "XGBoost": xgb
}
```

This is a Python dictionary.

It stores each model with a readable name.

Why is this useful?

Instead of writing separate training code three times, we can use a loop to train and evaluate every model.

### Step 8: Train and evaluate the models

```python
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
```

**What does `fit()` do?**

It trains the model using the training data.

For example:

```python
ada.fit(X_train, y_train)
```

means the AdaBoost model learns from the flower measurements and their correct species labels.

**What does `predict()` do?**

It uses the trained model to predict the species of the test flowers.

```python
pred = model.predict(X_test)
```

The variable `pred` contains the model's predictions.

**What does the loop do?**

It repeats the same process for AdaBoost, GBM, and XGBoost.

This is a convenient way to compare multiple models fairly using the same training and test split.

### Step 9: Understand the evaluation measures

The program uses four evaluation measures.

#### A. Accuracy

Accuracy tells us the proportion of predictions that were correct.

Formula in plain text:

**Accuracy = Correct predictions / Total predictions**

For example, if a model correctly classifies 27 out of 30 test flowers:

Accuracy = 27 / 30 = 0.90, or 90%.

Higher accuracy is generally better.

#### B. Precision

Precision tells us how often the model's positive predictions for a class are correct.

For a particular class:

**Precision = True Positives / (True Positives + False Positives)**

In simple words, it answers:

"When the model predicts this species, how often is it correct?"

#### C. Recall

Recall tells us how many of the actual examples of a class the model successfully identifies.

For a particular class:

**Recall = True Positives / (True Positives + False Negatives)**

In simple words, it answers:

"Of all the flowers that really belong to this species, how many did the model identify correctly?"

#### D. F1-score

F1-score combines precision and recall into a single measure using their harmonic mean.

It is useful when you want to consider both missed examples and incorrect positive predictions.

For this practical, we use:

```python
average="macro"
```

This calculates the metric separately for each species and then takes the unweighted average across the three species.

**Remember:**

| Metric | Main question |
|---|---|
| Accuracy | How many predictions were correct overall? |
| Precision | How reliable were the predictions for a class? |
| Recall | How many actual examples of a class were found? |
| F1-score | How well were precision and recall balanced? |

### Step 10: Understand hard voting and soft voting

This is another important part of your assignment.

We have trained three different models. Now we'll combine their predictions.

#### Hard voting

```python
VotingClassifier(
    estimators=[("ada", ada), ("gbm", gbm), ("xgb", xgb)],
    voting="hard"
)
```

Hard voting means each model votes for one predicted class.

Imagine the predictions are:

| Model | Predicted species |
|---|---|
| AdaBoost | Setosa |
| GBM | Versicolor |
| XGBoost | Setosa |

Setosa receives two votes, while Versicolor receives one.

Therefore, the final prediction is **Setosa**.

This is majority voting.

#### Soft voting

```python
VotingClassifier(
    estimators=[("ada", ada), ("gbm", gbm), ("xgb", xgb)],
    voting="soft"
)
```

Soft voting combines the class probabilities predicted by each model and selects the class with the highest average probability.

Imagine the models produce these probabilities for Setosa:

| Model | Probability of Setosa |
|---|---:|
| AdaBoost | 0.70 |
| GBM | 0.60 |
| XGBoost | 0.80 |

Average probability for Setosa = (0.70 + 0.60 + 0.80) / 3 = 0.70.

Soft voting calculates these averages for all three classes and chooses the class with the highest average.

**Main difference:**

- Hard voting uses the predicted class from each model.
- Soft voting combines the predicted probabilities.

Soft voting requires each model to provide class probabilities, which the three classifiers in our program support.

### Step 11: Display the comparison table

```python
result_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-score"
])

print(result_df.round(3))
```

This creates a table containing the five models and their four evaluation measures.

The results help us compare the performance of:

1. AdaBoost
2. GBM
3. XGBoost
4. Hard Voting
5. Soft Voting

A higher score is generally better for these four metrics. However, because the Iris dataset is small and relatively easy to classify, multiple models may achieve similar scores.

**Do not assume that XGBoost or voting must always win.** The actual result depends on the data, training settings, and test split.

---

## 6. TOP 5 VIVA QUESTIONS

**Q1. What is ensemble learning?**

Ensemble learning combines multiple models or learners to improve predictions or make them more robust.

**Q2. What is AdaBoost?**

AdaBoost is a boosting algorithm that builds sequential learners and gives more attention to examples misclassified by earlier learners.

**Q3. What is Gradient Boosting?**

Gradient Boosting builds models sequentially, with each new model helping reduce the remaining errors of the combined model.

**Q4. What is the difference between hard voting and soft voting?**

Hard voting combines predicted classes by voting. Soft voting combines predicted class probabilities and selects the class with the highest average probability.

**Q5. Why do we use accuracy, precision, recall, and F1-score?**

They measure different aspects of classification performance and help us compare models more thoroughly than using accuracy alone.

---

## 7. LAST-MINUTE REVISION

Remember this sequence:

**Load Iris → Split data → Create three models → Train models → Calculate evaluation scores → Apply hard and soft voting → Compare results.**

And remember these five definitions:

- **Ensemble learning:** Combining learners.
- **AdaBoost:** Focuses more on examples earlier learners misclassify.
- **GBM:** Sequentially reduces the remaining prediction errors.
- **XGBoost:** An optimized gradient-boosted tree implementation.
- **Voting:** Combines the predictions of multiple models.

One final point: your original code includes many extra graphs and classification reports. They can be useful for deeper analysis, but they are not necessary to understand the core practical. Start with the shorter code above, make sure you understand `fit()`, `predict()`, the four metrics, and the difference between hard and soft voting, and then expand only if your instructor specifically requires the extra outputs.