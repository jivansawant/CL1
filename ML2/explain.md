# ML Practical 2: Regression Analysis on Diabetes Data

## 1. CORE TAKEAWAY

**This practical uses statistical measures to understand diabetes data, linear regression to predict a continuous value, and logistic regression to predict a binary outcome. It also demonstrates multiple regression, where several patient measurements are used together to make a prediction.**

You have only `diabetics.csv` (your sample contains the Pima Indians Diabetes dataset columns). That's enough to perform the main statistical and regression analyses, but **you cannot honestly compare two datasets without a second dataset**.

Your practical title mentions both a UCI diabetes dataset and the Pima Indians Diabetes dataset. We'll handle this in two stages:

- **Now:** Complete the analysis using your available Pima dataset.
- **Later, if required:** Obtain the second dataset and repeat the same analyses to compare their results.

There is also an important correction to your original code: it trains and evaluates several models on the same observations. That can make the results look better than their performance on unseen data. We'll use a training/testing split for the regression models so the evaluation is more meaningful.

---

## 2. MINIMALIST PYTHON CODE

### Markdown Cell 1 — Complete Python code

**Before running:** Keep `diabetics.csv` in the same folder as your Jupyter Notebook. If your actual filename is `diabetes.csv`, change the filename in the first line that reads the file.

This version includes all the essential operations: univariate statistics, frequency analysis, bivariate linear and logistic regression, multiple linear regression, multiple logistic regression, and model evaluation.

```python
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("diabetics.csv")

# Select numerical features
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Remove rows with missing values in the required columns
df = df.dropna(subset=features + ["Outcome"])

# A. UNIVARIATE ANALYSIS
print("Outcome frequency:")
print(df["Outcome"].value_counts().sort_index())

stats = pd.DataFrame({
    "Mean": df[features].mean(),
    "Median": df[features].median(),
    "Mode": df[features].mode().iloc[0],
    "Variance": df[features].var(),
    "Standard Deviation": df[features].std(),
    "Skewness": df[features].apply(skew),
    "Kurtosis": df[features].apply(kurtosis)
})

print("\nUnivariate Statistics:")
print(stats.round(3))

print("\nFrequency of Glucose values:")
print(df["Glucose"].value_counts().head(10))

# B. BIVARIATE LINEAR REGRESSION
X = df[["Glucose"]]
y = df["BMI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear = LinearRegression()
linear.fit(X_train, y_train)
pred = linear.predict(X_test)

print("\nBivariate Linear Regression")
print("BMI =", round(linear.intercept_, 3), "+",
      round(linear.coef_[0], 3), "* Glucose")
print("R2:", round(r2_score(y_test, pred), 3))
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3))

plt.scatter(X_test["Glucose"], y_test, label="Actual")
plt.scatter(X_test["Glucose"], pred, label="Predicted")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Linear Regression: Glucose vs BMI")
plt.legend()
plt.show()

# B. BIVARIATE LOGISTIC REGRESSION
X = df[["Glucose"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logistic = make_pipeline(StandardScaler(), LogisticRegression())
logistic.fit(X_train, y_train)
pred = logistic.predict(X_test)

print("\nBivariate Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# C. MULTIPLE LINEAR REGRESSION
multiple_features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "DiabetesPedigreeFunction", "Age"
]

X = df[multiple_features]
y = df["BMI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

multiple_linear = LinearRegression()
multiple_linear.fit(X_train, y_train)
pred = multiple_linear.predict(X_test)

print("\nMultiple Linear Regression")
print("R2:", round(r2_score(y_test, pred), 3))
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 3))

# D. MULTIPLE LOGISTIC REGRESSION
X = df[multiple_features]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

multiple_logistic = make_pipeline(StandardScaler(), LogisticRegression())
multiple_logistic.fit(X_train, y_train)
pred = multiple_logistic.predict(X_test)

print("\nMultiple Logistic Regression")
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
```

### Important notes about this code

1. **The target variables are chosen for demonstration.** Linear regression predicts BMI from Glucose, while logistic regression predicts `Outcome` from Glucose. Multiple regression uses several features to predict BMI or `Outcome`, respectively. If your instructor specifies a different dependent variable, we should follow that requirement.
2. **We use a test set.** The model learns from the training data and is evaluated on separate observations.
3. **Standardization is used for logistic regression.** This makes the feature scales more comparable and helps the model fit reliably.
4. **The code removes missing rows, but does not automatically replace zero measurements.** In the Pima dataset, zeros in some clinical measurements can represent missing or invalid measurements rather than genuine readings. Check this with your instructor before deciding how to handle them.
5. **The second dataset comparison remains incomplete.** This code analyzes your available dataset only; it does not claim to compare two datasets.

---

## 3. WHY CODE WALKTHROUGH

### Step 1: Import the libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
```

- `pandas` handles the dataset and calculates statistics.
- `matplotlib` draws the graph.

```python
from scipy.stats import skew, kurtosis
```

These functions calculate skewness and kurtosis, which describe the shape of a numerical distribution.

```python
from sklearn.model_selection import train_test_split
```

This separates the data into training and testing sets.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
```

These provide the two regression algorithms, feature scaling, and a convenient way to apply scaling before logistic regression.

The remaining imports calculate performance metrics such as R², MAE, RMSE, accuracy, and the confusion matrix.

### Step 2: Load and prepare the dataset

```python
df = pd.read_csv("diabetics.csv")
```

This reads the CSV file.

```python
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
```

This identifies the eight numerical input measurements in your sample.

```python
df = df.dropna(subset=features + ["Outcome"])
```

This removes rows containing missing values in the required columns.

**Why?** Regression algorithms generally need usable numerical values to calculate a model. Removing incomplete rows is a simple preprocessing approach.

However, this does not fix biologically invalid zeros. For example, a zero glucose reading in this dataset may indicate a missing measurement.

### A. Univariate analysis

Univariate means analyzing **one variable at a time**.

#### Step 3: Frequency

```python
print(df["Outcome"].value_counts().sort_index())
```

This counts how many records belong to each outcome class.

Typically:
- `Outcome = 0`: Diabetes not indicated by the dataset's label.
- `Outcome = 1`: Diabetes indicated by the dataset's label.

The counts show the distribution of the target classes.

```python
print(df["Glucose"].value_counts().head(10))
```

This displays the ten most frequent glucose values and their counts.

**Why do we calculate frequency?**

Frequency tells us how often a particular value or category occurs.

#### Step 4: Calculate descriptive statistics

```python
stats = pd.DataFrame({
    "Mean": df[features].mean(),
    "Median": df[features].median(),
    "Mode": df[features].mode().iloc[0],
    "Variance": df[features].var(),
    "Standard Deviation": df[features].std(),
    "Skewness": df[features].apply(skew),
    "Kurtosis": df[features].apply(kurtosis)
})
```

This calculates seven statistical measures for each numerical feature.

Here is what each measure means:

| Measure | Meaning | Why we calculate it |
|---|---|---|
| Mean | Arithmetic average | Understand the central value |
| Median | Middle value after sorting | Understand the center while being less affected by extreme values |
| Mode | Most frequently occurring value | Identify the most common value |
| Variance | Measure of squared deviation from the mean | Understand how spread out the values are |
| Standard deviation | Square root of variance | Measure spread in the original units |
| Skewness | Measures distribution asymmetry | Understand whether the distribution leans toward one side |
| Kurtosis | Describes tail weight relative to a normal distribution | Identify distributions with relatively heavy or light tails |

**Mean formula:**

Mean = Sum of all observations / Number of observations

**Variance formula:**

Sample variance = Sum of squared differences from the mean / (Number of observations − 1)

**Standard deviation formula:**

Standard deviation = Square root of variance

Pandas uses sample variance and sample standard deviation by default.

For skewness:
- Positive skewness generally indicates a longer right tail.
- Negative skewness generally indicates a longer left tail.
- A value near zero suggests approximate symmetry, although it does not prove that the distribution is normal.

For kurtosis, SciPy's default `kurtosis()` calculates **excess kurtosis**:
- Around zero: similar kurtosis to a normal distribution.
- Positive: heavier tails than a normal distribution, in the usual interpretation.
- Negative: lighter tails than a normal distribution, in the usual interpretation.

Don't confuse kurtosis with a direct measure of how tall a graph looks.

### B. Bivariate linear regression

Bivariate analysis studies the relationship between two variables.

Here, we use:
- Independent variable: `Glucose`
- Dependent variable: `BMI`

#### Step 5: Separate input and output

```python
X = df[["Glucose"]]
y = df["BMI"]
```

`X` contains the glucose measurement used to make predictions.

`y` contains the actual BMI values we want to predict.

Why use double square brackets for `X`?

`df[["Glucose"]]` produces a DataFrame with two-dimensional structure, as expected by scikit-learn. In contrast, `df["Glucose"]` produces a one-dimensional Series.

#### Step 6: Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

This divides the dataset into:
- 80% training data.
- 20% testing data.

`random_state=42` makes the split reproducible. Running the same code with the same data produces the same split.

Why split the data?

We want to check how well the model predicts observations it did not use for training.

#### Step 7: Train linear regression

```python
linear = LinearRegression()
linear.fit(X_train, y_train)
pred = linear.predict(X_test)
```

- `LinearRegression()` creates the model.
- `fit()` learns the relationship between glucose and BMI.
- `predict()` estimates BMI for the testing observations.

Linear regression models a relationship using a straight line.

**Equation in plain text:**

BMI = Intercept + (Coefficient × Glucose)

The intercept is the predicted BMI when glucose equals zero, according to the fitted line. The coefficient describes the estimated change in BMI associated with a one-unit increase in glucose.

This describes a statistical association, not necessarily a causal relationship.

#### Step 8: Evaluate the model

```python
r2_score(y_test, pred)
mean_absolute_error(y_test, pred)
mean_squared_error(y_test, pred) ** 0.5
```

These calculate three regression metrics.

| Metric | Meaning | Better result |
|---|---|---|
| R² | How much variation in the test target is explained relative to predicting its mean | Generally higher |
| MAE | Average absolute prediction error | Lower |
| RMSE | Square root of average squared prediction error | Lower |

R² can be negative when predictions perform worse than the baseline that predicts the mean of the training target.

MAE and RMSE are expressed in the target's units. In this case, they are expressed in BMI units.

#### Step 9: Plot the regression results

```python
plt.scatter(X_test["Glucose"], y_test, label="Actual")
plt.scatter(X_test["Glucose"], pred, label="Predicted")
```

This plots actual and predicted BMI values against glucose.

The graph helps us visually inspect how closely the predictions match the observed values.

Your simplified code uses two sets of points instead of adding an extra regression line. This is sufficient to demonstrate the comparison.

### B. Bivariate logistic regression

Now we predict the binary `Outcome` using only `Glucose`.

#### Step 10: Select the variables

```python
X = df[["Glucose"]]
y = df["Outcome"]
```

Here:
- Input: Glucose.
- Target: Outcome.

Unlike linear regression, the target is categorical and contains two classes.

#### Step 11: Split and train the model

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

The split again reserves 20% of the observations for testing.

`stratify=y` attempts to preserve the proportion of each outcome class in both sets.

```python
logistic = make_pipeline(StandardScaler(), LogisticRegression())
logistic.fit(X_train, y_train)
pred = logistic.predict(X_test)
```

The pipeline performs two operations:

1. Standardizes glucose using statistics learned from the training data.
2. Trains logistic regression on the standardized glucose measurements.

When predictions are made, the same fitted scaler is applied automatically.

**Why logistic regression instead of linear regression?**

Linear regression predicts continuous numerical values. Logistic regression estimates the probability of a class, making it suitable for a binary outcome.

For example, a logistic model might estimate a probability of 0.75 for a particular sample. With the usual 0.5 decision threshold, that sample would be predicted as class 1.

This is an illustrative example, not a result from your dataset.

#### Step 12: Evaluate the classification

```python
accuracy_score(y_test, pred)
confusion_matrix(y_test, pred)
```

Accuracy measures the proportion of test samples classified correctly.

Accuracy = Correct predictions / Total predictions

The confusion matrix summarizes the correct and incorrect predictions.

For binary classification, scikit-learn arranges it as:

|  | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | True negatives | False positives |
| Actual 1 | False negatives | True positives |

A false negative is particularly important to understand in a medical prediction task: it means a sample labelled as class 1 was predicted as class 0.

**Important:** Accuracy alone does not tell us how well the model identifies every positive case. That is one reason your original code included recall and precision.

### C. Multiple linear regression

Multiple linear regression predicts a continuous target using several input features.

In this practical:
- Target: BMI.
- Predictors: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction, and Age.

#### Step 13: Select multiple predictors

```python
X = df[multiple_features]
y = df["BMI"]
```

Instead of using glucose alone, the model uses seven predictors.

Why?

BMI may be statistically associated with several patient measurements. Multiple regression allows the model to estimate their combined relationship with BMI.

#### Step 14: Train and evaluate the model

```python
multiple_linear = LinearRegression()
multiple_linear.fit(X_train, y_train)
pred = multiple_linear.predict(X_test)
```

The model learns a linear equation with multiple predictors.

**Equation in plain text:**

BMI = Intercept + (Coefficient 1 × Pregnancies) + (Coefficient 2 × Glucose) + ... + (Coefficient 7 × Age)

Each coefficient represents the model's estimated change in BMI for a one-unit increase in that feature, holding the other included features constant.

We then calculate R², MAE, and RMSE to evaluate the predictions on the test set.

### D. Multiple logistic regression

Multiple logistic regression predicts the binary `Outcome` using several input measurements.

#### Step 15: Select the variables

```python
X = df[multiple_features]
y = df["Outcome"]
```

The model uses seven predictors to estimate the probability of the diabetes outcome.

We deliberately exclude BMI from the predictors here because the original code's selected feature list excludes it. BMI could be included if that matches the instructor's requirements.

#### Step 16: Train and evaluate

```python
multiple_logistic = make_pipeline(StandardScaler(), LogisticRegression())
multiple_logistic.fit(X_train, y_train)
pred = multiple_logistic.predict(X_test)
```

The pipeline standardizes the predictors before fitting logistic regression.

Why standardize multiple predictors?

They use different numerical scales. Standardization makes the numerical optimization more stable and makes the regularization used by scikit-learn's logistic regression less dependent on the original measurement scales.

The model is then evaluated using accuracy and the confusion matrix.

### E. Comparing the models

Your practical has two different comparison tasks.

**First: Compare bivariate and multiple regression on your available dataset.**

- Compare bivariate linear regression against multiple linear regression using R², MAE, and RMSE.
- Compare bivariate logistic regression against multiple logistic regression using accuracy and their confusion matrices.

For these comparisons to be fair, both models for a given target should be evaluated on the same test observations. The current code uses reproducible splits, but the multiple regression and bivariate regression splits are created separately. They will generally use the same rows when the dataset and split settings are unchanged, but stratification changes the sampling behavior for logistic regression. A rigorous comparison should explicitly reuse a common split for each target.

**Second: Compare the two datasets named in the assignment.**

You cannot complete this part yet because you have supplied only one dataset. Once you obtain the second dataset, repeat the same analyses and compare the corresponding statistics and test-set metrics. Different datasets may contain different variables, units, populations, and sample sizes, so comparisons must account for those differences.

---

## 4. VIVA CONCEPT CRASH COURSE

These are the most important concepts to understand for this practical.

### A. Linear regression vs logistic regression

| Property | Linear Regression | Logistic Regression |
|---|---|---|
| Purpose | Predict a continuous value | Predict a class |
| Example in your code | Predict BMI | Predict diabetes Outcome |
| Target | Numerical | Binary: 0 or 1 |
| Output | Numerical prediction | Probability, then predicted class |
| Typical evaluation | R², MAE, RMSE | Accuracy, confusion matrix |

**Memory trick:**
- Linear regression → How much?
- Logistic regression → Which class?

### B. Bivariate vs multiple regression

**Bivariate regression** examines a relationship between two variables, usually one predictor and one target.

Example:

Glucose → BMI

**Multiple regression** uses several predictors to estimate one target.

Example:

Pregnancies + Glucose + BloodPressure + Insulin + Age + other selected features → BMI

Remember that multiple regression does not mean predicting multiple target variables. It means using multiple predictors to predict a target.

### C. What is the difference between a parameter and a metric?

A **parameter** is a value learned by the model, such as a regression coefficient or intercept.

A **metric** measures how well the model performs, such as R² or accuracy.

For example:
- Linear regression coefficient: learned parameter.
- R²: performance metric.
- Logistic regression coefficient: learned parameter.
- Accuracy: performance metric.

### D. What are training and testing data?

Training data teaches the model the relationship between the inputs and the target.

Testing data evaluates the model on observations that were not used for training.

We split the data because evaluating a model on its training observations can give an overly optimistic picture of how well it performs on new data.

In this practical, we use an 80:20 split.

### E. What is the difference between R², MAE, and RMSE?

**R²:** Measures predictive performance relative to a baseline that predicts the target mean. Higher is generally better.

**MAE:** Measures the average absolute difference between actual and predicted values.

MAE = Average of the absolute prediction errors.

**RMSE:** Measures prediction error while penalizing larger errors more strongly.

RMSE = Square root of the average squared prediction errors.

For MAE and RMSE, lower is better.

### F. What is a confusion matrix?

A confusion matrix summarizes the predictions made by a classification model.

It tells us:
- True positives: Actual positive cases predicted as positive.
- True negatives: Actual negative cases predicted as negative.
- False positives: Actual negative cases predicted as positive.
- False negatives: Actual positive cases predicted as negative.

In medical prediction, false negatives can be especially important because they represent positive cases that the model fails to identify.

### G. What are precision, recall, and F1-score?

Your original code calculates these metrics, so you should understand them even though we have simplified the main version.

**Precision** measures how many predicted positive cases are actually positive.

Precision = True positives / (True positives + False positives)

**Recall** measures how many actual positive cases the model correctly identifies.

Recall = True positives / (True positives + False negatives)

**F1-score** combines precision and recall using their harmonic mean.

F1-score = 2 × Precision × Recall / (Precision + Recall)

These measures are useful when the classes are imbalanced or when different types of classification errors have different consequences.

### H. Why might multiple regression perform better?

Multiple regression has access to more predictors than bivariate regression, so it can capture additional relationships with the target.

However, **more predictors do not guarantee better performance on unseen data**. Additional features may be unhelpful, noisy, or redundant.

Compare the models using test-set results rather than assuming the multiple regression model is automatically superior.

### I. What is the purpose of univariate analysis?

Univariate analysis describes individual variables before examining relationships between them.

For example:
- Mean glucose gives an idea of the average glucose measurement.
- Median BMI describes the middle BMI value.
- Glucose variance describes its spread.
- Glucose skewness describes its asymmetry.

These statistics help us understand the dataset before fitting regression models.

### J. What are the limitations of this practical?

1. **Missing or invalid values:** Removing missing rows does not correct zero values that may represent missing clinical measurements.
2. **Limited predictors:** A model using glucose alone may miss other useful relationships.
3. **Statistical associations:** Regression coefficients do not automatically establish causation.
4. **Model evaluation:** Test-set metrics can vary depending on the split.
5. **Dataset comparison:** Comparing two datasets requires access to both and careful consideration of their variables and populations.

Finally, these models are for learning regression analysis. Their predictions should not be used to diagnose an individual or replace medical evaluation.

---

## 5. TOP 3 EXPECTED VIVA QUESTIONS

### Q1. What is the difference between linear and logistic regression?

**Answer:** Linear regression predicts continuous numerical values, such as BMI. Logistic regression predicts the probability of a categorical outcome, such as the binary diabetes label.

### Q2. What is the difference between bivariate and multiple regression?

**Answer:** Bivariate regression uses one predictor and one target to study their relationship. Multiple regression uses two or more predictors to estimate a target.

### Q3. Why do we divide the dataset into training and testing sets?

**Answer:** Training data is used to fit the model, while testing data evaluates its performance on unseen observations. This gives a more realistic estimate of how well the model may generalize.

---

## Final Practical Revision

Memorize this sequence:

**Load data → Calculate statistics → Perform linear regression → Perform logistic regression → Perform multiple regression → Evaluate models → Compare results.**

The practical has two important limitations to remember:

- You currently have only the Pima Indians Diabetes dataset, so the required comparison between two datasets is still pending.
- The provided code demonstrates both continuous-value prediction and binary classification, but the exact dependent variables should match your instructor's requirements.

**Your next step:** Run this simplified code on `diabetics.csv`. If it produces an error, send me the exact error message. If it runs successfully, keep the output because your actual statistical values and model metrics are worth revising for the viva.