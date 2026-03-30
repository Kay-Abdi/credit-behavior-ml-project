import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.header("(A) Overview")

st.write("""
### Naïve Bayes Overview

Naïve Bayes (NB) is a classification algorithm based on probability. It uses Bayes’ Theorem to estimate how likely a data point belongs to a certain class, then assigns it to the class with the highest probability. It’s called “naïve” because it assumes all features are independent, which isn’t really true in real data, but it still works surprisingly well.

NB is usually used when you want something fast, simple, and efficient. It works well with high-dimensional data and is often used as a baseline model. You’ll see it a lot in text classification, spam detection, and prediction problems like this one.

There are different versions of Naïve Bayes depending on the type of data.

Gaussian Naïve Bayes (GNB) is used for continuous numerical data like income, credit score, or utilization ratio. It assumes the data follows a normal distribution, which fits most financial variables in this project.

Multinomial Naïve Bayes (MNB) is used for count-based data. It focuses on how often something happens instead of just whether it happens. For example, in text classification it looks at word counts. In this project, it could relate to how often certain behaviors occur.

Bernoulli Naïve Bayes (BNB) works with binary data (0 or 1). It only cares about whether something happened, not how many times. For example, whether a customer missed a payment.

Categorical Naïve Bayes (CNB) is used for categorical features like education level or gender. It works directly with categories instead of converting them into continuous numbers.

### Why Smoothing is Needed

One issue with Naïve Bayes is that if a feature never appears in the training data for a class, the probability becomes zero. Since NB multiplies probabilities together, one zero would make the entire prediction zero.

To fix this, smoothing is used. Smoothing adds a small value to every probability so nothing becomes zero. This helps the model handle unseen data better.

Overall, all Naïve Bayes models follow the same idea. The main difference is the type of data they assume. In this project, Gaussian NB fits best since most variables are continuous.
""")

st.image("https://betterexplained.com/ColorizedMath/img/Bayes_Theorem.png")
st.image("https://media.geeksforgeeks.org/wp-content/uploads/20260227115947481605/outlook.webp")
st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*oBpcc5GIf6hcZoxNqpslqg.png")

st.header("(b) Data Prep")

df = pd.read_csv("data/Credit_Card_Dataset.csv")

df_model = df.copy()
for col in df_model.select_dtypes(include=["object"]).columns:
    df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

st.write("""
Before applying Naïve Bayes, the data needs to be prepared. This is a supervised learning task, so we need a label. In this project, the label is whether a customer defaulted.

Categorical variables were encoded into numbers so the model can process them. The target variable (Defaulted) was separated from the features.

The features (X) represent customer behavior, while the label (y) represents default.
""")

st.subheader("Dataset Preview")
st.dataframe(df_model.head(10))

st.write("Target column:", "Defaulted")

X = df_model.drop("Defaulted", axis=1)
y = df_model["Defaulted"]

st.subheader("Features (X)")
st.dataframe(X.head())

st.subheader("Target (y)")
st.dataframe(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

col1, col2 = st.columns(2)

with col1:
    st.write("Training Set")
    st.dataframe(X_train.head())
    st.write("Shape:", X_train.shape)

with col2:
    st.write("Testing Set")
    st.dataframe(X_test.head())
    st.write("Shape:", X_test.shape)

st.write("""
The training set is used to build the model, and the testing set is used to evaluate it on unseen data. These sets must be separate, otherwise the results would be misleading.
""")

st.subheader("Data for Different NB Models")

X_gaussian = X.copy()

X_multinomial = X.copy()
for col in X_multinomial.columns:
    if X_multinomial[col].min() < 0:
        X_multinomial[col] = X_multinomial[col] - X_multinomial[col].min()

X_bernoulli = X.copy()
for col in X_bernoulli.columns:
    X_bernoulli[col] = (X_bernoulli[col] > X_bernoulli[col].median()).astype(int)

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Gaussian NB")
    st.dataframe(X_gaussian.head())

with col2:
    st.write("Multinomial NB")
    st.dataframe(X_multinomial.head())

with col3:
    st.write("Bernoulli NB")
    st.dataframe(X_bernoulli.head())

st.write("""
Each Naïve Bayes model requires a different data format:

- Gaussian NB uses continuous data  
- Multinomial NB requires non-negative values  
- Bernoulli NB requires binary data  

So the dataset was adjusted depending on the model being used.
""")

st.header("(c) Code")

st.write("""
The Naïve Bayes models (Gaussian, Multinomial, and Bernoulli) were implemented in Python using scikit-learn. 
The full code for data preparation, model training, and evaluation can be found below.
""")

st.markdown("[View Naïve Bayes Code on GitHub](https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/NB.py)")
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_gaussian, y, test_size=0.2, random_state=42, stratify=y
)

Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_multinomial, y, test_size=0.2, random_state=42, stratify=y
)

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_bernoulli, y, test_size=0.2, random_state=42, stratify=y
)

gnb = GaussianNB()
gnb.fit(Xg_train, yg_train)
gnb_acc = accuracy_score(yg_test, gnb.predict(Xg_test))

mnb = MultinomialNB()
mnb.fit(Xm_train, ym_train)
mnb_acc = accuracy_score(ym_test, mnb.predict(Xm_test))

bnb = BernoulliNB()
bnb.fit(Xb_train, yb_train)
bnb_acc = accuracy_score(yb_test, bnb.predict(Xb_test))

results = pd.DataFrame({
    "Model": ["Gaussian NB", "Multinomial NB", "Bernoulli NB"],
    "Accuracy": [gnb_acc, mnb_acc, bnb_acc]
})

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.subheader("Confusion Matrices")

fig1, ax1 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(yg_test, gnb.predict(Xg_test), ax=ax1)
ax1.set_title("Gaussian NB Confusion Matrix")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(ym_test, mnb.predict(Xm_test), ax=ax2)
ax2.set_title("Multinomial NB Confusion Matrix")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(yb_test, bnb.predict(Xb_test), ax=ax3)
ax3.set_title("Bernoulli NB Confusion Matrix")
st.pyplot(fig3)

st.subheader("Accuracy Comparison")

fig4, ax4 = plt.subplots()
ax4.bar(results["Model"], results["Accuracy"])
ax4.set_ylabel("Accuracy")
ax4.set_title("Naïve Bayes Model Accuracy Comparison")
st.pyplot(fig4)

st.write("""
The confusion matrices show how each model performed by comparing the predicted values to the actual values. The diagonal values represent correct predictions, while the off-diagonal values represent mistakes. This makes it easier to see not just overall accuracy, but also what kinds of errors each model is making.

The bar chart gives a quick visual comparison of the three Naïve Bayes models. This helps show which version performed best on this dataset.
""")
st.subheader("Model Accuracy")

st.dataframe(results)

st.header("(e) Conclusions")

st.write("""
From this, I learned that customer behavior does show patterns that relate to default risk. Even with the independence assumption, Naïve Bayes still performs reasonably well.

Gaussian NB worked best because most variables are continuous. The other models required transforming the data, which can lose some information.

The results suggest that combinations of financial behaviors, like high utilization and debt levels, are linked to higher risk. This shows that behavior patterns matter, not just single variables.

At the same time, Naïve Bayes has limits because it assumes features are independent. More complex models like Decision Trees or Logistic Regression can capture relationships better.

Overall, Naïve Bayes is a solid baseline and shows that default risk can be predicted from customer behavior.
""")