import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


st.subheader("Linear Regression")
st.write("""
Linear regression is a method used to predict a continuous numeric value. It works by fitting a straight line through the data that best represents the relationship between the input variables and the outcome. The model tries to minimize the difference between predicted and actual values. It is useful when the goal is to estimate something like income, spending, or debt as a number. Overall, it helps capture trends between variables in a simple and interpretable way.
""")

st.write("""
Logistic regression is used for classification problems, usually when the outcome has two categories like yes or no. Instead of predicting a number, it predicts the probability that a data point belongs to a certain class. That probability is then used to assign a final label based on a threshold. In this project, it is used to predict whether a customer will default or not.
""")

st.write("""
Both linear and logistic regression model relationships between input variables and an outcome using weighted combinations of features. However, linear regression predicts continuous values, while logistic regression predicts probabilities for classification. Logistic regression is better suited for problems like predicting default risk.
""")

st.write("""
Yes, logistic regression uses the sigmoid function. The sigmoid function transforms any input value into a number between 0 and 1, allowing the model to output probabilities. These probabilities are then used to classify outcomes.
""")

st.write("""
Maximum likelihood is the method used to train logistic regression. Instead of minimizing error, it finds the parameters that maximize the probability of the observed data. This allows the model to better match predictions to actual outcomes.
""")

st.subheader("Data Preparation")

df = pd.read_csv("data/Credit_Card_Dataset.csv")

y = df["Defaulted"]
X = df.drop(columns=["Defaulted", "Customer_ID"], errors="ignore")

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("Training set size:", X_train.shape)
st.write("Testing set size:", X_test.shape)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

st.subheader("Code")

st.write("""
Logistic regression, Naïve Bayes, and Decision Tree models were implemented in Python using scikit-learn. 
The full code for preprocessing, model training, and evaluation can be found below.
""")

st.markdown("[View Regression Code on GitHub](https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/Regression.py)")

st.subheader("Model Comparison")

st.write("### Logistic Regression")
st.write("Accuracy:", accuracy_score(y_test, log_preds))
st.write(pd.DataFrame(
    confusion_matrix(y_test, log_preds),
    columns=["Pred 0", "Pred 1"],
    index=["Actual 0", "Actual 1"]
))

fig1, ax1 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, log_preds, ax=ax1)
ax1.set_title("Logistic Regression Confusion Matrix")
st.pyplot(fig1)

st.write("### Naive Bayes")
st.write("Accuracy:", accuracy_score(y_test, nb_preds))
st.write(pd.DataFrame(
    confusion_matrix(y_test, nb_preds),
    columns=["Pred 0", "Pred 1"],
    index=["Actual 0", "Actual 1"]
))

fig2, ax2 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, nb_preds, ax=ax2)
ax2.set_title("Naive Bayes Confusion Matrix")
st.pyplot(fig2)

st.write("### Decision Tree")
st.write("Accuracy:", accuracy_score(y_test, dt_preds))
st.write(pd.DataFrame(
    confusion_matrix(y_test, dt_preds),
    columns=["Pred 0", "Pred 1"],
    index=["Actual 0", "Actual 1"]
))

fig3, ax3 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, dt_preds, ax=ax3)
ax3.set_title("Decision Tree Confusion Matrix")
st.pyplot(fig3)

st.write("""
The confusion matrices show how each model performs beyond just accuracy. The diagonal values are correct predictions, while the off-diagonal values show mistakes.

This makes it easier to see how well each model identifies default and non-default customers instead of only looking at one overall score.
""")

st.subheader("Important Features (Logistic Regression)")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": log_model.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)

st.write(feature_importance.head(10))

st.subheader("Results and Comparison")

st.write("""
Logistic regression achieved the highest accuracy and provided the most balanced performance. It was better at identifying both defaulting and non-defaulting customers. This is likely because it can model relationships between variables without assuming independence.

Naive Bayes performed slightly worse and predicted most customers as non-defaulting. This suggests that its assumption of feature independence does not hold well for financial data, where variables like income, spending, and credit utilization are closely related.

The decision tree model was more flexible and identified more default cases than Naive Bayes, but it had lower overall accuracy. It is more sensitive to the data and can overfit, which may explain the drop in performance.

Overall, logistic regression performed best because it captures how multiple financial behaviors interact rather than treating them as independent.
""")
st.subheader("Conclusion")

st.write("""
Logistic regression performs the best here because it balances both types of predictions. From the confusion matrix, it correctly identifies a large number of non-default customers while still catching a decent amount of default cases. It is not perfect, but it avoids heavily favoring one class over the other.

Naive Bayes struggles the most. The confusion matrix shows that it predicts most customers as non-default, which leads to a large number of missed default cases. This means it has a high number of false negatives, which is a serious issue in this context. Missing a default is worse than incorrectly flagging someone, so this model would not be reliable for real use.

The decision tree does a better job than Naive Bayes at identifying default cases, but it makes more mistakes overall, especially false positives. It is more flexible, but also more sensitive to the data, which makes its predictions less stable.

These results show that default risk comes from a combination of financial behaviors, not just one variable. Models that can capture relationships between variables perform better, which is why logistic regression works best here.

If this were used in a real financial setting, logistic regression would be the best choice. It provides the most balanced predictions and avoids missing too many high-risk customers. In practice, it would be better to slightly over-predict risk than to miss actual defaults, since false negatives carry a higher cost.
""")