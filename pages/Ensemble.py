import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

st.title("Ensemble Learning")

# ---------------------------
# Overview
# ---------------------------
st.header("Overview")

st.write("""
Ensemble learning is a machine learning approach that combines multiple models instead of relying on just one. The basic idea is that a collection of models can often make better predictions than a single model alone. Rather than depending on one decision rule, ensemble methods combine the outputs of many smaller models and use them to make one final prediction. This often improves accuracy, stability, and generalization.
""")

st.write("""
There are different ways ensemble learning can work. Some methods, like Random Forest, build many models independently and then combine them through voting. Other methods, like Gradient Boosting, build models one after another, where each new model tries to correct mistakes made by the earlier ones. In this sense, ensemble learning works by either averaging many models together or improving predictions step by step.
""")

st.image("https://media.geeksforgeeks.org/wp-content/uploads/20251216112828071155/bagging.webp")
st.caption("This shows how bagging works. Multiple models are trained on different subsets of the data, and their predictions are combined to make a final decision. This is the core idea behind Random Forest.")
st.write("""
In this project, ensemble learning is useful because customer default risk is not usually determined by one variable alone. It is more likely influenced by a combination of financial behaviors such as spending, repayment, and credit usage. Ensemble methods are helpful here because they can capture more of that complexity than a single model.
""")

st.write("""
For this project, the ensemble methods used were Random Forest and Gradient Boosting. Random Forest works by building many decision trees on slightly different samples of the data and combining their predictions. Gradient Boosting also uses trees, but it builds them sequentially so that each new tree focuses on correcting earlier mistakes. Together, these methods provide two different ensemble strategies for modeling customer default risk.
""")

# ---------------------------
# Data / Code
# ---------------------------
st.header("Code")

st.write("""
The code for this part of the project uses Python and scikit-learn to train two ensemble classification models: Random Forest and Gradient Boosting. The code first loads the credit card dataset, keeps only the numeric variables, and separates the predictors from the labeled target variable, Defaulted. It then splits the data into training and testing sets so the models can be trained on one portion of the data and evaluated on unseen data.
""")

st.write("""
This code applies directly to the project because the goal is to predict customer default risk based on financial behavior. Ensemble methods are useful here because default risk is likely influenced by multiple interacting variables rather than a single feature. Random Forest helps by combining many decision trees through voting, while Gradient Boosting builds trees step by step to correct earlier mistakes.
""")

st.write("""
The code provides several results that help evaluate the models. These include accuracy scores, confusion matrices, classification reports, and feature importance rankings. Together, these results show how well each ensemble method predicts default risk and which financial behavior variables contribute most to those predictions.
""")

# ---- LOAD DATA ----
df = pd.read_csv("data/Credit_Card_Dataset.csv")
target_col = "Defaulted"

numeric_df = df.select_dtypes(include=["number"])
X = numeric_df.drop(columns=[target_col])
y = numeric_df[target_col]

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.subheader("Training and Testing Split")
split_df = pd.DataFrame({
    "Set": ["Training", "Testing"],
    "Count": [len(X_train), len(X_test)]
})

st.write(f"Training set size: {len(X_train)}")
st.write(f"Testing set size: {len(X_test)}")
st.bar_chart(split_df.set_index("Set"))
st.caption("The dataset was split into disjoint training and testing sets. The training set was used to build the models, while the testing set was used to evaluate performance on unseen data.")
# ---------------------------
# Data Preparation Visualization
# ---------------------------
st.subheader("Data Before Preparation")
st.dataframe(df.head(5))
st.caption("This shows the dataset before selecting the variables used for modeling.")

st.subheader("Data After Preparation")
st.dataframe(numeric_df.head(5))
st.caption("This shows the dataset after keeping only numeric variables and the labeled target used for the ensemble models.")

st.subheader("Sample Training Data")
st.dataframe(X_train.head(5))

st.subheader("Sample Testing Data")
st.dataframe(X_test.head(5))
st.markdown("**Dataset Link:** https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/data/Credit_Card_Dataset.csv")
st.markdown("**Code Link:** https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/Ensemble.py")

# ---------------------------
# Model Training
# ---------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
rf_report = classification_report(y_test, rf_pred, output_dict=True)

gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)
gb_cm = confusion_matrix(y_test, gb_pred)
gb_report = classification_report(y_test, gb_pred, output_dict=True)

comparison_df = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Accuracy": rf_accuracy,
        "Precision (1)": rf_report["1"]["precision"],
        "Recall (1)": rf_report["1"]["recall"],
        "F1-score (1)": rf_report["1"]["f1-score"]
    },
    {
        "Model": "Gradient Boosting",
        "Accuracy": gb_accuracy,
        "Precision (1)": gb_report["1"]["precision"],
        "Recall (1)": gb_report["1"]["recall"],
        "F1-score (1)": gb_report["1"]["f1-score"]
    }
])

# ---------------------------
# Results
# ---------------------------
st.header("Results")

st.write("""
Both ensemble methods were used to classify customers into default and non-default groups based on their financial behavior. Random Forest combines many independently built trees through voting, while Gradient Boosting builds trees sequentially and focuses on correcting earlier mistakes.
""")

st.subheader("Model Performance Comparison")
st.dataframe(comparison_df)

st.write(f"""
Random Forest achieved an accuracy of **{rf_accuracy:.4f}**, while Gradient Boosting achieved an accuracy of **{gb_accuracy:.4f}**.
""")

# ---- RANDOM FOREST OUTPUTS ----
st.subheader("Random Forest Results")
st.write(f"Accuracy: **{rf_accuracy:.4f}**")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Random Forest Confusion Matrix")
st.pyplot(fig)

st.text("Random Forest Classification Report")
st.text(classification_report(y_test, rf_pred))

rf_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

st.subheader("Top 10 Random Forest Feature Importances")
st.dataframe(rf_importance_df)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(rf_importance_df["Feature"], rf_importance_df["Importance"])
ax.set_title("Top 10 Random Forest Feature Importances")
ax.set_ylabel("Importance")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

# ---- GRADIENT BOOSTING OUTPUTS ----
st.subheader("Gradient Boosting Results")
st.write(f"Accuracy: **{gb_accuracy:.4f}**")

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(gb_cm, annot=True, fmt="d", cmap="Purples", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Gradient Boosting Confusion Matrix")
st.pyplot(fig)

st.text("Gradient Boosting Classification Report")
st.text(classification_report(y_test, gb_pred))

gb_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gb_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

st.subheader("Top 10 Gradient Boosting Feature Importances")
st.dataframe(gb_importance_df)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(gb_importance_df["Feature"], gb_importance_df["Importance"])
ax.set_title("Top 10 Gradient Boosting Feature Importances")
ax.set_ylabel("Importance")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)

if gb_accuracy > rf_accuracy:
    st.write("""
In this case, Gradient Boosting performed slightly better than Random Forest. This suggests that the boosting approach was better able to improve performance by correcting mistakes step by step.
""")
elif rf_accuracy > gb_accuracy:
    st.write("""
In this case, Random Forest performed slightly better than Gradient Boosting. This suggests that combining many independently built trees provided a more stable fit for this dataset.
""")
else:
    st.write("""
In this case, both models performed about the same. This suggests that the dataset contains useful predictive structure, but neither ensemble strategy had a strong advantage over the other.
""")
st.write("""
In simple terms, both models were able to use customer financial behavior to make reasonably good guesses about which customers were more likely to default. One model performed slightly better, but both showed that spending, repayment, and credit usage patterns contain useful information about risk. At the same time, neither model was perfect, which shows that default risk is still a difficult outcome to predict exactly.
""")
# ---------------------------
# Conclusions
# ---------------------------
st.header("Conclusions")

st.write("""
The ensemble results show that combining multiple decision trees can be an effective way to classify customer default risk. Both Random Forest and Gradient Boosting were able to capture meaningful structure in the data, although one performed slightly better than the other depending on the final accuracy.
""")

st.write("""
This part of the project also shows that default risk is influenced by multiple financial behavior variables working together rather than by one single factor. The feature importance results reinforce that idea by showing which variables contributed most to the models’ predictions.
""")

st.write("""
The confusion matrices highlight an important tradeoff. The models are able to correctly identify a portion of default cases, but they also misclassify some non-default customers. This reflects the challenge of predicting financial risk, where improving detection often comes at the cost of more false positives.
""")

st.write("""
Overall, the ensemble analysis supports the broader idea of this project: customer financial behavior contains meaningful patterns that can be used to predict risk. Comparing Random Forest and Gradient Boosting gives a more complete understanding of customer behavior and shows how different ensemble methods can produce different strengths.
""")