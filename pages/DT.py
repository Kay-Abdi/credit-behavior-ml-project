import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Decision Trees")

# -----------------------------
# Overview
# -----------------------------
st.header("Overview")

st.write(
    "A decision tree is a supervised machine learning method used for classification and prediction. "
    "It works by splitting data into smaller and smaller groups based on the feature that does the best job "
    "separating the classes. In this project, I can use a decision tree to predict whether a customer is likely "
    "to default or not based on financial behavior like credit utilization, debt-to-income ratio, annual income, "
    "spending, and credit score. One of the biggest strengths of decision trees is that they are easy to understand "
    "because they show the logic of the prediction as a series of rules instead of hiding it inside a black box model."
)

st.write(
    "Decision trees are used when the goal is to understand how different variables help predict an outcome. "
    "They can be used for classification problems, like predicting yes or no, default or no default, and also "
    "for regression problems, where the target is a number. In my project, the decision tree is useful because "
    "it helps show which financial behaviors are most important for separating higher-risk customers from lower-risk customers."
)

st.write(
    "To decide where to split the data, a decision tree needs a way to measure how good a split is. "
    "Two common measures are Gini impurity and Entropy. Both are used to measure how mixed a group is. "
    "If a node contains a mix of defaults and non-defaults, the impurity is higher. If a node contains mostly "
    "one class, the impurity is lower. A good split is one that creates purer groups. Information Gain tells us "
    "how much uncertainty is reduced after a split, so it helps the tree choose the split that improves the data the most."
)

st.write(
    "For example, suppose a node has 10 customers: 6 did not default and 4 did default. Before splitting, "
    "the data is somewhat mixed. The entropy of that node is:\n\n"
    "**Entropy = -(6/10)log2(6/10) - (4/10)log2(4/10) = about 0.971**\n\n"
    "Now suppose I split the data using credit utilization ratio:\n\n"
    "- Left group: 5 customers, all non-default\n"
    "- Right group: 5 customers, 1 non-default and 4 default\n\n"
    "The left group is perfectly pure, so its entropy is 0. "
    "The right group is still somewhat mixed, so its entropy is about 0.722.\n\n"
    "The weighted entropy after the split is:\n\n"
    "**(5/10)(0) + (5/10)(0.722) = 0.361**\n\n"
    "Then the Information Gain is:\n\n"
    "**0.971 - 0.361 = 0.610**\n\n"
    "Since the entropy dropped a lot, this would be considered a strong split."
)

st.write(
    "It is generally possible to create an extremely large number of decision trees because there are many choices "
    "at each step. A tree can split on different variables, choose different cutoff values, stop at different depths, "
    "or be pruned in different ways. Even when using the same dataset, small changes in settings or feature selection "
    "can produce a different tree. Because there are so many possible combinations of splits and tree structures, "
    "there is not just one single possible decision tree."
)

st.image(
    "https://media.geeksforgeeks.org/wp-content/uploads/20250626155813124211/Decision-Tree-2.webp",
    caption="Example of a decision tree where each node represents a decision and each branch represents an outcome based on a condition (Source: GeeksforGeeks)"
)

st.image(
    "https://storage.googleapis.com/lds-media/images/gini-impurity-diagram.width-1200.png",
    caption="Visualization of Gini impurity showing how node purity changes based on class distribution (Source: Google Developers)"
)

# -----------------------------
# Data Prep
# -----------------------------
st.header("Data Prep")

st.write(
    "For this section, I used the same credit behavior dataset from earlier in the project. "
    "The goal is to predict whether a customer will default or not using financial behavior variables. "
    "The target variable is 'Defaulted', and the features include variables such as credit score, "
    "annual income, total spending, credit utilization ratio, and debt-to-income ratio."
)

st.write(
    "Before training the decision tree, I prepared the data by removing unnecessary columns like IDs, "
    "and converting any categorical variables into numeric form using one-hot encoding. This is required "
    "because decision tree models in Python expect numeric input."
)

file_path = "data/Credit_Card_Dataset.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

    df = df.drop(columns=["Customer_ID"], errors="ignore")

    y = df["Defaulted"]
    X = df.drop(columns=["Defaulted"])
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.subheader("Training and Testing Sets")
    st.write("Training set shape:", X_train.shape)
    st.write("Testing set shape:", X_test.shape)

    st.write("Training sample:")
    st.dataframe(X_train.head())

    st.write("Testing sample:")
    st.dataframe(X_test.head())

    st.write(
        "The training and testing sets must be disjoint, meaning they do not share any of the same observations. "
        "The training set is used to build the model, while the testing set is used to evaluate how well the model performs "
        "on unseen data. If the same data appeared in both sets, the model could memorize the answers instead of learning real patterns, "
        "which would lead to misleading results and overly high accuracy."
    )

    st.write("Sample data is shown below. Full dataset is stored locally in the project files.")
    st.subheader("Sample of Dataset")
    st.dataframe(df.head(10))

    # -----------------------------
    # Code
    # -----------------------------
    st.header("Code")

    st.write(
        "The code below loads the dataset, separates the target variable from the predictors, "
        "converts categorical variables into numeric form, splits the data into training and testing sets, "
        "and trains a decision tree classifier using entropy as the split criterion."
    )

    dt_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42
    )

    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write("Decision tree model has been trained successfully.")
    st.write("Accuracy:", round(acc, 4))
    st.write("Confusion Matrix:")
    st.write(cm)

    st.markdown(
        "[Link to DT code](https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/DT.py)"
    )

    # -----------------------------
    # Results
    # -----------------------------
    st.header("Results")
    st.write(
    "The decision tree model was evaluated using accuracy and a confusion matrix. "
    "The model achieved an accuracy of approximately 0.6515, meaning it correctly classified about 65% of customers. "
    "The confusion matrix shows how many customers were correctly and incorrectly classified as default or non-default."
    )

    st.write(
    "From the confusion matrix, we can see that the model correctly identified a large number of non-default customers, "
    "but it also made some errors when predicting default cases. This suggests that while the model captures general patterns, "
    "there is still some overlap between higher-risk and lower-risk customer behavior."
    )
    st.write("Confusion matrix and accuracy are shown above. Tree visualizations will go here next.")

    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    st.subheader("Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(15, 8))
    plot_tree(
        dt_model,
        feature_names=X_train.columns,
        class_names=["No Default", "Default"],
        filled=True,
        ax=ax
    )

    st.pyplot(fig)

    criterion="entropy"
    max_depth=4

    st.subheader("Decision Tree (Deeper Tree)")

    tree2 = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=6,
        random_state=42
    )

    tree2.fit(X_train, y_train)

    fig2, ax2 = plt.subplots(figsize=(15, 8))
    plot_tree(
        tree2,
        feature_names=X_train.columns,
        class_names=["No Default", "Default"],
        filled=True,
        ax=ax2
    )

    st.pyplot(fig2)

    st.subheader("Decision Tree (Gini Criterion)")

    tree3 = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42
    )

    tree3.fit(X_train, y_train)

    fig3, ax3 = plt.subplots(figsize=(15, 8))
    plot_tree(
        tree3,
        feature_names=X_train.columns,
        class_names=["No Default", "Default"],
        filled=True,
        ax=ax3
    )

    st.pyplot(fig3)
  
    st.write(
        "To better understand how decision trees behave, I created multiple trees with different settings. "
        "The first tree is a moderate-depth tree using entropy, which keeps the model relatively simple and easier to interpret. "
        "The second tree increases the maximum depth, allowing the model to make more splits and capture more detailed patterns in the data. "
        "However, this also makes the tree more complex and can lead to overfitting, where the model starts memorizing the training data instead of generalizing well."
    )

    st.write(
        "The third tree uses the Gini impurity instead of entropy to decide splits. While both methods aim to create pure nodes, "
        "they measure impurity slightly differently, which can lead to different split decisions and different root nodes. "
        "As seen in the visualizations, changing the depth and the splitting criterion results in trees with different structures, "
        "showing that there is not a single correct way to model the data."
    )

    st.write(
        "Overall, these differences show that decision trees are flexible, but their performance and structure depend heavily on the chosen parameters. "
        "Simpler trees are easier to interpret, while deeper trees can capture more complex relationships but may become harder to understand and less generalizable."
    )
    st.write("Notice how the deeper tree has many more branches, while the simpler tree focuses on the most important splits near the top.")
  #----------------------------
    # Conclusions
    # -----------------------------
    st.header("Conclusions")
    st.write(
    "From the decision tree analysis, I learned that default risk is not random and is strongly connected to financial behavior. "
    "Variables like credit utilization, debt-to-income ratio, spending patterns, and credit score consistently showed up in the tree splits, "
    "which suggests that these features play a major role in separating higher-risk customers from lower-risk ones."
)

    st.write(
        "The model achieved an accuracy of about 65%, which shows that it is able to capture meaningful patterns in the data, "
        "but it is not perfect. This makes sense because customer behavior can overlap, and not all defaults can be predicted using a simple rule-based model."
    )
    st.write(
    "This analysis helps answer several of the main questions in this project. "
    "First, it shows that customers can be separated into different risk groups based on their financial behavior. "
    "Second, it highlights that combinations of variables like high credit utilization and high debt-to-income ratio are associated with higher default risk. "
    "Third, it confirms that spending and credit usage patterns provide additional insight beyond just income alone."
)
    st.write(
    "One of the biggest advantages of decision trees in this project is interpretability. "
    "Unlike some other models, the decision tree clearly shows the step-by-step logic used to classify customers. "
    "This makes it easier to understand which behaviors are driving risk, which is especially important in financial decision-making contexts."
)
    st.write(
    "Overall, the decision tree model provides a useful and interpretable way to understand default risk, "
    "even though it may not be the most accurate model. It works best as a tool for explaining patterns in the data "
    "rather than making perfect predictions."
)
else:
    st.error(f"File not found: {file_path}")