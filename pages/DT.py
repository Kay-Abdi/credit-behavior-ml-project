import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

st.title("Decision Trees")
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
    caption="Example of a decision tree"
)

st.image(
    "https://storage.googleapis.com/lds-media/images/gini-impurity-diagram.width-1200.png",
    caption="Visualization of Gini impurity"
)

st.header("Data Prep")

st.write("""
For this section, I used the same credit behavior dataset from earlier in the project. The goal is to predict whether a customer will default or not using financial behavior variables. The target variable is Defaulted, and the features include variables like credit score, annual income, total spending, credit utilization ratio, and debt-to-income ratio.

Before training the model, I prepared the data by removing unnecessary columns like IDs and converting categorical variables into numeric form using one-hot encoding. This is needed because decision tree models in Python require numeric input.
""")

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

    st.write("""
The training set is used to build the model, while the testing set is used to evaluate how well it performs on unseen data. These two sets need to be fully separate. If the same rows showed up in both, the results would look better than they really are.
""")

    st.subheader("Sample of Dataset")
    st.dataframe(df.head(10))

    st.header("Code")

    st.write("""
The code below loads the dataset, separates the target variable from the predictors, converts categorical variables into numbers, splits the data into training and testing sets, and trains a decision tree classifier using entropy as the split criterion.
""")

    dt_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=4,
        random_state=42
    )

    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write("Decision tree model has been trained successfully.")
    st.write("Accuracy:", round(acc, 4))

    st.subheader("Confusion Matrix")

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    ax_cm.set_title("Decision Tree Confusion Matrix")
    st.pyplot(fig_cm)

    st.subheader("Code Snippet")

    st.write("""
    The decision tree model was implemented in Python using scikit-learn. The full code for data preparation, model training, and visualization can be found below.
    """)

    st.markdown("[View Decision Tree Code on GitHub](https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/DT.py)")

    st.header("Results")

    st.write("""
The decision tree model was evaluated using accuracy and a confusion matrix. The model gets around 65% accuracy, which means it is picking up on real patterns in the data, but it is not perfect.

The confusion matrix gives a better breakdown of what is actually happening. The diagonal values are correct predictions, and the off-diagonal values are errors. From this, you can see that the model does a better job predicting non-default customers than default ones. That means it captures general behavior pretty well, but it struggles more with higher-risk cases.

Overall, there is still overlap between low-risk and high-risk customers, which makes prediction harder.
""")

    st.subheader("Decision Tree Visualization")

    fig1, ax1 = plt.subplots(figsize=(15, 8))
    plot_tree(
        dt_model,
        feature_names=X_train.columns,
        class_names=["No Default", "Default"],
        filled=True,
        ax=ax1
    )
    st.pyplot(fig1)

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
        max_features=3,
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

    st.write("""
To better understand how decision trees behave, I made three different versions. The first tree uses entropy with a moderate depth, so it stays simpler and easier to read. The second tree goes deeper, which lets it capture more detailed patterns, but it also makes the model more complex and easier to overfit. The third tree uses Gini instead of entropy, which can lead to different split choices and a different tree structure.

Looking at all three makes it easier to see that there is not just one single way to build a decision tree on the same dataset. Small changes in settings can lead to different results and different levels of complexity.
""")

    st.header("Conclusions")

    st.write("""
From the decision tree analysis, I learned that default risk is not random and is clearly connected to financial behavior. Variables like credit utilization, debt-to-income ratio, spending, and credit score show up in the splits, which means they play a big role in separating higher-risk customers from lower-risk ones.

The model is not perfect, but it still captures meaningful patterns. It helps answer the main questions in this project by showing that combinations of behaviors matter, not just one variable by itself. It also shows that spending and credit usage patterns give extra insight beyond income alone.

One of the biggest strengths of decision trees in this project is interpretability. The tree shows the step-by-step logic behind the prediction, which makes it easier to understand what is driving customer risk.

Overall, the decision tree model gives a useful and interpretable way to understand default risk, even if it is not the most accurate model.
""")

else:
    st.error(f"File not found: {file_path}")