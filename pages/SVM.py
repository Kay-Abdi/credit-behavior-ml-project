import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("SVM")

# ---------------------------
# Overview
# ---------------------------
st.header("Overview")

st.write("""
Support Vector Machines (SVMs) are supervised learning models used for classification. That basically means the model learns from data that already has labels and then tries to classify new data based on what it learned. The goal of an SVM is to find the best boundary that separates different classes. But instead of just picking any boundary, it tries to find the one with the largest margin, meaning the most space between the two groups. The points closest to that boundary are called support vectors, and they are the ones that really determine where the boundary ends up.
""")

st.write("""
SVMs are called linear separators because, at their core, they try to separate classes using a straight line (or a flat boundary). In 2D this is just a line, in 3D it becomes a plane, and in higher dimensions it’s called a hyperplane. If the data is already cleanly separated, this works really well. But in real life, data usually isn’t that simple, and the classes might overlap or be shaped in a more complicated way.
""")

st.write("""
That’s where kernels come in. Kernels let SVM handle more complex data by basically transforming it into a higher-dimensional space where it becomes easier to separate. The key idea is that we don’t actually have to compute all those new dimensions directly. Instead, the kernel trick lets us act like we did, without doing all the extra work. This makes the model way more efficient while still being able to capture more complex patterns.
""")

st.write("""
The dot product plays a big role in all of this because most kernel functions are built around it. The dot product is just a way of measuring how similar two data points are. SVM uses that idea to understand relationships between points and decide how to separate them. So instead of just looking at raw values, it’s really looking at how points relate to each other in space.
""")

st.latex(r"K(x, z) = (x \cdot z + r)^d")

st.write("""
The polynomial kernel builds on the dot product by adding a constant and raising the result to a power. This allows the model to create curved decision boundaries instead of just straight lines. The degree controls how complex that curve can get.
""")

st.latex(r"K(x, z) = e^{-\gamma \|x - z\|^2}")

st.write("""
The RBF (radial basis function) kernel works differently. Instead of focusing on dot products, it looks at the distance between points. This makes it really flexible and good at handling more complicated patterns. The gamma value controls how much influence each point has. A higher gamma means the model focuses more on nearby points.
""")

st.write("""
For example, if we take a 2D point like (x1, x2) and apply a polynomial kernel with r = 1 and degree = 2, that point can be transformed into a higher-dimensional space with extra features like squared terms and interaction terms. So instead of just (x1, x2), it becomes something like (1, x1, x2, x1², x2², x1x2). This is important because data that wasn’t separable before might become separable after this transformation.
""")

st.write("""
In the context of this project, SVM is useful because customer financial behavior isn’t always simple or cleanly separated. Some customers might look similar in one feature but very different when you consider multiple behaviors together. By using different kernels, we can test whether the relationship between behavior and risk is mostly linear or if it needs a more flexible boundary to separate the groups properly.
""")

st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*oRk-5aab0G8SkBX2fpw8Gw.png")
st.caption("This shows how SVM separates the data by creating the widest possible gap between the two groups. The points closest to the boundary (support vectors) are what actually determine where that boundary goes.")

st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*2syBCIlXnIwF6LNjRrObeQ.png")
st.caption("This shows the difference between a nonlinear boundary (left) and a linear one (right). The nonlinear boundary fits the data way better, which is why kernels matter for more complex patterns.")

# ---------------------------
# Data
# ---------------------------
st.header("Data")

df = pd.read_csv("data/Credit_Card_Dataset.csv")
target_col = "Defaulted"

numeric_df = df.select_dtypes(include=["number"])
X = numeric_df.drop(columns=[target_col])
y = numeric_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write("""
All machine learning models require data in a specific format, and supervised models like SVM are no exception. In supervised learning, the model needs labeled data. This means that for every observation, we already know the correct outcome or category. The model learns patterns from these labeled examples and then uses those patterns to make predictions on new data. Without labeled data, the model would not know what it is trying to predict, so supervised methods would not work.
""")

st.write("""
Another important step is splitting the data into a training set and a testing set. The training set is used to build the model, meaning this is where the model learns the relationships between the features and the target variable. The testing set is then used to evaluate how well the model performs on new, unseen data. This helps give a more realistic measure of how accurate the model actually is.
""")

st.write("""
It is important that the training and testing sets are completely separate, or disjoint. If the same data points appear in both sets, the model could end up memorizing the answers instead of actually learning patterns. This would make the model seem more accurate than it really is, which is misleading. By keeping them separate, we make sure the evaluation reflects real-world performance.
""")

st.write("""
SVMs also require that all input features are numeric. This is because the model relies heavily on mathematical operations like distances, margins, and dot products. These operations only make sense with numerical values, so any categorical data must either be converted into numbers or removed before training the model. In this project, only numeric features were used.
""")

st.write("""
After preparing the data, it was split into training and testing sets using a standard approach. The training set was used to fit the model, while the testing set was used to evaluate its performance. This setup ensures that the results reflect how the model would behave on new data rather than just the data it has already seen.
""")

st.subheader("Sample of Data Before Transformation")
st.dataframe(df.head(10))

st.subheader("Train vs Test Split")
st.write(f"Training set size: {len(X_train)}")
st.write(f"Testing set size: {len(X_test)}")

st.bar_chart({
    "Training": [len(X_train)],
    "Testing": [len(X_test)]
})

st.subheader("Sample Training Data")
st.dataframe(X_train.head(5))

st.subheader("Sample Testing Data")
st.dataframe(X_test.head(5))

st.markdown("**Dataset Link:** https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/data/Credit_Card_Dataset.csv")
st.markdown("**Code Link:** https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/SVM.py")

# ---------------------------
# Code
# ---------------------------
st.header("Code")

st.write("""
The SVM model was implemented in Python using scikit-learn. Because SVM depends on distances and margins, the numeric features were standardized before training so that variables on larger scales would not dominate the model. Three kernels were tested: linear, polynomial, and radial basis function (RBF). Different cost values were also tested to compare how strongly the model penalized misclassification.
""")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# transformed data preview
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

st.subheader("Sample of Data After Transformation")
st.write("Scaled Training Data Sample")
st.dataframe(X_train_scaled_df.head(5))

st.write("Scaled Testing Data Sample")
st.dataframe(X_test_scaled_df.head(5))

models = {
    "Linear, C=0.1": SVC(kernel="linear", C=0.1),
    "Linear, C=1": SVC(kernel="linear", C=1),
    "Linear, C=10": SVC(kernel="linear", C=10),

    "Polynomial, C=0.1": SVC(kernel="poly", degree=3, C=0.1, gamma="scale"),
    "Polynomial, C=1": SVC(kernel="poly", degree=3, C=1, gamma="scale"),
    "Polynomial, C=10": SVC(kernel="poly", degree=3, C=10, gamma="scale"),

    "RBF, C=0.1": SVC(kernel="rbf", C=0.1, gamma="scale"),
    "RBF, C=1": SVC(kernel="rbf", C=1, gamma="scale"),
    "RBF, C=10": SVC(kernel="rbf", C=10, gamma="scale"),
}

results = []
saved_outputs = {}

with st.spinner("Training SVM models..."):
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)

        results.append({
            "Model": name,
            "Accuracy": acc
        })

        saved_outputs[name] = {
            "model": model,
            "y_pred": y_pred,
            "cm": cm,
            "accuracy": acc,
            "report": report
        }

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

st.subheader("Model Accuracy Comparison")
st.dataframe(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_accuracy = results_df.iloc[0]["Accuracy"]

st.write(f"Best model so far: **{best_model_name}** with an accuracy of **{best_accuracy:.4f}**")

# ---------------------------
# Results
# ---------------------------
st.header("Results")

st.write("""
The SVM models were tested using three different kernels: linear, polynomial, and RBF. For each kernel, three different cost values were tried in order to compare how model flexibility and penalty strength affected performance. Accuracy, confusion matrices, and classification reports were used to evaluate the results.
""")

st.write("""
The linear kernel acts as the most basic version and tests whether the classes can be separated with a straight boundary. The polynomial kernel allows for more curved boundaries by introducing higher-order feature interactions. The RBF kernel is the most flexible of the three and is often useful when the relationship between variables is more complex.
""")

# show all 9 confusion matrices + classification reports
st.subheader("Confusion Matrices and Classification Reports for All Models")

for model_name in results_df["Model"]:
    st.subheader(model_name)
    st.write(f"Accuracy: {saved_outputs[model_name]['accuracy']:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        saved_outputs[model_name]["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)

    st.text("Classification Report")
    st.text(saved_outputs[model_name]["report"])

st.subheader("Accuracy by Model")
chart_df = results_df.copy().set_index("Model")
st.bar_chart(chart_df)
st.write(f"""
Based on the models tested, the best-performing SVM was **{best_model_name}**, with an accuracy of **{best_accuracy:.4f}**. 

This model achieved the highest accuracy among all tested configurations, indicating that it provides the best balance between model flexibility and generalization. In particular, the polynomial kernel with C=1 suggests that moderate complexity allows the model to capture meaningful feature interactions without overfitting.
""")

st.write("""
Comparing the kernels provides insight into the structure of the data. The linear kernel assumes that the classes can be separated using a straight boundary, and while it performed reasonably well, its accuracy was slightly lower than the nonlinear models. This suggests that the data is not perfectly linearly separable.

The polynomial kernel performed the best overall, which indicates that interactions between features are important for predicting default risk. By introducing higher-order relationships, the model is able to capture patterns that a linear model cannot.

The RBF kernel also performed competitively, showing that more flexible boundaries can model the data effectively. However, it did not outperform the polynomial model, which suggests that the data benefits from some nonlinearity, but does not require extremely complex transformations.
""")

st.write("""
Looking at the confusion matrices, most models were able to correctly classify a large portion of non-default cases, but struggled more with identifying default cases. This is likely due to class imbalance, where default cases are less frequent and therefore harder for the model to learn.
""")

st.write("""
Looking across all nine model settings, changing the cost value did affect performance, but larger values did not always improve the results. This shows that increasing model complexity or penalizing misclassification more aggressively does not automatically lead to better predictions. In this case, moderate values of C provided a better balance between fitting the data well and generalizing to unseen examples.
""")
from sklearn.decomposition import PCA
import numpy as np
st.subheader("Decision Boundary Visualizations")
# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Models to visualize
viz_models = {
    "Linear (C=1)": SVC(kernel="linear", C=1),
    "Polynomial (C=1)": SVC(kernel="poly", degree=3, C=1),
    "RBF (C=1)": SVC(kernel="rbf", C=1)
}

# Plot each model
for name, model in viz_models.items():
    model.fit(X_pca, y_train)

    # Create grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()

    # background
    ax.contourf(xx, yy, Z, levels=20, alpha=0.3)

    # decision boundary
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

    # data points
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_train,
        edgecolors='k'
    )

    ax.set_title(f"Decision Boundary - {name}")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

    st.pyplot(fig)


st.write("""
To better understand how SVM separates the data, the dataset was reduced to two dimensions using PCA so that the decision boundaries could be visualized. Because the original dataset contains many features, it is not possible to directly visualize the true high-dimensional boundary. Instead, PCA provides a simplified 2D representation that captures the most important variation in the data.

This means that the visualizations shown here are approximations of how the models behave, rather than exact representations of the true decision boundaries. As a result, the boundaries may not appear perfectly clean or clearly separated.
""")
st.write("""
The decision boundary visualizations highlight how difficult it is to separate the data, even after reducing it to two dimensions using PCA. Because the original dataset contains many features, the PCA transformation compresses that information into just two components, which results in significant overlap between the classes.

In the linear model, the decision boundary appears as a straight diagonal line, but it does not meaningfully separate the two groups. Most of the data points from both classes are mixed together, which shows that the data is not linearly separable in this reduced space.

The polynomial model introduces some curvature into the boundary, but the overall separation is still weak. While the model is slightly more flexible, the heavy overlap between classes makes it difficult for the boundary to clearly distinguish between default and non-default cases.

The RBF model produces a more complex and localized boundary, with regions that attempt to adapt more closely to the data. However, even with this added flexibility, the classes remain highly intermixed, and the boundary does not result in a clear separation.

Overall, these visualizations show that the dataset is inherently difficult to separate when projected into two dimensions. While the models are able to learn patterns in higher-dimensional space (as reflected in their accuracy scores), those patterns are not easily visible in the PCA-reduced plots. This reinforces the idea that the relationship between financial behavior and default risk is complex and not easily captured by simple visual boundaries.
""")
# ---------------------------
# Conclusions
# ---------------------------
st.subheader("Conclusions")

st.write("""
The SVM results show that customer financial behavior can be used to classify default risk with moderate accuracy. Among the models tested, the polynomial kernel with C=1 performed the best, achieving an accuracy of approximately 0.67. This suggests that the relationship between financial behavior and default risk is not purely linear, but also does not require extremely complex decision boundaries.
""")

st.write("""
The linear models all performed similarly, with accuracy values around 0.66. This indicates that a simple linear separation captures some of the structure in the data, but not all of it. In contrast, the polynomial kernel was able to slightly improve performance by capturing interactions between features, which suggests that combinations of financial behaviors play a role in determining risk.
""")

st.write("""
The RBF kernel also performed well, with accuracy close to the polynomial model. This shows that more flexible nonlinear boundaries can help capture patterns in the data. However, increasing the cost parameter too much did not improve performance and in some cases made it worse. This suggests that overly complex models may begin to overfit the data rather than generalize well.
""")

st.write("""
Overall, the SVM analysis reinforces the idea that default risk is influenced by multiple interacting behavioral factors rather than a single variable. While the model is able to capture meaningful patterns, the accuracy levels suggest that financial behavior alone does not fully determine default outcomes, and additional variables or modeling approaches may be needed for stronger predictions.
""")

st.write("""
From a practical perspective, this means that while SVM can help identify higher-risk customers based on behavior, it should likely be used alongside other models or features for more reliable predictions. The comparison between kernels also highlights the importance of testing both simple and flexible models when analyzing real-world financial data.
""")