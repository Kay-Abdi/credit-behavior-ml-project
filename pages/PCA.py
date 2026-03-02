import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("PCA")


st.header("Overview")
st.write(
    "Principal Component Analysis (PCA) is a technique used to reduce the number of features in a dataset while still keeping most of the important information. "
    "It works by transforming correlated variables into new variables called principal components, which are linear combinations of the original features and are ordered based on how much variance they explain. "
    "In the context of credit card behavior, many variables like credit limit, bill amounts, payments, and usage are naturally correlated with each other. "
    "PCA helps simplify these relationships by projecting the data into fewer dimensions, making it easier to visualize patterns in customer behavior and understand which directions in the data explain the most variation. "
    "Instead of working with a large number of overlapping features, PCA allows us to capture the core structure of the data in a more compact and interpretable way."
)


st.header("Data")

df = pd.read_csv("data/Credit_Card_Dataset.csv")

st.subheader("Raw data preview")
st.dataframe(df.head())
st.write("Dataset used: Kaggle Credit Card User Behavior Dataset")
st.write("Link: https://www.kaggle.com/datasets/aadarshvani/credit-card-dataset-comprehensive")

if "Defaulted" not in df.columns:
    st.error("Column 'Defaulted' not found in the dataset.")
    st.stop()

y = df["Defaulted"]

drop_cols = ["Customer_ID", "Gender", "Marital_Status", "Education_Level", "Employment_Status", "Defaulted"]
missing = [c for c in drop_cols if c not in df.columns]
if missing:
    st.error(f"These columns were expected but not found: {missing}")
    st.stop()

X = df.drop(columns=drop_cols)

st.subheader("Cleaned quantitative data used for PCA (label removed)")
st.dataframe(X.head())

st.write("Label column saved separately for later comparisons:", y.name)
st.write("Shape of X (rows, features):", X.shape)


st.header("Code")

st.subheader("Standardizing the Data (StandardScaler)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
st.write("First 5 rows of scaled data:")
st.dataframe(X_scaled_df.head())

st.subheader("PCA with 2 Components")
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

pca_2_df = pd.DataFrame(X_pca_2, columns=["PC1", "PC2"])
st.write("First 5 rows of 2D PCA-transformed data:")
st.dataframe(pca_2_df.head())

st.write("Explained variance ratio (PC1, PC2):", pca_2.explained_variance_ratio_)
st.write("Cumulative explained variance (2D):", float(pca_2.explained_variance_ratio_.sum()))

st.subheader("2D PCA Visualization")
fig, ax = plt.subplots()
ax.scatter(X_pca_2[:, 0], X_pca_2[:, 1], alpha=0.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA (2 Components)")
st.pyplot(fig)


st.subheader("PCA with 3 Components")

pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)

pca_3_df = pd.DataFrame(X_pca_3, columns=["PC1", "PC2", "PC3"])

st.write("First 5 rows of 3D PCA-transformed data:")
st.dataframe(pca_3_df.head())

st.write("Explained variance ratio (PC1, PC2, PC3):")
st.write(pca_3.explained_variance_ratio_)

st.write("Cumulative explained variance (3D):")
st.write(float(pca_3.explained_variance_ratio_.sum()))
from mpl_toolkits.mplot3d import Axes3D

st.subheader("3D PCA Visualization")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X_pca_3[:, 0],
    X_pca_3[:, 1],
    X_pca_3[:, 2],
    alpha=0.4
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (3 Components)")

st.pyplot(fig)


st.subheader("Number of Components Needed for 95% Variance")

# PCA with all components
pca_full = PCA()
pca_full.fit(X_scaled)

# Cumulative variance
import numpy as np
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find minimum components for 95%
num_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

st.write("Cumulative variance by component:")
st.write(cumulative_variance)

st.write(f"Number of components needed to retain at least 95% variance: {num_components_95}") 
st.subheader("Top 3 Eigenvalues")

st.write(pca_full.explained_variance_[:3])


st.subheader("Top Feature Weights for PC1")

loadings = pd.DataFrame(
    pca_full.components_[0],
    index=X.columns,
    columns=["PC1 Weight"]
)

st.dataframe(loadings.sort_values(by="PC1 Weight", ascending=False)) 


st.header("Results")

st.write(
    f"Using PCA with two components (PC1 and PC2), the dataset retains about "
    f"{pca_2.explained_variance_ratio_.sum()*100:.2f}% of the original variance. "
    "This means that while 2D is helpful for visualization, it does not capture most of the overall complexity in customer credit behavior. "
    "The scatter plot shows broad structure, but there is still significant variation spread across other dimensions."
)

st.write(
    f"When adding the third principal component (PC3), the cumulative explained variance increases to "
    f"{pca_3.explained_variance_ratio_.sum()*100:.2f}%. "
    "This improves the representation of the data, but even in 3D we are not retaining the majority of the total variance. "
    "This suggests that credit card behavior patterns are distributed across many features rather than being dominated by just a few strong directions."
)
st.write(
    "To retain at least 95% of the total variance, the dataset requires 15 principal components. "
    "Since the original dataset contains 18 numerical features, this means the information is distributed "
    "across many dimensions rather than being concentrated in only a few dominant directions. "
    "In other words, the credit card behavior data is inherently high-dimensional."
)

st.write(
    "Looking at the PCA loadings for PC1, the strongest contributing variables are Total_Transactions, "
    "Unique_Transaction_Cities, and Unique_Merchant_Categories. This suggests that the primary direction of variation "
    "in the dataset is driven by overall transaction activity and behavioral diversity rather than traditional risk metrics "
    "like credit score or debt-to-income ratio. In other words, the most dominant pattern in the data reflects how active "
    "and behaviorally diverse customers are, rather than purely financial risk characteristics."
)



st.write("Link: https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/PCA.py")
