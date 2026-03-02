import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

st.title("Clustering")


st.header("Compare and Contrast: KMeans, Hierarchical, and DBSCAN")
st.write(
    "All three clustering methods aim to group customers with similar behavioral patterns, but they operate under different assumptions and reveal different characteristics of the dataset.\n\n"
    "KMeans is a partition based method that requires selecting the number of clusters in advance. It assigns each point to the nearest centroid and minimizes within cluster variance. This method performs efficiently and provides clear cluster assignments, but it assumes roughly spherical structure and forces every point into a group even if natural separation is weak.\n\n"
    "Hierarchical clustering, specifically using Ward linkage in this analysis, builds clusters progressively by merging the closest groups step by step. It does not require choosing the number of clusters upfront. Instead, the dendrogram visually represents how structure forms across distance levels. This method is useful for understanding the underlying organization of the data, although it can become computationally expensive as dataset size increases.\n\n"
    "DBSCAN is a density based approach that identifies clusters by locating regions of high point concentration. Unlike KMeans, it does not require specifying the number of clusters. Points that do not belong to dense regions are labeled as noise. This allows the method to detect irregular shapes and outliers, but it is sensitive to parameter selection such as eps and minimum samples. \n\n "
    "Both KMeans and hierarchical clustering rely heavily on distance metrics and therefore are sensitive to feature scaling, which is why normalization is important prior to clustering. "
    "KMeans tends to perform best when clusters are compact and convex, while DBSCAN can discover arbitrarily shaped clusters."
)


st.header("Data Preparation")
st.write(
    "Before applying clustering, the dataset was cleaned and transformed to ensure only quantitative features were used. "
    "The target variable Defaulted was removed and stored separately so clustering could be performed in an unsupervised manner. "
    "Categorical identifiers and demographic variables were also removed to avoid distortion in distance based calculations."
)

df = pd.read_csv("data/Credit_Card_Dataset.csv")

st.write("Raw dataset preview (before dropping label/categoricals):")
st.dataframe(df.head())

# Save label separately
y = pd.to_numeric(df["Defaulted"], errors="coerce")

drop_cols = [
    "Customer_ID", "Gender", "Marital_Status",
    "Education_Level", "Employment_Status", "Defaulted"
]
X = df.drop(columns=drop_cols)

st.write("Feature set after dropping label + non-numeric columns:")
st.dataframe(X.head())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
st.write("First 5 rows of scaled data:")
st.dataframe(X_scaled_df.head())
# PCA to 3D
st.write(
    "Principal Component Analysis was applied after standardization in order to reduce dimensionality while preserving as much variance as possible. "
    "Reducing to three components allows visualization in three dimensional space while maintaining a meaningful portion of the original information."
)
pca_3 = PCA(n_components=3, random_state=42)
X_pca_3 = pca_3.fit_transform(X_scaled)

st.subheader("PCA Reduction (3D)")
var_ratio = pca_3.explained_variance_ratio_
st.write("Explained variance ratio (PC1, PC2, PC3):", var_ratio)
st.write("Total variance retained (3 PCs):", f"{var_ratio.sum()*100:.2f}%")
st.write("Shape after PCA:", X_pca_3.shape)



st.header("KMeans Clustering on PCA Reduced Data")
st.write(
    "KMeans clustering was performed for multiple values of k. The silhouette score was used to evaluate how well separated the clusters are. "
    "Higher silhouette values indicate clearer separation, while lower values suggest overlap between groups."
)
sil_scores = {}

for k in [2, 3, 4]:
    st.subheader(f"KMeans with k = {k}")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca_3)

    # Silhouette score
    score = silhouette_score(X_pca_3, cluster_labels)
    sil_scores[k] = score
    st.write(f"Silhouette Score: {score:.4f}")

    centroids = kmeans.cluster_centers_

    # Plot: colored by cluster labels
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2],
               c=cluster_labels, cmap="viridis", alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               c="red", s=200, marker="X")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"KMeans (k={k}) — colored by cluster + centroids")
    st.pyplot(fig)

    # Plot: colored by original label, centroids still shown (this matches the rubric wording better)
    st.caption("Same PCA space, but now colored by the original label (Defaulted). Centroids still shown.")
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2],
                c=y, cmap="coolwarm", alpha=0.6)
    ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                c="black", s=200, marker="X")

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.set_title(f"KMeans (k={k}) — colored by Defaulted + centroids")
    st.pyplot(fig2)


st.header("Hierarchical Clustering (Ward Linkage)")

st.write(
    "Hierarchical clustering was implemented using Ward linkage, which merges clusters in a way that minimizes variance within groups. "
    "To maintain computational efficiency, a random sample of the PCA reduced dataset was used to construct the dendrogram."
)

sample_n = 750
X_sample = (
    pd.DataFrame(X_pca_3, columns=["PC1", "PC2", "PC3"])
    .sample(n=sample_n, random_state=42)
    .values
)

Z = linkage(X_sample, method="ward")

st.subheader(f"Dendrogram (Truncated) — sample size = {sample_n}")
fig = plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("Hierarchical Dendrogram (Ward, Truncated)")
plt.xlabel("Sampled Points")
plt.ylabel("Distance")
st.pyplot(fig)

st.write(
    "The dendrogram does not display a single dramatic increase in linkage distance that would clearly indicate a natural number of clusters. "
    "Instead, cluster merging occurs gradually. This suggests the dataset contains overlapping structure rather than sharply defined segments. "
    "Compared to KMeans, which forces discrete partitions, hierarchical clustering indicates that separation between groups is moderate rather than strong."
)

st.header("DBSCAN Density Based Clustering")
st.write(
    "DBSCAN was applied to evaluate whether the dataset contains clearly separated dense regions. "
    "Unlike KMeans, this method does not force every observation into a cluster. Points that do not belong to sufficiently dense neighborhoods are labeled as noise. "
    "This makes DBSCAN particularly useful for identifying whether true structural density exists in the data."
)

eps_value = st.slider("DBSCAN eps", min_value=0.2, max_value=2.0, value=0.6, step=0.1)
min_samples_value = st.slider("DBSCAN min_samples", min_value=3, max_value=30, value=10, step=1)

dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
db_labels = dbscan.fit_predict(X_pca_3)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()

st.write(f"Estimated number of clusters: **{n_clusters}**")
st.write(f"Number of noise points: **{n_noise}**")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2],
           c=db_labels, cmap="plasma", alpha=0.6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("DBSCAN clusters (PCA space)")
st.pyplot(fig)

st.write(
    f"For eps equal to {eps_value} and minimum samples equal to {min_samples_value}, "
    f"DBSCAN identified {n_clusters} clusters and labeled {n_noise} observations as noise. "
    "A large number of noise points suggests that the dataset does not form strongly separated dense groupings. "
    "Many customers appear to occupy transitional regions rather than compact clusters."
)

st.header("Results and Interpretation")

st.write(
    "The silhouette scores for KMeans remain relatively low across all tested values of k. "
    "Although one value of k achieves the highest score, the differences are minor, indicating limited separation strength overall."
)

st.write(
    "When comparing cluster assignments to the true Defaulted label, there is no clear visual alignment. "
    "Defaulted and non defaulted customers are distributed throughout the PCA space rather than concentrated within specific clusters. "
    "This suggests that default risk alone does not define the primary structural grouping of behavioral patterns."
)

st.write(
    "Hierarchical clustering supports this conclusion, as the dendrogram reveals gradual merging rather than distinct breaks. "
    "DBSCAN further reinforces the interpretation by identifying either a single dominant structure or labeling a substantial portion of observations as noise."
)

st.write(
    "Overall, the dataset demonstrates behavioral continuity rather than sharply segmented customer groups. "
    "Clustering in this context is more informative for describing general behavior patterns than for isolating default risk into clearly separated categories."
)

st.header("Conclusion")

st.write(
    "Across all three clustering approaches, the overall structure of the dataset appears continuous rather than sharply segmented. "
    "KMeans produces partitions, but the silhouette scores remain modest, indicating overlap between groups. "
    "Hierarchical clustering shows gradual merging without a clear structural break, and DBSCAN identifies either a single dominant structure "
    "or a large number of transitional points labeled as noise."
)

st.write(
    "Taken together, these results suggest that customer behavior in this dataset does not naturally divide into strongly separated clusters. "
    "Instead, behavior patterns vary along smooth gradients. Default risk is distributed across this space rather than confined to specific groups."
)

st.write(
    "This indicates that clustering in this context is more useful for describing behavioral tendencies than for isolating default outcomes. "
    "The analysis highlights that customer default is likely influenced by a combination of factors rather than a single distinct behavioral segment."
)

st.write("Link: https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/Clustering.py")