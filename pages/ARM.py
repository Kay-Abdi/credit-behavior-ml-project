import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import networkx as nx


st.title("ARM")

st.header("Overview")

st.write(
    "Association Rule Mining finds relationships between items that co occur together. "
    "A rule has the form A implies B, meaning when A appears in a transaction, B tends to appear as well. "
    "To evaluate rules, we use three main measures. "
    "Support measures how often the full rule occurs in the dataset. "
    "Confidence measures how often B occurs when A occurs. "
    "Lift compares the observed co occurrence to what would be expected if A and B were independent. "
    "Lift greater than 1 suggests a positive association between A and B."
)

st.write(
    "The Apriori algorithm is a classic method for finding frequent itemsets. "
    "It works by using the downward closure property: if an itemset is infrequent, "
    "then any larger itemset that contains it must also be infrequent. "
    "Apriori starts by finding frequent single items, then builds up to larger itemsets by pruning candidates that cannot be frequent."
)

st.subheader("Visual 1: Support, Confidence, Lift")

# Simple example to illustrate the metrics
N = 100
support_A = 30
support_B = 40
support_AB = 18

conf_A_to_B = support_AB / support_A
lift_A_to_B = (support_AB / N) / ((support_A / N) * (support_B / N))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["Support(A)", "Support(B)", "Support(A and B)"],
       [support_A / N, support_B / N, support_AB / N])
ax.set_ylabel("Proportion of transactions")
ax.set_title("Support measures how often an itemset occurs")
st.pyplot(fig)

st.write(
    f"Example values from the chart: confidence(A implies B) = {conf_A_to_B:.2f}, "
    f"lift(A implies B) = {lift_A_to_B:.2f}. Lift above 1 suggests a positive association."
)

st.subheader("Visual 2: Apriori Level Wise Search and Pruning")

levels = ["1 itemsets", "2 itemsets", "3 itemsets", "4 itemsets"]
candidates = [15, 105, 455, 1365]          # example growth
frequent_after_prune = [15, 40, 10, 2]     # example pruning result

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(levels, candidates, marker="o", label="Candidates before pruning")
ax2.plot(levels, frequent_after_prune, marker="o", label="Frequent itemsets after pruning")
ax2.set_ylabel("Count")
ax2.set_title("Apriori grows itemsets level by level and prunes using support")
ax2.legend()
st.pyplot(fig2)

st.write(
    "This illustrates the main Apriori idea: the number of possible itemsets grows fast, "
    "but support based pruning removes most candidates early, which keeps the search manageable."
)

st.header("Data Preparation")

st.write(
    "ARM requires unlabeled transaction style data where each row is a set of items. "
    "This dataset is numeric customer behavior data, so it is transformed into transactions by discretizing key numeric features "
    "into categorical bins. Each customer becomes a transaction of behavioral categories such as High_Total_Transactions or Low_Fraud_Transactions."
)

df = pd.read_csv("data/Credit_Card_Dataset.csv")
st.caption("Dataset file in repo: data/Credit_Card_Dataset.csv")

# Save label separately (not used for ARM input)
if "Defaulted" in df.columns:
    y = pd.to_numeric(df["Defaulted"], errors="coerce")
else:
    y = None

drop_cols = ["Customer_ID", "Gender", "Marital_Status", "Education_Level", "Employment_Status", "Defaulted"]
drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols)

# Use PCA to justify feature selection, then discretize those top features for ARM
st.subheader("Selecting Important Features Using PCA")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)

pc1_loadings = pd.DataFrame(
    {"PC1 Weight": pca_full.components_[0]},
    index=X.columns
)
pc1_loadings["Absolute Weight"] = pc1_loadings["PC1 Weight"].abs()
top_features = pc1_loadings.sort_values("Absolute Weight", ascending=False).head(5)

st.write("Top features based on absolute PC1 loading:")
st.dataframe(top_features)

selected_cols = top_features.index.tolist()

st.write("Features used for ARM transactions:")
st.write(selected_cols)

st.subheader("Discretizing Into Behavioral Items")

arm_df = df[selected_cols].copy()

# Discretize into 3 bins: Low, Medium, High
for col in selected_cols:
    arm_df[col] = pd.qcut(
        arm_df[col],
        q=3,
        labels=[f"Low_{col}", f"Medium_{col}", f"High_{col}"]
    )

st.write("Preview of discretized categorical data (this becomes the item set per customer):")
st.dataframe(arm_df.head())

transactions = arm_df.values.tolist()

st.write("Sample transactions (first 5 rows):")
st.write(transactions[:5])

st.write(
    "At this point, the data is unlabeled transaction data. "
    "Each row is a list of categorical items, which is the required input format for Apriori."
)



st.header("Code ARM")

st.write(
    "The code below uses mlxtend to one hot encode the transaction lists, then runs Apriori to find frequent itemsets, "
    "then generates association rules from those itemsets."
)

st.subheader("Thresholds Used")

min_support = st.slider("Min support", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
min_confidence = st.slider("Min confidence", min_value=0.01, max_value=0.90, value=0.30, step=0.01)

st.write("Selected thresholds:")
st.write(f"Min support: {min_support}")
st.write(f"Min confidence: {min_confidence}")

# One hot encode
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
onehot = pd.DataFrame(te_array, columns=te.columns_)

# Frequent itemsets
itemsets = apriori(onehot, min_support=min_support, use_colnames=True)

# Rules
if itemsets.empty:
    st.warning("No frequent itemsets found. Lower min support.")
    st.stop()

rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)

# Keep rules of length at least 2 total items
if not rules.empty:
    rules = rules[(rules["antecedents"].apply(len) + rules["consequents"].apply(len)) >= 2]

if rules.empty:
    st.warning("No rules found with these thresholds. Lower min confidence or min support.")
    st.stop()

st.write("Number of frequent itemsets found:", len(itemsets))
st.write("Number of rules found:", len(rules))



st.header("Results")

st.write(
    "Below are the top rules ranked by support, confidence, and lift. "
    "Support highlights rules that occur often. Confidence highlights reliability. "
    "Lift highlights strength beyond chance."
)

def format_itemset(s):
    return ", ".join(sorted(list(s)))

rules_display = rules.copy()
rules_display["antecedents"] = rules_display["antecedents"].apply(format_itemset)
rules_display["consequents"] = rules_display["consequents"].apply(format_itemset)

show_cols = ["antecedents", "consequents", "support", "confidence", "lift"]

st.subheader("Top 15 Rules by Support")
top_support = rules_display.sort_values("support", ascending=False).head(15)[show_cols]
st.dataframe(top_support)

st.subheader("Top 15 Rules by Confidence")
top_conf = rules_display.sort_values("confidence", ascending=False).head(15)[show_cols]
st.dataframe(top_conf)

st.subheader("Top 15 Rules by Lift")
top_lift = rules_display.sort_values("lift", ascending=False).head(15)[show_cols]
st.dataframe(top_lift)

st.subheader("Network Visualization of Associations")

st.write("""
This graph shows some of the strongest relationships found in the data.
Each circle represents a behavior, and the arrows show which behaviors tend to lead to others.
Thicker arrows mean the connection between them is stronger (higher lift).
""")

# Build a graph from top N lift rules
top_n = st.slider("Number of rules to visualize", min_value=5, max_value=30, value=15, step=1)
vis_rules = rules.sort_values("lift", ascending=False).head(top_n)

G = nx.DiGraph()

for _, row in vis_rules.iterrows():
    ants = list(row["antecedents"])
    cons = list(row["consequents"])
    lift = float(row["lift"])
    conf = float(row["confidence"])

    # If multiple items appear in antecedent or consequent, connect each antecedent item to each consequent item
    for a in ants:
        for c in cons:
            if G.has_edge(a, c):
                # Keep the strongest association if duplicate appears
                if lift > G[a][c]["lift"]:
                    G[a][c]["lift"] = lift
                    G[a][c]["confidence"] = conf
            else:
                G.add_edge(a, c, lift=lift, confidence=conf)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

edges = list(G.edges())
weights = [G[u][v]["lift"] for u, v in edges]

nx.draw_networkx_nodes(G, pos, node_size=900, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=8)

# Draw directed edges with width based on lift
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=edges,
    width=[max(0.5, w) for w in weights],
    alpha=0.6,
    arrows=True,
    arrowsize=12
)

plt.title("Association Rule Network (Top Lift Rules)")
plt.axis("off")
st.pyplot(plt.gcf())
plt.clf()



st.header("Conclusions")

st.write(
    "These rules describe combinations of customer behavior categories that frequently appear together. "
    "Because the items were created by discretizing the most influential behavioral features, the discovered associations are interpretable as behavioral patterns. "
    "High lift rules suggest combinations that occur more often than expected by chance, which can be useful for describing customer segments and identifying co occurring behaviors. "
    "In the context of this dataset, ARM is most useful as a descriptive tool that highlights common behavioral profiles, rather than a direct predictor of default on its own."
)

st.write("Link: https://github.com/Kay-Abdi/credit-behavior-ml-project/blob/main/pages/ARM.py")