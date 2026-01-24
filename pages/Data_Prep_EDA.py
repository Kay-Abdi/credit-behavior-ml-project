import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.world_bank_api import df_world_bank

st.title("Data Prep + EDA")

st.write("""
My project will use a credit card customer behavior dataset as the primary source of individual-level financial behavior data, 
and World Bank API data on domestic credit to the private sector as macroeconomic context.
""")
st.write("""
During data exploration, I considered both customer-level and transaction-level financial datasets.
While transaction-level data provides very granular detail, it requires heavy aggregation and feature
engineering before it can be used for behavioral modeling. Since the goal of this project is to analyze
credit behavior using clustering, classification, and dimensionality reduction techniques introduced
in this course, I selected a customer-level dataset as the primary data source. This allows for more
direct interpretation of behavioral patterns while still capturing meaningful financial risk signals.
""")


st.subheader("Raw Data Overview")

df_raw=pd.read_csv("data/Credit_Card_Dataset.csv")
st.write("**Credit Card Dataset (raw)**")
st.write(f"Rows: {df_raw.shape[0]:,} | Columns: {df_raw.shape[1]:,}")

st.write("Preview of raw data (first 10 rows):")
st.dataframe(df_raw.head(20))

st.write("Column names:")
st.code(list(df_raw.columns))

st.subheader("Data Cleaning and Preparation")
dup_count = df_raw.duplicated().sum()
st.write(f"Number of duplicate rows: {dup_count}")

missing = df_raw.isna().sum()
missing_nonzero = missing[missing > 0]
if missing_nonzero.empty:
    st.success("No missing values found in any column.")
else:
    st.dataframe(missing_nonzero)



st.write("Data types (raw):")
st.dataframe(df_raw.dtypes.astype(str))
st.write("Summary statistics (numeric columns):")
st.dataframe(df_raw.describe())

st.write("Removing non-informative or out-of-scope columns for behavioral analysis.")

columns_to_drop = ["Customer_ID", "Marital_Status"]
df_clean = df_raw.drop(columns=columns_to_drop, errors="ignore")

st.write(f"Columns removed: {columns_to_drop}")
st.write(f"Cleaned dataset shape: Rows: {df_clean.shape[0]:,} | Columns: {df_clean.shape[1]:,}")

st.write("""
Customer_ID was removed because it is a unique identifier and does not represent customer behavior. 
Marital_Status was excluded because it is not directly related to credit card usage, transactions, 
or financial risk in the context of this project.

Demographic variables such as Age, Gender, and Education_Level were retained to allow for later 
interpretation of behavioral patterns across demographic groups, while the primary focus remains 
on transaction, credit usage, and risk-related features.
""")
st.write("Histogram of Credit Score")
fig, ax = plt.subplots()
sns.histplot(data=df_clean, x="Credit_Score", kde=True, ax=ax)
st.pyplot(fig)

st.write("Histogram of Annual_Income")
fig, ax = plt.subplots()
sns.histplot(data=df_clean, x="Annual_Income", kde=True, ax=ax)
st.pyplot(fig)

st.write("Histogram of Total_Spend_Last_Year")
fig, ax = plt.subplots()
sns.histplot(data=df_clean, x="Total_Spend_Last_Year", kde=True, ax=ax)
st.pyplot(fig)



st.write("Histogram of Credit_Utilization_Ratio")
fig, ax = plt.subplots()
sns.histplot(data=df_clean, x="Credit_Utilization_Ratio", kde=True, ax=ax)
st.pyplot(fig)

st.write("Histogram of Debt_To_Income_Ratio")
fig, ax = plt.subplots()
sns.histplot(data=df_clean, x="Debt_To_Income_Ratio", kde=True, ax=ax)
st.pyplot(fig)

st.write("""
The dataset shows clear variation in customer financial behavior. Credit scores are roughly normally
distributed around moderate risk levels, suggesting a mix of stable and vulnerable customers. In
contrast, income and spending are right-skewed, meaning a smaller group of customers accounts for
disproportionately high values. Credit utilization and debt-to-income ratios span a wide range, indicating
that risk-related behavior varies significantly across customers and is not driven by demographics alone.
""")


st.write("""Do defaulters show higher utilization and more extreme behavior?""")
st.write("Credit Utilization Ratio by Default Status")
fig, ax = plt.subplots()
sns.boxplot(data=df_clean, x="Defaulted", y="Credit_Utilization_Ratio", ax=ax)
st.pyplot(fig)
st.write("""
This boxplot shows that customers who default tend to have higher credit utilization ratios.
Defaulters exhibit both a higher median utilization and greater variability, suggesting more extreme
credit usage behavior. While this does not imply causation, the pattern indicates that elevated and
unstable credit utilization is strongly associated with default risk.
""")


st.write("Average Number of Late Payments by Employment Status")

fig, ax = plt.subplots()
sns.barplot(
    data=df_clean,
    x="Employment_Status",
    y="Number_of_Late_Payments",
    estimator="mean",
    ax=ax
)

ax.set_xlabel("Employment Status")
ax.set_ylabel("Average Number of Late Payments")

st.pyplot(fig)
st.write("""
The average number of late payments is similar across employment categories, with substantial overlap
between groups. This suggests that employment status alone does not meaningfully explain missed payment
behavior. Instead, financial risk appears to be driven more by how credit is used rather than employment
classification itself.
""")

st.write("Credit Score by Default Status")

fig, ax = plt.subplots()
sns.boxplot(data=df_clean, x="Defaulted", y="Credit_Score", ax=ax)
st.pyplot(fig)

st.write("""
Credit score distributions for defaulted and non-defaulted customers largely overlap, with nearly
identical medians and spreads. This indicates that credit score alone does not strongly differentiate
default risk in this dataset. The result reinforces the idea that dynamic behavioral variables, such as
credit utilization and payment history, provide more informative risk signals than a single aggregate
score.
""")


st.write("Late Payments vs Credit Utilization Ratio")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df_clean,
    x="Credit_Utilization_Ratio",
    y="Number_of_Late_Payments",
    alpha=0.4,
    ax=ax
)

ax.set_xlabel("Credit Utilization Ratio")
ax.set_ylabel("Number of Late Payments")

st.pyplot(fig)



st.write("""
This scatterplot shows the relationship between credit utilization and the number of late payments.
Although there is no strong linear trend, higher utilization levels are associated with greater
variability and higher maximum counts of late payments. Customers with low utilization tend to cluster
around fewer missed payments, while higher utilization corresponds to increased payment instability.
This supports the role of credit usage intensity as a behavioral risk indicator rather than a
deterministic predictor.
""")


st.write("Correlation Heatmap of Numeric Financial Variables")

numeric_cols = df_clean.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    numeric_cols.corr(),
    cmap="coolwarm",
    center=0,
    ax=ax
)

st.pyplot(fig)
st.write("""
The correlation heatmap reveals clear groupings among behavioral financial variables. Transaction
volume, transaction amounts, merchant diversity, and customer lifetime value form a strongly correlated
cluster, reflecting a shared dimension of customer engagement and spending intensity. In contrast,
credit score and demographic variables show weak relationships with most behavioral features and with
default status. Default itself does not strongly correlate with any single variable, suggesting that
financial risk emerges from combinations of behaviors rather than isolated metrics. These findings
motivate the use of multivariate and dimensionality reduction techniques in later modeling stages.
""")

st.markdown("""
**Data Sources:**
- Credit Card Customer Dataset (Kaggle): https://www.kaggle.com/datasets/aadarshvani/credit-card-dataset-comprehensive?resource=download
""")

st.subheader("Macroeconomic Context Data (World Bank API)")

st.write("""
To provide macroeconomic context for individual credit behavior, data was retrieved
using the World Bank API. Specifically, the indicator **Domestic credit to the private
sector (% of GDP)** was used to capture the overall credit environment.
""")

st.write("World Bank API endpoint used:")
st.code("https://api.worldbank.org/v2/country/USA/indicator/FS.AST.PRVT.GD.ZS")

st.dataframe(df_world_bank.head())

st.write("World Bank Trend: Domestic Credit to Private Sector (% of GDP)")

df_plot = df_world_bank.copy()
df_plot["year"] = df_plot["year"].astype(int)
df_plot = df_plot.sort_values("year")

fig, ax = plt.subplots()
ax.plot(df_plot["year"], df_plot["credit_private_sector_pct_gdp"])
ax.set_xlabel("Year")
ax.set_ylabel("Credit to Private Sector (% of GDP)")
st.pyplot(fig)
st.write("""
This World Bank indicator is basically a “how much credit exists in the economy” measure.
The U.S. values are very high (often near or above 200% of GDP), which reinforces that borrowing
and credit usage are normal and widespread at the macro level. I’m not merging this directly with
the customer dataset, but I’m using it as context: individual default risk and credit behavior are
happening inside a country where credit access and lending are structurally large.
""")


st.subheader("Code Repository")

st.markdown("""
All data collection, cleaning, and analysis code for this project is available
on GitHub:

https://github.com/Kay-Abdi/credit-behavior-ml-project
""")
