import streamlit as st

st.title("Introduction")

st.write("""
Customer financial behavior plays a central role in shaping financial stability and risk in modern economies. Beyond income level or wealth, the ways individuals spend, borrow, save, and repay money influence their long-term financial outcomes. Everyday decisions such as relying on credit, paying balances on time, or maintaining consistent spending habits can accumulate into patterns that either support financial resilience or increase vulnerability. As financial systems become increasingly digitized, these behavioral patterns are recorded at a large scale, making it possible to examine how routine financial actions relate to broader measures of risk. Understanding financial behavior is therefore important not only for financial institutions and policymakers, but also for consumers navigating increasingly complex financial environments.
""")

st.write("""
Financial risk is not evenly distributed, even among individuals with similar economic resources. People with comparable incomes or employment situations often experience very different financial outcomes due to differences in behavior, habits, and decision-making. Prior research in economics and consumer finance has shown that behavioral factors such as payment consistency, spending regularity, and credit usage can be strong indicators of financial stress or stability. Rather than viewing financial risk as a binary outcome, this topic approaches risk as something that emerges from patterns of behavior over time. By examining customer financial behavior at a broader level, this project seeks to identify common behavioral profiles and explore how those profiles relate to differing levels of financial risk.
""")

st.write("""
To guide this exploration, the following questions focus on identifying patterns in financial behavior and understanding how those patterns may be associated with financial vulnerability or resilience. These questions are exploratory in nature and are intended to evolve as the project develops, serving as a foundation for investigating how financial behaviors cluster, interact, and signal risk.
""")

st.subheader("Guiding Questions")
questions = [
    "Do customers naturally cluster into distinct groups based on financial behavior?",
    "What combinations of financial behaviors are most strongly associated with default risk?",
    "Are there identifiable behavioral profiles that distinguish higher-risk customers from lower-risk customers?",
    "Which behavioral variables contribute most to differentiating customer risk profiles?",
    "Do spending and credit usage patterns provide additional insight into risk beyond income and credit score?",
    "Are certain financial behaviors more informative when considered together rather than individually?",
    "How does credit utilization interact with payment behavior in relation to default status?",
    "Can multivariate behavioral patterns explain risk better than any single financial metric?",
    "What latent dimensions of financial behavior emerge through dimensionality reduction techniques?",
    "How can customer-level financial behavior data be used to characterize financial vulnerability and stability?"
]

for i, q in enumerate(questions, start=1):
    st.write(f"{i}. {q}")

st.image(
    "https://images.unsplash.com/photo-1554224155-6726b3ff858f",
    caption="Everyday financial decisions shape long-term financial risk and stability.",
    width=700
)

