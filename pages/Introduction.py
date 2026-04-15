import streamlit as st

st.title("Introduction")

st.write("""
Customer financial behavior plays a central role in shaping financial stability and risk in modern economies. Beyond income level or wealth, the ways individuals spend, borrow, save, and repay money influence their long-term financial outcomes. Everyday decisions such as relying on credit, paying balances on time, or maintaining consistent spending habits can accumulate into patterns that either support financial resilience or increase vulnerability. As financial systems become increasingly digitized, these behavioral patterns are recorded at scale, making it possible to examine how routine financial actions relate to broader measures of risk.
""")

st.write("""
One of the key challenges in understanding financial risk is that it is not evenly distributed, even among individuals with similar economic resources. People with comparable incomes or credit scores can experience very different financial outcomes depending on how they manage their money. This suggests that financial risk is not driven by a single factor, but instead emerges from combinations of behaviors over time. Rather than treating risk as a simple yes-or-no outcome, this project approaches it as something that develops through patterns in spending, credit usage, and repayment behavior.
""")

st.write("""
To explore this idea, this project analyzes a credit card customer dataset and applies a range of data analysis and machine learning methods. These include exploratory data analysis to understand overall trends, dimensionality reduction to simplify complex relationships, clustering to identify natural groupings in behavior, and classification models to predict default risk. Each method provides a different perspective, allowing the analysis to move beyond a single model and instead build a more complete understanding of how financial behavior relates to risk.
""")

st.write("""
Across these methods, a consistent theme emerges: financial behavior does contain meaningful patterns, but those patterns are not always cleanly separated or easy to predict. Some models capture general trends well but struggle with higher-risk cases, while others provide more flexibility but do not significantly improve accuracy. This reflects the reality that financial behavior is complex, overlapping, and influenced by multiple interacting factors rather than one dominant variable.
""")

st.write("""
At the same time, the goal of this project is not only to identify financial risk, but also to better understand stability. Many individuals demonstrate consistent and controlled financial habits that allow them to manage credit effectively and avoid default. By comparing both higher-risk and lower-risk patterns, this project highlights what contributes to financial vulnerability as well as what supports long-term financial health. This balanced perspective allows for a deeper understanding of how everyday financial decisions shape outcomes over time.
""")

st.write("""
To guide this exploration, the following questions focus on identifying patterns in financial behavior and understanding how those patterns relate to financial risk. These questions serve as a foundation for analyzing how behaviors cluster, interact, and contribute to different levels of financial stability.
""")

st.image(
    "https://images.unsplash.com/photo-1556740749-887f6717d7e4",
    caption="Financial behavior is shaped by everyday decisions around spending, borrowing, and repayment.",
    width=700
)

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
    "What underlying patterns in financial behavior emerge when the data is simplified or grouped?",
    "How can customer-level financial behavior data be used to better understand financial stability and risk?"
]

for i, q in enumerate(questions, start=1):
    st.write(f"{i}. {q}")

st.image(
    "https://images.unsplash.com/photo-1554224155-6726b3ff858f",
    caption="Patterns in financial behavior help explain both risk and long-term stability.",
    width=700
)