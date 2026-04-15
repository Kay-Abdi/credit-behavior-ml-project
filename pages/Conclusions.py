import streamlit as st

st.title("Conclusions")

st.write("""
This project set out to understand whether customer financial behavior can help explain and predict financial risk, particularly the likelihood of default. Across all of the methods used, a clear takeaway emerged: financial behavior does matter. Patterns in spending, credit usage, repayment habits, and overall financial activity are not random. They form consistent patterns that can be used to better understand differences between higher-risk and lower-risk customers.
""")

st.write("""
One of the strongest findings is that customers do not all behave the same. When the data was grouped and explored, distinct behavioral patterns began to appear. Some customers showed stable and consistent financial habits, while others showed patterns that were more irregular or risky. These differences were reflected in how likely each group was to default. This suggests that financial risk is not evenly spread across individuals, but instead tends to concentrate within certain types of behavior.
""")

st.write("""
Another key takeaway is that risk is not driven by just one variable. While certain features, like credit utilization, stood out as especially important, no single factor fully explained default risk on its own. Instead, it was the combination of multiple behaviors that made the biggest difference. Models that were able to capture relationships between variables performed better than those that treated each feature independently. This shows that financial risk comes from how behaviors interact, not from one isolated metric.
""")

st.write("""
At the same time, the results also showed that predicting financial risk is not easy. While all of the models were able to find patterns, none of them achieved extremely high accuracy. There was still overlap between higher-risk and lower-risk customers, meaning that some individuals behave similarly but experience different outcomes. This highlights an important limitation: even though behavior provides useful insight, it does not fully determine financial outcomes on its own.
""")

st.write("""
Comparing different models helped reinforce this idea. Simpler models were able to capture general patterns, while more flexible models were able to explore more complex relationships. However, increasing model complexity did not always lead to better results. In many cases, more advanced approaches only provided small improvements. This suggests that the limitation is not just the model itself, but the nature of the data and the complexity of human financial behavior.
""")

st.write("""
Overall, this project shows that customer financial behavior can be used to better understand financial risk, but it should not be viewed as a perfect predictor. Instead, it provides meaningful signals that, when combined, help identify patterns of vulnerability and stability. The most important insight is that financial risk is shaped by multiple interacting behaviors, and understanding those patterns is key to making better predictions and more informed decisions.
""")

st.image(
    "https://images.unsplash.com/photo-1554224155-6726b3ff858f",
    caption="Financial behavior patterns help explain differences in risk, but no single factor tells the whole story.",
    width=700
)

st.write("""
From a broader perspective, these findings suggest that financial institutions, policymakers, and even individuals can benefit from focusing more on behavior rather than just static metrics. Looking at how people actually use credit and manage their finances over time provides a deeper understanding of risk than relying on a single score or number. By recognizing patterns in behavior, it becomes possible to better identify both risk and long-term financial stability.
""")

st.image(
    "https://images.unsplash.com/photo-1556740749-887f6717d7e4",
    caption="Understanding everyday financial decisions is key to understanding long-term financial outcomes.",
    width=700
)