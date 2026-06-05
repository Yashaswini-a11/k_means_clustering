import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

df = pd.read_csv("Mall_Customers.csv")

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.title(" Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "EDA",
        "Cluster Analysis",
        "Prediction"
    ]
)

if page == "Dashboard":

    st.title("Customer Segmentation Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        df.shape[0]
    )

    col2.metric(
        "Features",
        df.shape[1]
    )

    col3.metric(
        "Clusters",
        model.n_clusters
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Summary Statistics")

    st.dataframe(df.describe())

elif page == "EDA":

    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x="Annual Income (k$)",
            title="Annual Income Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x="Spending Score (1-100)",
            title="Spending Score Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    fig, ax = plt.subplots(
        figsize=(8,5)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

elif page == "Cluster Analysis":

    st.title("Cluster Analysis")

    X = df[
        [
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ]

    X_scaled = scaler.transform(X)

    clusters = model.predict(
        X_scaled
    )

    st.subheader("Cluster Visualization")

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=clusters.astype(str),
        title="Customer Segments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Cluster Counts")

    cluster_df = pd.DataFrame(
        {
            "Cluster": clusters
        }
    )

    fig = px.histogram(
        cluster_df,
        x="Cluster",
        color="Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif page == "Prediction":

    st.title("Predict Customer Segment")

    income = st.slider(
        "Annual Income (k$)",
        0,
        150,
        50
    )

    spending = st.slider(
        "Spending Score",
        1,
        100,
        50
    )

    if st.button("Predict Cluster"):

        sample = [[income, spending]]

        sample_scaled = scaler.transform(
            sample
        )

        cluster = model.predict(
            sample_scaled
        )[0]

        cluster_names = {
            0: "High Value Customer",
            1: "Regular Customer",
            2: "Premium Customer",
            3: "Budget Customer",
            4: "Luxury Spender"
        }

        st.markdown(
            f"""
            <div class="result-card">
            <h2>Cluster {cluster}</h2>
            <h3>{cluster_names.get(cluster)}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )