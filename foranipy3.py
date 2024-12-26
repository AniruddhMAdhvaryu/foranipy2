import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

# Title
st.title("Heart Disease Analysis - A New Perspective")

# Read the data
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.header("Explore Data")
show_raw_data = st.sidebar.checkbox("Show Raw Data")

if show_raw_data:
    st.subheader("Raw Data")
    st.dataframe(df)

# Key Metrics in columns with a new style
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Patients", len(df))
with col2:
    st.metric("Heart Disease Prevalence", f"{(df['target'].mean() * 100):.2f}%")
with col3:
    st.metric("Average Age", f"{df['age'].mean():.1f} years")
with col4:
    gender_ratio = len(df[df['sex'] == 1]) / len(df[df['sex'] == 0])
    st.metric("Male to Female Ratio", f"{gender_ratio:.2f}")

# New tab layout with a more engaging approach
tab1, tab2, tab3 = st.tabs(["Patient Distribution", "Feature Relationships", "Advanced Insights"])

with tab1:
    st.subheader("Distribution of Patients")
    
    # Gender Distribution Pie Chart
    gender_counts = df['sex'].map({0: "Female", 1: "Male"}).value_counts()
    fig1 = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution", color_discrete_sequence=["#FF7F50", "#1E90FF"])
    st.plotly_chart(fig1, use_container_width=True)
    
    # Age Distribution by Disease Status
    fig2 = px.histogram(df, x="age", color="target", marginal="rug", title="Age Distribution by Disease Status", labels={"target": "Heart Disease"})
    st.plotly_chart(fig2, use_container_width=True)

    # Chest Pain Type Distribution
    fig3 = px.histogram(df, x="cp", color="target", barmode="group", title="Chest Pain Types by Disease Status")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Feature Relationships")
    
    # Heart Rate vs Age Scatter Plot
    fig4 = px.scatter(df, x="age", y="thalach", color="target", size="chol", hover_data=["sex", "cp"], title="Heart Rate vs Age (Cholesterol Size)")
    st.plotly_chart(fig4, use_container_width=True)

    # Correlation Heatmap using Plotly
    corr_matrix = df.corr()
    fig5 = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1))
    fig5.update_layout(title="Feature Correlation Matrix", xaxis_title="Features", yaxis_title="Features")
    st.plotly_chart(fig5, use_container_width=True)

    # Violin Plot for Feature Distribution by Disease Status
    features = ["age", "thalach", "chol", "trestbps"]
    fig6 = make_subplots(rows=2, cols=2, subplot_titles=features)

    for i, feature in enumerate(features):
        row = i // 2 + 1
        col = i % 2 + 1
        fig6.add_trace(go.Violin(x=df[df['target'] == 0][feature], name="No Disease", side="negative", line_color="green"), row=row, col=col)
        fig6.add_trace(go.Violin(x=df[df['target'] == 1][feature], name="Disease", side="positive", line_color="red"), row=row, col=col)
        fig6.update_xaxes(title_text=feature.title(), row=row, col=col)
        fig6.update_yaxes(title_text="Frequency", row=row, col=col)

    fig6.update_layout(height=800, title="Feature Distribution by Heart Disease Status")
    st.plotly_chart(fig6, use_container_width=True)

with tab3:
    st.subheader("Statistical Insights")

    # Summary statistics by target status with a clean design
    summary = df.groupby("target").agg({
        "age": ["mean", "std", "min", "max"],
        "thalach": ["mean", "std"],
        "chol": ["mean", "std"],
        "trestbps": ["mean", "std"]
    }).round(2)
    
    # Rename columns for better readability
    summary.columns = [f"{col[0]}_{col[1]}".replace("_", " ").title() for col in summary.columns]
    summary.index = ["No Heart Disease", "Heart Disease"]
    st.dataframe(summary)

    # Filtered Analysis
    st.subheader("Filter Data for Custom Analysis")
    
    # Sliders and filters for user input
    age_range = st.slider("Select Age Range", min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=(30, 70))
    gender = st.selectbox("Select Gender", options=["Both", "Female", "Male"], index=0)

    if gender == "Female":
        filtered_df = df[(df['age'].between(age_range[0], age_range[1])) & (df['sex'] == 0)]
    elif gender == "Male":
        filtered_df = df[(df['age'].between(age_range[0], age_range[1])) & (df['sex'] == 1)]
    else:
        filtered_df = df[df['age'].between(age_range[0], age_range[1])]

    st.write(f"Filtered Data: {len(filtered_df)} patients")
    st.dataframe(filtered_df)

# Footer
st.markdown("---")
st.markdown(
    """
    **Data Dictionary:**
    - target: Heart disease presence (1: Yes, 0: No)
    - sex: Gender (1: Male, 0: Female)
    - cp: Chest pain type
    - thalach: Maximum heart rate achieved
    - chol: Cholesterol level
    """
)

