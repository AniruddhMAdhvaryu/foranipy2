import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Anime Analysis Dashboard", layout="wide")

# Title
st.title("Anime Analysis Dashboard")

# Read the data
@st.cache_data
def load_data():
    df = pd.read_csv("anime.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.header("Dashboard Controls")
show_raw_data = st.sidebar.checkbox("Show Raw Data")

if show_raw_data:
    st.subheader("Raw Data")
    st.dataframe(df)

# Key Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Anime", len(df))
with col2:
    st.metric("Average Rating", f"{df['rating'].mean():.2f}")
with col3:
    st.metric("Average Episodes", f"{pd.to_numeric(df['episodes'], errors='coerce').mean():.1f}")

# Tabs for analysis
tab1, tab2, tab3 = st.tabs(["Distribution Plots", "Relationship Plots", "Statistical Analysis"])

with tab1:
    st.subheader("Genre Distribution")
    genre_counts = df["genre"].str.split(", ").explode().value_counts()
    fig1 = px.bar(
        genre_counts,
        x=genre_counts.index,
        y=genre_counts.values,
        labels={"x": "Genre", "y": "Count"},
        title="Most Common Genres",
    )
    st.plotly_chart(fig1)

    st.subheader("Type Distribution")
    type_counts = df["type"].value_counts()
    fig2 = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Anime Types Distribution",
    )
    st.plotly_chart(fig2)

with tab2:
    st.subheader("Rating vs Members")
    fig3 = px.scatter(
        df,
        x="rating",
        y="members",
        size="members",
        color="type",
        hover_data=["name"],
        labels={"rating": "Rating", "members": "Members"},
        title="Rating vs Members",
    )
    st.plotly_chart(fig3)

    st.subheader("Episodes vs Rating")
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')  # Handle non-numeric values
    fig4 = px.box(
        df,
        x="type",
        y="rating",
        color="type",
        points="all",
        labels={"rating": "Rating", "type": "Anime Type"},
        title="Rating Distribution by Type",
    )
    st.plotly_chart(fig4)

with tab3:
    st.subheader("Statistical Summary")
    stats = df[["rating", "members"]].describe().T
    st.write(stats)

    st.subheader("Filtered Analysis")
    min_rating, max_rating = st.slider(
        "Select Rating Range",
        min_value=float(df["rating"].min()),
        max_value=float(df["rating"].max()),
        value=(8.0, 10.0),
    )
    filtered_df = df[(df["rating"] >= min_rating) & (df["rating"] <= max_rating)]
    st.write(f"Filtered Results ({len(filtered_df)} anime):")
    st.dataframe(filtered_df)

# Footer
st.markdown("---")
st.markdown(
    """
    *Data Dictionary:*
    - anime_id: Unique identifier for each anime.
    - name: Name of the anime.
    - genre: Genre of the anime.
    - type: Type of anime (e.g., TV, Movie).
    - episodes: Number of episodes.
    - rating: Average user rating.
    - members: Number of members in the community.
    """
)
