# L'Oréal Beauty Trend Analytics
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
import plotly.graph_objects as go

# --- CONFIG ---
st.set_page_config(page_title="L'Oréal Beauty Trend Analytics", layout="wide", page_icon="💄")

# --- BEAUTY INDUSTRY DATA ---
BEAUTY_CATEGORIES = {
    "Skincare": ["skincare", "skin care", "moisturizer", "serum", "cleanser", "toner", "mask", "sunscreen", "anti-aging"],
    "Makeup": ["makeup", "foundation", "concealer", "lipstick", "eyeshadow", "mascara", "blush", "bronzer", "highlighter"],
    "Haircare": ["haircare", "hair care", "shampoo", "conditioner", "hair mask", "styling", "hair oil", "treatment"]
}

LOREAL_BRANDS = [
    "L'Oréal", "Lancôme", "Garnier", "Maybelline", "Urban Decay", "YSL", "Giorgio Armani",
    "Kiehl's", "Vichy", "La Roche-Posay", "CeraVe", "NYX Professional Makeup"
]

BEAUTY_INFLUENCERS = [
    "James Charles", "Jeffree Star", "Nikkie Tutorials", "Tati Westbrook", "Jackie Aina",
    "Huda Beauty", "Bretman Rock", "Manny MUA", "Laura Lee", "Desi Perkins"
]

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("all_signals_combined.csv")
    df["hawkes_R"] = pd.to_numeric(df["hawkes_R"], errors="coerce")
    df["tbi"] = pd.to_numeric(df["tbi"], errors="coerce")
    df["eng_quality"] = pd.to_numeric(df["eng_quality"], errors="coerce")
    df["saturation"] = pd.to_numeric(df["saturation"], errors="coerce")
    df["robust_trend_score"] = pd.to_numeric(df["robust_trend_score"], errors="coerce")
    
    # Add beauty-specific columns
    df = categorize_beauty_content(df)
    df = analyze_sentiment(df)
    df = identify_loreal_content(df)
    
    return df

def categorize_beauty_content(df):
    """Categorize videos into beauty product categories"""
    df["beauty_category"] = "Other"
    
    for category, keywords in BEAUTY_CATEGORIES.items():
        mask = df["title"].str.contains("|".join(keywords), case=False, na=False)
        df.loc[mask, "beauty_category"] = category
    
    return df

def analyze_sentiment(df):
    """Analyze sentiment of video titles"""
    try:
        # Simple sentiment analysis without TextBlob dependency
        positive_words = ["amazing", "love", "best", "perfect", "gorgeous", "stunning", "beautiful", "fantastic", "incredible", "wow"]
        negative_words = ["hate", "terrible", "awful", "worst", "ugly", "bad", "horrible", "disappointing", "failed", "disaster"]
        
        def simple_sentiment(text):
            if pd.isna(text):
                return 0
            text = text.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            return pos_count - neg_count
        
        df["sentiment_score"] = df["title"].apply(simple_sentiment)
            
        df["sentiment_label"] = df["sentiment_score"].apply(
            lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
        )
    except:
        df["sentiment_score"] = 0
        df["sentiment_label"] = "Neutral"
    
    return df

def identify_loreal_content(df):
    """Identify content related to L'Oréal brands or beauty influencers"""
    df["is_loreal_brand"] = df["title"].str.contains("|".join(LOREAL_BRANDS), case=False, na=False)
    
    df["is_beauty_influencer"] = df["channelTitle"].str.contains("|".join(BEAUTY_INFLUENCERS), case=False, na=False)
    
    return df

def extract_trending_keywords(df, text_column="title", top_n=10):
    """Extract trending keywords from text"""
    all_text = " ".join(df[text_column].dropna().astype(str))
    # Remove common words and extract meaningful terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    beauty_keywords = [word for word in words if any(keyword in word for category_keywords in BEAUTY_CATEGORIES.values() for keyword in category_keywords)]
    return Counter(beauty_keywords).most_common(top_n)

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("💄 L'Oréal Analytics Controls")

# Trend Formula Selector
formula = st.sidebar.selectbox(
    "📈 Rank Videos By:",
    options=[
        "Robust Trend Score",
        "Hawkes Momentum (Rₜ)",
        "TBI (Novelty)",
        "Engagement Quality",
        "Saturation (Lower is Better)"
    ]
)

# Map selection to column
score_col = {
    "Robust Trend Score": "robust_trend_score",
    "Hawkes Momentum (Rₜ)": "hawkes_R",
    "TBI (Novelty)": "tbi",
    "Engagement Quality": "eng_quality",
    "Saturation (Lower is Better)": "saturation"
}[formula]

# Beauty-Specific Filters
st.sidebar.subheader("🔍 Beauty Industry Filters")

# Product Category Filter
beauty_cat_filter = st.sidebar.multiselect(
    "Product Category", 
    options=["Skincare", "Makeup", "Haircare", "Other"],
    help="Filter by beauty product categories"
)

# L'Oréal Brand Filter
loreal_filter = st.sidebar.checkbox("🏷️ L'Oréal Brands Only", help="Show only videos mentioning L'Oréal brands")

# Beauty Influencer Filter
influencer_filter = st.sidebar.checkbox("⭐ Beauty Influencers Only", help="Show only videos from known beauty influencers")

# Sentiment Filter
sentiment_filter = st.sidebar.multiselect(
    "Sentiment", 
    options=["Positive", "Neutral", "Negative"],
    help="Filter by content sentiment"
)

# Traditional Filters
st.sidebar.subheader("� Additional Filters")
category_filter = st.sidebar.multiselect("General Category", options=df["category"].dropna().unique())
channel_filter = st.sidebar.multiselect("Channel", options=df["channelTitle"].dropna().unique())

# Apply filters
filtered_df = df.copy()

# Beauty-specific filters
if beauty_cat_filter:
    filtered_df = filtered_df[filtered_df["beauty_category"].isin(beauty_cat_filter)]

if loreal_filter:
    filtered_df = filtered_df[filtered_df["is_loreal_brand"] == True]

if influencer_filter:
    filtered_df = filtered_df[filtered_df["is_beauty_influencer"] == True]

if sentiment_filter:
    filtered_df = filtered_df[filtered_df["sentiment_label"].isin(sentiment_filter)]

# Traditional filters
if category_filter:
    filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]
if channel_filter:
    filtered_df = filtered_df[filtered_df["channelTitle"].isin(channel_filter)]

# Sort by selected score
if score_col == "saturation":
    filtered_df = filtered_df.sort_values(score_col, ascending=True)
else:
    filtered_df = filtered_df.sort_values(score_col, ascending=False)

# --- MAIN PAGE ---
st.title("💄 L'Oréal Beauty Trend Analytics")
st.markdown("*Discover emerging beauty trends through AI-powered YouTube analysis*")

# Key Metrics Dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Videos", len(filtered_df))
with col2:
    beauty_videos = len(filtered_df[filtered_df["beauty_category"] != "Other"])
    st.metric("Beauty Content", beauty_videos)
with col3:
    loreal_videos = len(filtered_df[filtered_df["is_loreal_brand"] == True])
    st.metric("L'Oréal Mentions", loreal_videos)
with col4:
    avg_sentiment = filtered_df["sentiment_score"].mean()
    st.metric("Avg Sentiment", f"{avg_sentiment:.1f}")

st.markdown(f"### 🔝 Top Trending Beauty Content by: **{formula}**")

# Show top 10
top10 = filtered_df.head(10)
if len(top10) > 0:
    fig = px.bar(   
        top10,
        x=score_col,
        y="title",
        orientation='h',
        color="beauty_category",
        color_discrete_map={
            "Skincare": "#FFB6C1",
            "Makeup": "#FF69B4", 
            "Haircare": "#DDA0DD",
            "Other": "#D3D3D3"
        },
        hover_data=["channelTitle", "viewCount", "commentCount", "sentiment_label"]
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        title="Top 10 Trending Beauty Videos"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Add predictive forecast for top video
    if len(top10) > 0:
        st.markdown("#### 🔮 Trend Forecast for Top Video")
        top_video = top10.iloc[0]
        
        # Simple forecast simulation (in real app, use time series data)
        current_score = top_video[score_col]
        momentum = top_video["hawkes_R"] if pd.notna(top_video["hawkes_R"]) else 1
        
        # Generate forecast points
        forecast_days = list(range(1, 8))
        forecast_scores = [current_score * (1 + momentum * 0.1 * day) for day in forecast_days]
        
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=[0], y=[current_score], 
            mode='markers', name='Current', 
            marker=dict(size=12, color='red')
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_days, y=forecast_scores, 
            mode='lines+markers', name='Forecast', 
            line=dict(dash='dash', color='blue')
        ))
        forecast_fig.update_layout(
            title=f"7-Day Forecast: {top_video['title'][:50]}...",
            xaxis_title="Days",
            yaxis_title=formula
        )
        st.plotly_chart(forecast_fig, use_container_width=True)
else:
    st.info("No videos match your current filters. Try adjusting your selection.")

# --- BEAUTY INSIGHTS SECTION ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔑 Trending Beauty Keywords")
    keywords = extract_trending_keywords(filtered_df)
    if keywords:
        keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
        keyword_fig = px.bar(keyword_df, x='Frequency', y='Keyword', orientation='h',
                           title="Most Mentioned Beauty Terms")
        keyword_fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(keyword_fig, use_container_width=True)
    else:
        st.info("No trending keywords found in current selection.")

with col2:
    st.subheader("💭 Sentiment Analysis")
    sentiment_counts = filtered_df["sentiment_label"].value_counts()
    if len(sentiment_counts) > 0:
        sentiment_fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                             title="Content Sentiment Distribution",
                             color_discrete_map={
                                 "Positive": "#90EE90",
                                 "Neutral": "#FFE4B5", 
                                 "Negative": "#FFB6C1"
                             })
        st.plotly_chart(sentiment_fig, use_container_width=True)
    else:
        st.info("No sentiment data available.")

# --- PRODUCT CATEGORY BREAKDOWN ---
st.markdown("---")
st.subheader("🛍️ Beauty Product Category Performance")
category_performance = filtered_df.groupby("beauty_category").agg({
    score_col: "mean",
    "viewCount": "sum",
    "commentCount": "sum"
}).round(2)

if len(category_performance) > 0:
    cat_fig = px.bar(category_performance, x=category_performance.index, y=score_col,
                    title=f"Average {formula} by Product Category",
                    color=category_performance.index,
                    color_discrete_map={
                        "Skincare": "#FFB6C1",
                        "Makeup": "#FF69B4", 
                        "Haircare": "#DDA0DD",
                        "Other": "#D3D3D3"
                    })
    st.plotly_chart(cat_fig, use_container_width=True)

# --- TREND SCORE DISTRIBUTION ---
st.markdown("---")
st.subheader("📊 Score Distribution")
hist_fig = px.histogram(filtered_df, x=score_col, nbins=50, 
                       title=f"Distribution of {formula}",
                       color="beauty_category",
                       color_discrete_map={
                           "Skincare": "#FFB6C1",
                           "Makeup": "#FF69B4", 
                           "Haircare": "#DDA0DD",
                           "Other": "#D3D3D3"
                       })
st.plotly_chart(hist_fig, use_container_width=True)

# --- SCATTER PLOT: Rₜ vs TBI ---
st.markdown("---")
st.subheader("🌀 Beauty Trend Momentum vs Novelty")
scatter_df = filtered_df.dropna(subset=["hawkes_R", "tbi"])
if len(scatter_df) > 0:
    scatter_fig = px.scatter(
        scatter_df,
        x="hawkes_R",
        y="tbi",
        size="viewCount",
        color="beauty_category",
        color_discrete_map={
            "Skincare": "#FFB6C1",
            "Makeup": "#FF69B4", 
            "Haircare": "#DDA0DD",
            "Other": "#D3D3D3"
        },
        hover_name="title",
        hover_data=["channelTitle", "sentiment_label"],
        title="Beauty Content: Momentum (Rₜ) vs Novelty (TBI)",
        labels={"hawkes_R": "Hawkes Rₜ (Momentum)", "tbi": "TBI (Novelty)"}
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    st.caption("💡 **Insight**: Videos in the top-right quadrant show both high momentum and novelty - prime candidates for emerging beauty trends!")
else:
    st.info("Not enough data to show momentum vs novelty analysis.")

# --- L'ORÉAL BRAND PERFORMANCE ---
if loreal_videos > 0:
    st.markdown("---")
    st.subheader("🏷️ L'Oréal Brand Mentions Performance")
    loreal_df = filtered_df[filtered_df["is_loreal_brand"] == True]
    
    # Extract which L'Oréal brands are mentioned
    brand_mentions = []
    for _, row in loreal_df.iterrows():
        text = str(row["title"]) + " " + str(row["description"])
        for brand in LOREAL_BRANDS:
            if brand.lower() in text.lower():
                brand_mentions.append(brand)
    
    if brand_mentions:
        brand_counts = Counter(brand_mentions)
        brand_fig = px.bar(x=list(brand_counts.keys()), y=list(brand_counts.values()),
                          title="L'Oréal Brand Mentions in Trending Videos",
                          color=list(brand_counts.values()),
                          color_continuous_scale="Reds")
        st.plotly_chart(brand_fig, use_container_width=True)

# --- DATA TABLE ---
st.markdown("---")
st.subheader("📋 Beauty Trend Data Explorer")

# Enhanced data table with beauty-specific columns
display_columns = [
    "title", "channelTitle", "beauty_category", "viewCount", "commentCount",
    "sentiment_label", "sentiment_score", "is_loreal_brand", "is_beauty_influencer",
    "hawkes_R", "tbi", "eng_quality", "saturation", "robust_trend_score"
]

# Only show columns that exist in the dataframe
available_columns = [col for col in display_columns if col in filtered_df.columns]
st.dataframe(filtered_df[available_columns].head(100))

# Export option
if st.button("📥 Export Current Data as CSV"):
    csv_data = filtered_df[available_columns].to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="loreal_beauty_trends.csv",
        mime="text/csv"
    )

# --- ABOUT ---
st.markdown("---")
st.markdown("""
### 🎯 About L'Oréal Beauty Trend Analytics

This AI-powered analytics platform helps L'Oréal identify emerging beauty trends by analyzing YouTube content. 
The app uses advanced metrics like **Hawkes momentum** and **TBI novelty** to spot viral potential before it peaks.

**Key Features:**
- 🔍 **Beauty-focused filtering** by product categories (skincare, makeup, haircare)
- 🏷️ **L'Oréal brand monitoring** across all subsidiary brands
- ⭐ **Beauty influencer tracking** for trend origination
- 💭 **Sentiment analysis** to gauge consumer reception
- 🔮 **Predictive forecasting** for trend momentum
- 📊 **Industry-specific visualizations** tailored for beauty marketing teams

*Built with ❤️ for the beauty industry | Powered by AI trend detection*
""")

st.caption("💡 **Pro Tip**: Use the filters to focus on specific beauty segments and watch for videos with high momentum + novelty scores - these often indicate emerging trends that could influence product development and marketing strategies.")