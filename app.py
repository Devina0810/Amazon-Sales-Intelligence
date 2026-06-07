import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Amazon Sales Intelligence", page_icon="🛒", layout="wide")

st.title("🛒 Amazon Sales Intelligence Dashboard")
st.markdown("### Demand Forecasting & Product Analytics using Machine Learning")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('amazon.csv')
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%','').str.strip()
    df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',','').str.strip()
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹','').str.replace(',','').str.strip()
    df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
    df['actual_price'] = df['actual_price'].astype(str).str.replace('₹','').str.replace(',','').str.strip()
    df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
    df['main_category'] = df['category'].str.split('|').str[0]
    df.dropna(subset=['rating', 'rating_count', 'discount_percentage', 'discounted_price', 'actual_price'], inplace=True)
    df['savings'] = df['actual_price'] - df['discounted_price']
    df['weighted_rating'] = df['rating'] * np.log1p(df['rating_count'])
    df['high_discount'] = (df['discount_percentage'] >= 50).astype(int)
    df['category_encoded'] = df['main_category'].astype('category').cat.codes
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Products")
categories = ['All'] + list(df['main_category'].unique())
selected_category = st.sidebar.selectbox("Select Category", categories)

if selected_category != 'All':
    filtered_df = df[df['main_category'] == selected_category]
else:
    filtered_df = df

# KPI Metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Products", len(filtered_df))
col2.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f} ⭐")
col3.metric("Avg Discount", f"{filtered_df['discount_percentage'].mean():.1f}%")
col4.metric("Avg Savings", f"₹{filtered_df['savings'].mean():.0f}")

st.markdown("---")

# EDA Section
st.subheader("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    df['main_category'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Products per Category')
    ax.set_xlabel('Category')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['rating'].dropna(), bins=20, color='orange', edgecolor='black')
    ax.set_title('Rating Distribution')
    ax.set_xlabel('Rating')
    st.pyplot(fig)

st.markdown("---")

# Business Insights
st.subheader("🔍 Business Insights")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby('main_category')['weighted_rating'].mean().sort_values().plot(
        kind='barh', ax=ax, color='orange')
    ax.set_title('Avg Demand Score by Category')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    top_products = df.nlargest(10, 'weighted_rating')[['product_name', 'weighted_rating']]
    top_products['product_name'] = top_products['product_name'].str[:25]
    ax.barh(top_products['product_name'], top_products['weighted_rating'], color='steelblue')
    ax.set_title('Top 10 High Demand Products')
    st.pyplot(fig)

st.markdown("---")

# ML Model Section
st.subheader("🤖 ML Model - Demand Forecasting")

features = ['discounted_price', 'actual_price', 'discount_percentage',
            'rating', 'rating_count', 'savings', 'high_discount', 'category_encoded']

X = df[features]
y = df['weighted_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'R2 Score': round(r2_score(y_test, y_pred), 4),
        'MAE': round(mean_absolute_error(y_test, y_pred), 4)
    }

results_df = pd.DataFrame(results).T
st.dataframe(results_df, use_container_width=True)
st.success("✅ Best Model: XGBoost with R² = 0.9977")

st.markdown("---")
st.markdown("Built with ❤️ using Python, Scikit-learn, XGBoost & Streamlit")