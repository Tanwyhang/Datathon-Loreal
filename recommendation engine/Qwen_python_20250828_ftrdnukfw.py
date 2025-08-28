# beauty_innovation_engine.py

import pandas as pd
import numpy as np
from collections import Counter
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import requests
import json
import time
import os
from datetime import datetime, timedelta

# --- Configuration ---
# Replace <OPENROUTER_API_KEY> with your actual key
OPENROUTER_API_KEY = "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"
# Optional: Add your site URL and name for rankings
YOUR_SITE_URL = "" # e.g., "https://yourwebsite.com"
YOUR_SITE_NAME = "" # e.g., "Your Site Name"
# ---------------------

# --- File Paths (Update if paths differ) ---
FUSION_RESULTS_PATH = '/kaggle/input/fusion-engine/all_signals_combined.csv'
VIDEOS_PATH = '/kaggle/input/datathon-loreal/videos.csv'
COMMENTS_PATH = '/kaggle/input/data-cleaning/comments_enriched.parquet' # Optional

AMAZON_RATINGS_PATH = '/kaggle/input/amazon-ratings/ratings_Beauty.csv'
TOP_PRODUCTS_PATH = '/kaggle/input/most-used-beauty-cosmetics-products-in-the-world/most_used_beauty_cosmetics_products_extended.csv'
SUPPLY_CHAIN_PATH = '/kaggle/input/supply-chain-analysis/supply_chain_data.csv'
# ------------------------------------------

def forecast_product_revenue_and_margin(recommendations_df, sales_data_df, supply_chain_df=None):
    """
    Use machine learning to forecast revenue and margin projections for new product recommendations.
    Includes professional business metrics that companies need to see.
    """
    print("--- Building ML Revenue Forecasting Models ---")

    if sales_data_df.empty:
        print("No sales data available for forecasting. Using industry averages.")
        return _apply_industry_averages(recommendations_df)

    # Prepare training data from existing products
    training_data = _prepare_training_data(sales_data_df, supply_chain_df)

    if training_data.empty:
        print("Insufficient training data. Using industry averages.")
        return _apply_industry_averages(recommendations_df)

    # Build forecasting models
    revenue_model, margin_model = _build_forecasting_models(training_data)

    # Apply models to new recommendations
    forecasted_df = _apply_forecasting_models(recommendations_df, revenue_model, margin_model, training_data)

    # Add professional business metrics
    forecasted_df = _add_business_metrics(forecasted_df)

    print(f"Successfully forecasted {len(forecasted_df)} products with ML models")
    return forecasted_df

def _prepare_training_data(sales_data_df, supply_chain_df=None):
    """Prepare training data for ML models from existing sales data."""
    training_features = []

    # Basic product features
    if 'Rating' in sales_data_df.columns:
        sales_data_df['Rating'] = pd.to_numeric(sales_data_df['Rating'], errors='coerce')
    if 'Number_of_Reviews' in sales_data_df.columns:
        sales_data_df['Number_of_Reviews'] = pd.to_numeric(sales_data_df['Number_of_Reviews'], errors='coerce')

    # Create synthetic revenue and margin data for training
    # In a real scenario, this would come from actual sales data
    np.random.seed(42)  # For reproducible results

    for idx, row in sales_data_df.iterrows():
        if pd.isna(row.get('Rating', 0)) or pd.isna(row.get('Number_of_Reviews', 0)):
            continue

        # Estimate revenue based on rating and review volume
        base_revenue = row['Rating'] * row['Number_of_Reviews'] * np.random.uniform(10, 50)

        # Estimate margin based on category and brand strength
        category_factor = 1.0
        if 'Category' in row and pd.notna(row['Category']):
            category_factor = len(str(row['Category'])) / 20  # Simple category complexity factor

        brand_factor = 1.0
        if 'Brand' in row and pd.notna(row['Brand']):
            brand_factor = len(str(row['Brand'])) / 15  # Brand name length as proxy for brand strength

        margin_pct = 0.3 + (category_factor * 0.2) + (brand_factor * 0.1) + np.random.normal(0, 0.05)
        margin_pct = np.clip(margin_pct, 0.15, 0.65)  # Realistic margin range

        feature_dict = {
            'rating': row['Rating'],
            'review_count': row['Number_of_Reviews'],
            'category_factor': category_factor,
            'brand_factor': brand_factor,
            'revenue': base_revenue,
            'margin_pct': margin_pct,
            'profit': base_revenue * margin_pct
        }

        training_features.append(feature_dict)

    return pd.DataFrame(training_features)

import matplotlib.pyplot as plt

def _build_forecasting_models(training_data):
    """Build and evaluate ML models for revenue and margin forecasting with comprehensive improvements."""
    if len(training_data) < 10:
        print("Insufficient training data for ML models")
        return None, None

    try:
        # Data validation and cleaning
        print("Validating and cleaning training data...")
        training_data = training_data.copy()
        
        # Remove rows with missing values in key columns
        key_columns = ['rating', 'review_count', 'category_factor', 'brand_factor', 'revenue', 'margin_pct']
        initial_rows = len(training_data)
        training_data = training_data.dropna(subset=key_columns)
        print(f"Removed {initial_rows - len(training_data)} rows with missing values")
        
        if len(training_data) < 5:
            print("Insufficient clean training data after validation")
            return None, None

        # Feature engineering: Add interaction and log-transformed features
        training_data['rating_review_interaction'] = training_data['rating'] * training_data['review_count']
        training_data['log_review_count'] = np.log1p(training_data['review_count'])  # log(1 + x) to handle zeros
        training_data['revenue_per_review'] = training_data['revenue'] / (training_data['review_count'] + 1)  # Avoid division by zero
        training_data['brand_category_interaction'] = training_data['brand_factor'] * training_data['category_factor']

        # Remove outliers using IQR method for revenue and margin
        for col in ['revenue', 'margin_pct']:
            Q1 = training_data[col].quantile(0.25)
            Q3 = training_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_before = len(training_data)
            training_data = training_data[(training_data[col] >= lower_bound) & (training_data[col] <= upper_bound)]
            print(f"Removed {outliers_before - len(training_data)} outliers from {col}")

        feature_cols = ['rating', 'review_count', 'category_factor', 'brand_factor', 
                        'rating_review_interaction', 'log_review_count', 'revenue_per_review', 
                        'brand_category_interaction']

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(training_data[feature_cols]), 
            columns=feature_cols, 
            index=training_data.index
        )

        # Fast direct RandomForestRegressor for revenue model
        print("Training revenue forecasting model (fast RandomForest)...")
        y_revenue = training_data['revenue']
        best_rev_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            max_features='sqrt',
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )
        best_rev_model.fit(X_scaled, y_revenue)

        # Evaluation for revenue model
        cv_scores_rev_mae = cross_val_score(best_rev_model, X_scaled, y_revenue, cv=3, scoring='neg_mean_absolute_error')
        cv_scores_rev_r2 = cross_val_score(best_rev_model, X_scaled, y_revenue, cv=3, scoring='r2')
        cv_scores_rev_rmse = cross_val_score(best_rev_model, X_scaled, y_revenue, cv=3, scoring='neg_root_mean_squared_error')
        print(f"Revenue Model Performance:")
        print(f"  CV MAE: {-cv_scores_rev_mae.mean():.2f} (+/- {cv_scores_rev_mae.std() * 2:.2f})")
        print(f"  CV R²: {cv_scores_rev_r2.mean():.3f} (+/- {cv_scores_rev_r2.std() * 2:.3f})")
        print(f"  CV RMSE: {-cv_scores_rev_rmse.mean():.2f} (+/- {cv_scores_rev_rmse.std() * 2:.2f})")
        print(f"  Model Params: n_estimators=100, max_depth=10, max_features='sqrt', min_samples_split=2, min_samples_leaf=1")

        # Margin model with comprehensive evaluation
        print("Training margin forecasting model...")
        y_margin = training_data['margin_pct']

        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }

        margin_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
            )
        
        grid_search_mar = GridSearchCV(
            margin_model, 
            param_grid=param_grid,
            cv=min(5, len(training_data)//2),
            scoring='neg_mean_absolute_error', 
            n_jobs=-1,
            verbose=0
        )
            
        grid_search_mar.fit(X_scaled, y_margin)
        best_mar_model = grid_search_mar.best_estimator_

        # Comprehensive evaluation for margin model
        cv_scores_mar_mae = cross_val_score(best_mar_model, X_scaled, y_margin, cv=5, scoring='neg_mean_absolute_error')
        cv_scores_mar_r2 = cross_val_score(best_mar_model, X_scaled, y_margin, cv=5, scoring='r2')
        cv_scores_mar_rmse = cross_val_score(best_mar_model, X_scaled, y_margin, cv=5, scoring='neg_root_mean_squared_error')
        
        print(f"Margin Model Performance:")
        print(f"  CV MAE: {-cv_scores_mar_mae.mean():.3f} (+/- {cv_scores_mar_mae.std() * 2:.3f})")
        print(f"  CV R²: {cv_scores_mar_r2.mean():.3f} (+/- {cv_scores_mar_r2.std() * 2:.3f})")
        print(f"  CV RMSE: {-cv_scores_mar_rmse.mean():.3f} (+/- {cv_scores_mar_rmse.std() * 2:.3f})")
        print(f"  Best Margin Params: {grid_search_mar.best_params_}")

        # Feature importance analysis
        print("\nFeature Importance Analysis:")
        rev_feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_rev_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mar_feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_mar_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Revenue Model - Top 5 Features:")
        for i, row in rev_feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
            
        print("Margin Model - Top 5 Features:")
        for i, row in mar_feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        # Store scaler with models for later use
        best_rev_model.scaler = scaler
        best_mar_model.scaler = scaler
        best_rev_model.feature_cols = feature_cols
        best_mar_model.feature_cols = feature_cols

        return best_rev_model, best_mar_model

    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _apply_forecasting_models(recommendations_df, revenue_model, margin_model, training_data):
    """Apply trained models to forecast new product performance with improved scaling and error handling."""
    if revenue_model is None or margin_model is None:
        return _apply_industry_averages(recommendations_df)

    forecasted_products = []

    try:
        for idx, product in recommendations_df.iterrows():
            try:
                # Create feature vector for new product
                features = _extract_product_features(product, training_data)
                
                # Scale features using the stored scaler
                if hasattr(revenue_model, 'scaler') and hasattr(revenue_model, 'feature_cols'):
                    # Create DataFrame with proper column names for scaling
                    features_df = pd.DataFrame([features], columns=revenue_model.feature_cols)
                    features_scaled = revenue_model.scaler.transform(features_df)
                    
                    # Forecast revenue
                    predicted_revenue = revenue_model.predict(features_scaled)[0]
                    
                    # Forecast margin
                    predicted_margin = margin_model.predict(features_scaled)[0]
                else:
                    # Fallback to unscaled prediction if scaler is not available
                    print("Warning: Using unscaled features for prediction")
                    predicted_revenue = revenue_model.predict([features])[0]
                    predicted_margin = margin_model.predict([features])[0]

                # Ensure predictions are within reasonable bounds
                predicted_revenue = max(1000, min(500000, predicted_revenue))  # Between $1K-$500K
                predicted_margin = max(0.05, min(0.8, predicted_margin))  # Between 5%-80%

                # Calculate profit
                predicted_profit = predicted_revenue * predicted_margin

                # Determine confidence based on prediction stability
                confidence = 'High'
                if not training_data.empty:
                    revenue_median = training_data['revenue'].median()
                    if predicted_revenue < revenue_median * 0.5:
                        confidence = 'Low'
                    elif predicted_revenue < revenue_median:
                        confidence = 'Medium'

                # Add forecasting results to product
                product_dict = product.to_dict()
                product_dict.update({
                    'forecasted_yearly_revenue': predicted_revenue,
                    'forecasted_margin_pct': predicted_margin,
                    'forecasted_yearly_profit': predicted_profit,
                    'forecast_confidence': confidence,
                    'model_version': 'Enhanced_ML_v2.0'
                })

                forecasted_products.append(product_dict)

            except Exception as e:
                print(f"Warning: Error forecasting product {product.get('product_name', 'Unknown')}: {e}")
                # Use industry averages for this product
                product_dict = product.to_dict()
                product_dict.update({
                    'forecasted_yearly_revenue': 50000 * np.random.uniform(0.8, 1.2),
                    'forecasted_margin_pct': 0.35 * np.random.uniform(0.9, 1.1),
                    'forecasted_yearly_profit': 17500 * np.random.uniform(0.8, 1.2),
                    'forecast_confidence': 'Low (Fallback)',
                    'model_version': 'Fallback_Industry_Average'
                })
                forecasted_products.append(product_dict)

        return pd.DataFrame(forecasted_products)

    except Exception as e:
        print(f"Error in forecasting process: {e}")
        return _apply_industry_averages(recommendations_df)

def _extract_product_features(product, training_data):
    """Extract features from product recommendation for ML prediction with enhanced feature engineering."""
    # Default values based on industry standards for new products
    rating = 4.2  # Assumed rating for new products
    review_count = 100  # Assumed initial reviews

    # Category factor based on product category
    category_factor = 1.0
    if 'category' in product and pd.notna(product['category']):
        category_factor = len(str(product['category'])) / 20

    # Brand factor (new products get average brand factor)
    brand_factor = training_data['brand_factor'].mean() if not training_data.empty else 1.0

    # Calculate engineered features (same as in training)
    rating_review_interaction = rating * review_count
    log_review_count = np.log1p(review_count)
    revenue_per_review = 500  # Estimated revenue per review for new products
    brand_category_interaction = brand_factor * category_factor

    # Return all features in the same order as training
    features = [
        rating, 
        review_count, 
        category_factor, 
        brand_factor,
        rating_review_interaction,
        log_review_count,
        revenue_per_review,
        brand_category_interaction
    ]
    
    return features

def _apply_industry_averages(recommendations_df):
    """Apply industry average projections when ML models can't be built."""
    print("Applying industry average projections...")

    # Beauty industry averages (based on typical market data)
    avg_revenue_per_product = 50000  # $50K annual revenue per product
    avg_margin_pct = 0.35  # 35% margin
    avg_profit_per_product = avg_revenue_per_product * avg_margin_pct

    forecasted_products = []

    for idx, product in recommendations_df.iterrows():
        product_dict = product.to_dict()
        product_dict.update({
            'forecasted_yearly_revenue': avg_revenue_per_product * np.random.uniform(0.8, 1.2),
            'forecasted_margin_pct': avg_margin_pct * np.random.uniform(0.9, 1.1),
            'forecasted_yearly_profit': avg_profit_per_product * np.random.uniform(0.8, 1.2),
            'forecast_confidence': 'Low (Industry Average)'
        })

        forecasted_products.append(product_dict)

    return pd.DataFrame(forecasted_products)

def _add_business_metrics(forecasted_df):
    """Add professional business metrics that companies need to see."""

    # Calculate additional business metrics
    forecasted_df['forecasted_monthly_revenue'] = forecasted_df['forecasted_yearly_revenue'] / 12
    forecasted_df['forecasted_monthly_profit'] = forecasted_df['forecasted_yearly_profit'] / 12

    # Break-even analysis (assuming $10K fixed cost per product launch)
    fixed_cost_per_product = 10000
    forecasted_df['break_even_months'] = fixed_cost_per_product / forecasted_df['forecasted_monthly_profit']
    forecasted_df['break_even_months'] = forecasted_df['break_even_months'].clip(1, 24)  # Cap at 2 years

    # ROI calculation (Return on Investment)
    forecasted_df['roi_pct'] = (forecasted_df['forecasted_yearly_profit'] / fixed_cost_per_product) * 100

    # Market potential scoring
    try:
        forecasted_df['market_potential_score'] = pd.qcut(
            forecasted_df['forecasted_yearly_revenue'], 3, labels=['Low', 'Medium', 'High'], duplicates='drop'
        )
    except ValueError as e:
        # Fallback if qcut still fails - use simple percentile-based approach
        print(f"Warning: Could not create market potential bins with qcut: {e}")
        revenue_median = forecasted_df['forecasted_yearly_revenue'].median()
        revenue_75th = forecasted_df['forecasted_yearly_revenue'].quantile(0.75)
        
        conditions = [
            forecasted_df['forecasted_yearly_revenue'] >= revenue_75th,
            forecasted_df['forecasted_yearly_revenue'] >= revenue_median,
            forecasted_df['forecasted_yearly_revenue'] < revenue_median
        ]
        choices = ['High', 'Medium', 'Low']
        forecasted_df['market_potential_score'] = np.select(conditions, choices, default='Low')

    # Customer Acquisition Cost estimation (rough estimate)
    forecasted_df['estimated_cac'] = forecasted_df['forecasted_yearly_revenue'] * 0.15  # 15% of revenue

    # Customer Lifetime Value
    forecasted_df['estimated_clv'] = forecasted_df['forecasted_yearly_revenue'] * 2.5  # 2.5x annual revenue

    # Profitability index
    forecasted_df['profitability_index'] = forecasted_df['forecasted_margin_pct'] * forecasted_df['roi_pct'] / 100

    # Risk assessment
    forecasted_df['risk_level'] = pd.cut(
        forecasted_df['break_even_months'],
        bins=[0, 6, 12, float('inf')],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    # Investment recommendation
    conditions = [
        (forecasted_df['profitability_index'] > 1.5) & (forecasted_df['risk_level'] == 'Low Risk'),
        (forecasted_df['profitability_index'] > 1.0) & (forecasted_df['risk_level'].isin(['Low Risk', 'Medium Risk'])),
        (forecasted_df['profitability_index'] > 0.5)
    ]
    choices = ['Strong Recommend', 'Recommend', 'Consider']
    forecasted_df['investment_recommendation'] = np.select(conditions, choices, default='Not Recommended')

    return forecasted_df

# ------------------------------------------

def load_data():
    """Load and merge all required data files"""
    print("--- Loading Core Data Files ---")
    dataframes = {}

    # --- Core Data Files ---
    try:
        dataframes['fusion_results'] = pd.read_csv(FUSION_RESULTS_PATH)
        print(f"Loaded Fusion Results: {dataframes['fusion_results'].shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {FUSION_RESULTS_PATH}")
        dataframes['fusion_results'] = pd.DataFrame()

    try:
        dataframes['videos_df'] = pd.read_csv(VIDEOS_PATH)
        print(f"Loaded Videos Meta {dataframes['videos_df'].shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {VIDEOS_PATH}")
        dataframes['videos_df'] = pd.DataFrame()

    try:
        dataframes['comments_df'] = pd.read_parquet(COMMENTS_PATH)
        print(f"Loaded Comments Enriched: {dataframes['comments_df'].shape}")
    except FileNotFoundError:
        print(f"Warning: Could not find {COMMENTS_PATH}. Proceeding without comments data.")
        dataframes['comments_df'] = pd.DataFrame()
    except Exception as e:
        print(f"Warning: Error loading comments: {e}. Proceeding without comments data.")
        dataframes['comments_df'] = pd.DataFrame()

    # --- New Signal Data Files ---
    print("\n--- Loading New Signal Data Files ---")
    try:
        dataframes['amazon_ratings'] = pd.read_csv(AMAZON_RATINGS_PATH)
        print(f"Loaded Amazon Ratings: {dataframes['amazon_ratings'].shape}")
    except FileNotFoundError:
        print(f"Warning: Could not find {AMAZON_RATINGS_PATH}. Analysis will proceed without it.")
        dataframes['amazon_ratings'] = pd.DataFrame()
    except Exception as e:
        print(f"Warning: Error loading Amazon Ratings: {e}. Proceeding without it.")
        dataframes['amazon_ratings'] = pd.DataFrame()

    try:
        dataframes['top_products'] = pd.read_csv(TOP_PRODUCTS_PATH)
        print(f"Loaded Top Beauty Products 2024: {dataframes['top_products'].shape}")
    except FileNotFoundError:
        print(f"Warning: Could not find {TOP_PRODUCTS_PATH}. Analysis will proceed without it.")
        dataframes['top_products'] = pd.DataFrame()
    except Exception as e:
        print(f"Warning: Error loading Top Products: {e}. Proceeding without it.")
        dataframes['top_products'] = pd.DataFrame()

    try:
        dataframes['supply_chain'] = pd.read_csv(SUPPLY_CHAIN_PATH)
        print(f"Loaded Supply Chain Analysis: {dataframes['supply_chain'].shape}")
    except FileNotFoundError:
        print(f"Warning: Could not find {SUPPLY_CHAIN_PATH}. Analysis will proceed without it.")
        dataframes['supply_chain'] = pd.DataFrame()
    except Exception as e:
        print(f"Warning: Error loading Supply Chain: {e}. Proceeding without it.")
        dataframes['supply_chain'] = pd.DataFrame()

    # --- Merge Core Datasets ---
    print("\n--- Merging Core Datasets ---")
    merged_df = pd.DataFrame()
    if not dataframes['fusion_results'].empty and not dataframes['videos_df'].empty:
        merged_df = dataframes['fusion_results'].merge(dataframes['videos_df'], on='videoId', how='left')
        print(f"Merged Fusion Results with Videos: {merged_df.shape}")

        if not dataframes['comments_df'].empty:
            try:
                # Aggregate comments per video
                dataframes['comments_df']['text'] = dataframes['comments_df']['text'].fillna('')
                comments_agg = dataframes['comments_df'].groupby('videoId').agg({
                    'text': lambda x: ' '.join(x.astype(str)),
                    'likeCount': 'sum',
                    # Assuming 'authorDisplayName' count represents comment count if no specific count column
                    'authorDisplayName': 'count'
                }).reset_index()
                comments_agg.rename(columns={'text': 'all_comments', 'authorDisplayName': 'comment_count'}, inplace=True)
                merged_df = merged_df.merge(comments_agg, on='videoId', how='left')
                print(f"Merged with Aggregated Comments: {merged_df.shape}")
            except KeyError as e:
                print(f"Warning: Expected column not found in comments for aggregation: {e}. Skipping comment merge.")
            except Exception as e:
                print(f"Warning: Error aggregating/merging comments: {e}. Skipping comment merge.")
    else:
        print("Warning: Insufficient core data (fusion_results or videos_df) to perform merge.")
        # Return individual dataframes if core merge fails
        return dataframes

    print(f"Final Merged Core Dataset Shape: {merged_df.shape}")
    dataframes['merged_core'] = merged_df
    return dataframes

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == '':
        return ''
    # Convert to lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits (keep spaces and letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Optional: Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_beauty_terms(text):
    """Extract beauty-related terms from text"""
    if pd.isna(text) or text == '':
        return []
    beauty_keywords = [
        'skincare', 'moisturizer', 'serum', 'cream', 'lotion', 'toner', 'cleanser', 'exfoliant',
        'makeup', 'foundation', 'concealer', 'blush', 'eyeshadow', 'lipstick', 'mascara', 'eyeliner',
        'haircare', 'shampoo', 'conditioner', 'treatment', 'mask', 'oil',
        'fragrance', 'perfume', 'cologne', 'scent',
        'ingredient', 'vitamin', 'acid', 'retinol', 'hyaluronic', 'niacinamide', 'salicylic',
        'natural', 'organic', 'vegan', 'cruelty-free', 'sustainable',
        'trend', 'viral', 'popular', 'best', 'new', 'innovative',
        'skin', 'hair', 'beauty', 'cosmetic', 'product'
    ]
    text_lower = text.lower()
    found_terms = [keyword for keyword in beauty_keywords if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower)]
    return found_terms

# --- Functions for Gap Analysis ---
def extract_product_mentions(text):
    """Extract product mentions from text"""
    if pd.isna(text):
        return []
    text_lower = text.lower()
    product_patterns = [
        r'\b(\w+[-\s])*(serum|cream|lotion|moisturizer|mask|treatment|oil|gel)\b',
        r'\b(\w+[-\s])*(foundation|concealer|blush|bronzer|highlighter)\b',
        r'\b(\w+[-\s])*(eyeshadow|eyeliner|mascara|lipstick|lip gloss)\b',
        r'\b(\w+[-\s])*(shampoo|conditioner|hair mask|hair oil)\b',
        r'\b(\w+[-\s])*(perfume|cologne|body lotion|body wash)\b'
    ]
    products = []
    for pattern in product_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                product_parts = [m.strip() for m in match if m.strip()]
                if product_parts:
                    product = ' '.join(product_parts)
                else:
                    continue
            else:
                product = match.strip()
            if product and len(product) > 2 and product not in ['the', 'and', 'for']:
                 products.append(product)
    return products

def extract_ingredients(text):
    """Extract ingredient mentions from text"""
    if pd.isna(text):
        return []
    text_lower = text.lower()
    ingredients = [
        r'\bhyaluronic acid\b', r'\bvitamin c\b', r'\bvitamin e\b', r'\bretinol\b',
        r'\bniacinamide\b', r'\bsalicylic acid\b', r'\bglycolic acid\b', r'\blactic acid\b',
        r'\bazelaic acid\b', r'\bceramide\b', r'\bcollagen\b', r'\bpeptides\b',
        r'\bsnail mucin\b', r'\bcharcoal\b', r'\btea tree oil\b', r'\brosehip oil\b',
        r'\bargan oil\b', r'\bjojoba oil\b', r'\bshea butter\b', r'\baloe vera\b',
        r'\bwitch hazel\b', r'\bgreen tea\b', r'\bcentella asiatica\b',
        r'\btranexamic acid\b', r'\bkojic acid\b', r'\bmandelic acid\b',
        r'\bnatural\b', r'\borganic\b', r'\bvegan\b', r'\bcruelty[-\s]free\b'
    ]
    found_ingredients = []
    for ingredient_pattern in ingredients:
        matches = re.findall(ingredient_pattern, text_lower)
        found_ingredients.extend(matches)
    return found_ingredients
# --- End Gap Analysis Functions ---

def detect_trending_tags(df, top_n=100):
    """Detect trending tags from the dataset"""
    all_tags = []
    if 'tags' in df.columns:
        for tags in df['tags'].dropna():
            try:
                if isinstance(tags, str) and tags.startswith('[') and tags.endswith(']'):
                    tag_list = ast.literal_eval(tags)
                    if isinstance(tag_list, list):
                        all_tags.extend([tag.strip().lower() for tag in tag_list if tag.strip()])
                else:
                    delimiters = [',', '|', ';']
                    delimiter_found = False
                    for delim in delimiters:
                        if delim in str(tags):
                            tag_list = str(tags).split(delim)
                            all_tags.extend([tag.strip().lower() for tag in tag_list if tag.strip()])
                            delimiter_found = True
                            break
                    if not delimiter_found:
                         all_tags.append(str(tags).strip().lower())
            except (ValueError, SyntaxError):
                all_tags.append(str(tags).strip().lower())

    text_fields = []
    for idx, row in df.iterrows():
        combined_text = ''
        if 'title' in df.columns and pd.notna(row['title']):
            combined_text += str(row['title']) + ' '
        if 'description' in df.columns and pd.notna(row['description']):
            combined_text += str(row['description']) + ' '
        text_fields.append(combined_text.strip())

    if text_fields:
        try:
            tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=2)
            tfidf_matrix = tfidf.fit_transform(text_fields)
            feature_names = tfidf.get_feature_names_out()
            tfidf_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            tag_counts = Counter(all_tags)
            tag_scores = {tag: count for tag, count in tag_counts.items() if tag}
            tfidf_weight = 50
            for i, score in enumerate(tfidf_scores):
                if score > 0 and feature_names[i]:
                    tag_scores[feature_names[i]] = tag_scores.get(feature_names[i], 0) + score * tfidf_weight
            sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_tags[:top_n]
        except Exception as e:
            print(f"Warning: Error in TF-IDF processing for tags: {e}. Returning tag counts only.")
            tag_counts = Counter(all_tags)
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_tags[:top_n]
    else:
        tag_counts = Counter(all_tags)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:top_n]

def identify_product_gaps(df):
    """Identify product gaps based on frequently mentioned products"""
    loreal_brands = [
        'loreal', 'l\'oreal', 'lancome', 'kerastase', 'kiehl', 'kahl', 'ysl', 'yves saint laurent',
        'giorgio armani', 'armani', 'maybelline', 'nyx', 'essie', 'matrix', 'redken', 'pureology',
        'vichy', 'la roche-posay', 'derma tox', 'skin ceuticals', 'urban decay'
    ]

    if 'combined_text' not in df.columns:
        print("Creating temporary combined text for product gap analysis...")
        text_fields = []
        for idx, row in df.iterrows():
            combined_text = ''
            if 'title' in df.columns and pd.notna(row['title']):
                combined_text += str(row['title']) + ' '
            if 'description' in df.columns and pd.notna(row['description']):
                combined_text += str(row['description']) + ' '
            if 'tags' in df.columns and pd.notna(row['tags']):
                 combined_text += str(row['tags']) + ' '
            text_fields.append(combined_text.strip())
        df_temp_text = pd.Series(text_fields, name='combined_text')
    else:
        df_temp_text = df['combined_text']

    all_products = []
    for text in df_temp_text.dropna():
        products = extract_product_mentions(text)
        all_products.extend(products)

    product_counts = Counter(all_products)
    potential_gaps = []
    for product, count in product_counts.most_common(100):
        product_lower = product.lower()
        is_loreal = any(brand in product_lower for brand in loreal_brands)
        if not is_loreal and count > 5 and len(product) > 3:
             potential_gaps.append((product, count))
    return potential_gaps

def get_trending_ingredients(df):
    """Extract trending ingredients"""
    if 'combined_text' not in df.columns:
        print("Creating temporary combined text for ingredient analysis...")
        text_fields = []
        for idx, row in df.iterrows():
            combined_text = ''
            if 'title' in df.columns and pd.notna(row['title']):
                combined_text += str(row['title']) + ' '
            if 'description' in df.columns and pd.notna(row['description']):
                combined_text += str(row['description']) + ' '
            if 'tags' in df.columns and pd.notna(row['tags']):
                 combined_text += str(row['tags']) + ' '
            if 'all_comments' in df.columns and pd.notna(row['all_comments']):
                combined_text += str(row['all_comments']) + ' '
            text_fields.append(combined_text.strip())
        df_temp_text = pd.Series(text_fields, name='combined_text')
    else:
        df_temp_text = df['combined_text']

    all_ingredients = []
    for text in df_temp_text.dropna():
        ingredients = extract_ingredients(text)
        all_ingredients.extend(ingredients)

    ingredient_counts = Counter(all_ingredients)
    return ingredient_counts.most_common(50)

# --- New Signal Processing Functions ---
def analyze_amazon_data(df_amazon):
    """Analyze Amazon ratings data for popular products/brands."""
    print("Analyzing Amazon Ratings data...")
    if df_amazon.empty:
        print("  -> No Amazon data available.")
        return [], []

    try:
        # Calculate average rating and review count per ProductId
        product_stats = df_amazon.groupby('ProductId').agg(
            avg_rating=('Rating', 'mean'),
            num_reviews=('Rating', 'count')
        ).reset_index()

        # Filter for products with sufficient reviews (e.g., > 10) and high average rating (> 4.0)
        popular_products = product_stats[
            (product_stats['num_reviews'] > 10) &
            (product_stats['avg_rating'] > 4.0)
        ].nlargest(50, ['avg_rating', 'num_reviews']) # Get top 50

        # Placeholder: We don't have product names/brands from ProductId alone easily.
        # This would ideally be joined with product metadata.
        # For now, just report ProductIds.
        top_amazon_products = popular_products['ProductId'].tolist()
        print(f"  -> Identified {len(top_amazon_products)} popular products on Amazon (based on ratings).")
        return top_amazon_products, [] # Return empty list for brands for now

    except Exception as e:
        print(f"  -> Error analyzing Amazon  {e}")
        return [], []

def analyze_top_products_data(df_top_products):
    """Analyze the 'Top Beauty Products 2024' list."""
    print("Analyzing Top Beauty Products 2024 data...")
    if df_top_products.empty:
        print("  -> No Top Products data available.")
        return [], [], []

    try:
        # 1. Identify top categories
        top_categories = df_top_products['Category'].value_counts().head(10).index.tolist()

        # 2. Identify top brands (based on presence in list)
        top_brands = df_top_products['Brand'].value_counts().head(10).index.tolist()

        # 3. Identify products with high ratings and many reviews (success indicators)
        # Assuming 'Rating' and 'Number_of_Reviews' are numeric
        df_top_products['Rating'] = pd.to_numeric(df_top_products['Rating'], errors='coerce')
        df_top_products['Number_of_Reviews'] = pd.to_numeric(df_top_products['Number_of_Reviews'], errors='coerce')

        successful_products_df = df_top_products[
            (df_top_products['Rating'] > 4.0) &
            (df_top_products['Number_of_Reviews'] > 100) # Arbitrary threshold
        ].nlargest(30, 'Number_of_Reviews') # Get top 30 by review count

        successful_product_names = successful_products_df['Product_Name'].tolist()
        successful_brands_from_list = successful_products_df['Brand'].unique().tolist()

        print(f"  -> Top Categories: {top_categories[:5]}...")
        print(f"  -> Top Brands: {top_brands[:5]}...")
        print(f"  -> Successful Products (High Rating & Reviews): {len(successful_product_names)} found.")
        return top_categories, top_brands, successful_product_names, successful_brands_from_list

    except Exception as e:
        print(f"  -> Error analyzing Top Products  {e}")
        return [], [], [], []

def analyze_supply_chain_data(df_supply_chain):
    """Analyze supply chain data for high-performing product types."""
    print("Analyzing Supply Chain data...")
    if df_supply_chain.empty:
        print("  -> No Supply Chain data available.")
        return [], []

    try:
        # Calculate performance metrics per product type
        supply_metrics = df_supply_chain.groupby('Product type').agg(
            total_revenue=('Revenue generated', 'sum'),
            total_sold=('Number of products sold', 'sum'),
            avg_availability=('Availability', 'mean'),
            avg_lead_time=('Lead times', 'mean')
        ).reset_index()

        # Score product types (example: prioritize high revenue, high sales, high availability, low lead time)
        # Normalize metrics (simple min-max for this example)
        supply_metrics['norm_revenue'] = (supply_metrics['total_revenue'] - supply_metrics['total_revenue'].min()) / (supply_metrics['total_revenue'].max() - supply_metrics['total_revenue'].min() + 1e-8)
        supply_metrics['norm_sold'] = (supply_metrics['total_sold'] - supply_metrics['total_sold'].min()) / (supply_metrics['total_sold'].max() - supply_metrics['total_sold'].min() + 1e-8)
        supply_metrics['norm_avail'] = (supply_metrics['avg_availability'] - supply_metrics['avg_availability'].min()) / (supply_metrics['avg_availability'].max() - supply_metrics['avg_availability'].min() + 1e-8)
        # Invert lead time for scoring (lower is better)
        supply_metrics['norm_lead'] = 1 - ((supply_metrics['avg_lead_time'] - supply_metrics['avg_lead_time'].min()) / (supply_metrics['avg_lead_time'].max() - supply_metrics['avg_lead_time'].min() + 1e-8))

        # Simple weighted score (weights can be adjusted)
        supply_metrics['performance_score'] = (
            0.3 * supply_metrics['norm_revenue'] +
            0.3 * supply_metrics['norm_sold'] +
            0.2 * supply_metrics['norm_avail'] +
            0.2 * supply_metrics['norm_lead']
        )

        # Get top performing product types
        top_supply_types = supply_metrics.nlargest(10, 'performance_score')['Product type'].tolist()
        print(f"  -> Top Performing Supply Chain Product Types: {top_supply_types}")
        return top_supply_types

    except Exception as e:
        print(f"  -> Error analyzing Supply Chain  {e}")
        return []

# --- End New Signal Processing Functions ---

def generate_deepseek_recommendations(
    product_gaps, trending_ingredients, trending_tags,
    top_amazon_products, top_categories, top_brands,
    successful_products, top_supply_types, successful_brands_from_list
):
    """Generate product recommendations using OpenRouter DeepSeek API"""
    recommendations = []

    # Prepare data for the LLM
    product_gap_text = "\n".join([f"- {product} (mentioned {count} times)" for product, count in product_gaps[:20]])
    ingredient_text = "\n".join([f"- {ingredient} (mentioned {count} times)" for ingredient, count in trending_ingredients[:20]])
    tag_text = "\n".join([f"- {tag} (score: {score:.2f})" for tag, score in trending_tags[:30]])

    # New Signals
    amazon_text = "\n".join([f"- Product ID: {pid}" for pid in top_amazon_products[:20]]) if top_amazon_products else "No data available."
    categories_text = "\n".join([f"- {cat}" for cat in top_categories[:10]]) if top_categories else "No data available."
    brands_text = "\n".join([f"- {brand}" for brand in list(set(top_brands + successful_brands_from_list))[:15]]) if top_brands or successful_brands_from_list else "No data available."
    successful_products_text = "\n".join([f"- {prod}" for prod in successful_products[:20]]) if successful_products else "No data available."
    supply_chain_text = "\n".join([f"- {ptype}" for ptype in top_supply_types[:10]]) if top_supply_types else "No data available."

    prompt = f"""
    You are an expert beauty industry analyst and innovation strategist for L'Oréal. Your task is to analyze multiple data signals and recommend innovative beauty products that L'Oréal should consider developing to fill market gaps and capitalize on trends.

    Based on the following comprehensive beauty trend and market data, recommend 15-20 innovative beauty products that L'Oréal should consider developing. These products should NOT be part of L'Oréal's current portfolio and should leverage the trending concepts and market insights.

    --- Trending Data from Social Media/Video Analysis ---
    Trending Product Concepts (Market Gaps):
    {product_gap_text if product_gap_text else 'No data available.'}

    Trending Ingredients:
    {ingredient_text if ingredient_text else 'No data available.'}

    Trending Tags/Keywords:
    {tag_text if tag_text else 'No data available.'}

    --- Market Data Signals ---

    1. Popular Products on Amazon (High Ratings & Reviews):
    {amazon_text}

    2. Top Beauty Product Categories (2024 List):
    {categories_text}

    3. Leading Beauty Brands (2024 List):
    {brands_text}

    4. Highly Successful Existing Products (High Rating & Many Reviews):
    {successful_products_text}

    5. Top Performing Product Types in Supply Chain (High Revenue/Sales):
    {supply_chain_text}

    --- Instructions ---
    Please provide your recommendations in this exact format, one product per line:
    Product Name|Product Category|Key Ingredients|Target Market|Innovation Description

    Example format (do not include this example in your output):
    Vitamin C Glow Serum|Skincare|Vitamin C, Hyaluronic Acid|Millennials & Gen Z|A stable vitamin C formulation with time-release technology for consistent brightening

    Focus on products that are:
    1. Truly innovative and not currently in L'Oréal's portfolio.
    2. Based on the trending ingredients, concepts, categories, and successful market examples provided.
    3. Address specific consumer needs or market segments indicated by the data.
    4. Align with high-performing supply chain categories where possible.
    5. Provide a clear innovation description explaining the unique value proposition.
    6. Use the exact format specified.

    Provide only the list of recommendations, nothing else. Aim for diversity across categories (Skincare, Haircare, Makeup, Fragrance, Body Care) and target markets.
    """

    print("\n--- Generating Product Recommendations using DeepSeek-TNG-R1T2-Chimera via OpenRouter ---")

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        if YOUR_SITE_URL:
            headers["HTTP-Referer"] = YOUR_SITE_URL
        if YOUR_SITE_NAME:
            headers["X-Title"] = YOUR_SITE_NAME

        payload = {
            "model": "tngtech/deepseek-r1t2-chimera:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2500, # Increased token limit for potentially longer list
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=180 # Increased timeout
        )

        response.raise_for_status()
        response_data = response.json()

        if 'choices' in response_data and len(response_data['choices']) > 0:
            generated_text = response_data['choices'][0].get('message', {}).get('content', '')
            if generated_text:
                lines = generated_text.strip().split('\n')
                for line in lines:
                    if '|' in line and not line.startswith("---") and not "Example format" in line:
                        parts = line.split('|')
                        if len(parts) >= 5:
                            recommendations.append({
                                'product_name': parts[0].strip(),
                                'category': parts[1].strip(),
                                'key_ingredients': parts[2].strip(),
                                'target_market': parts[3].strip(),
                                'innovation_description': parts[4].strip()
                            })
                        else:
                            print(f"Warning: Skipping malformed recommendation line: {line}")

            else:
                print("Warning: No content found in the API response.")
        else:
            print("Warning: Unexpected API response structure.")
            print(f"API Response Sample: {str(response_data)[:500]}...")

        if not recommendations:
             print("Warning: No valid recommendations parsed from API response. Using fallback recommendations.")
             recommendations = [
                {
                    'product_name': 'Hyaluronic Acid Overnight Mask',
                    'category': 'Skincare',
                    'key_ingredients': 'Hyaluronic Acid, Ceramides',
                    'target_market': 'All Ages',
                    'innovation_description': 'Advanced moisture delivery system for intensive overnight hydration'
                }
             ]

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        print("Using fallback recommendations.")
        recommendations = [
            {
                'product_name': 'CBD Soothing Body Lotion',
                'category': 'Body Care',
                'key_ingredients': 'CBD, Aloe Vera, Chamomile',
                'target_market': 'Sensitive Skin',
                'innovation_description': 'Calms irritated skin and provides long-lasting hydration'
            }
        ]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from API: {e}")
        print("Using fallback recommendations.")
        recommendations = [
            {
                'product_name': 'Adaptogenic Stress Relief Cream',
                'category': 'Skincare',
                'key_ingredients': 'Ashwagandha, Reishi Mushroom',
                'target_market': 'Gen Z & Millennials',
                'innovation_description': 'Skincare that addresses stress-related skin concerns'
            }
        ]
    except Exception as e:
        print(f"An unexpected error occurred during API call or processing: {e}")
        print("Using fallback recommendations.")
        recommendations = [
            {
                'product_name': 'Multi-Peptide Firming Eye Cream',
                'category': 'Skincare',
                'key_ingredients': 'Matrixyl, Argireline, Peptides',
                'target_market': 'Aging Skin',
                'innovation_description': 'Targets multiple signs of aging around the eye area'
            }
        ]

    return recommendations

def main():
    """Main function to run the beauty trend analysis"""
    try:
        # --- 1. Load and Merge Data ---
        data_dict = load_data()

        # --- 2. Process Core Data for Trends ---
        df_core = data_dict.get('merged_core', pd.DataFrame())
        if df_core.empty:
             print("\nError: Failed to create merged core dataset. Cannot proceed with core trend analysis.")
             # We can still try to proceed with other available data signals
             df_core = pd.DataFrame() # Ensure it's a DataFrame even if empty
        else:
            print("\n--- Processing Core Data for Trends ---")
            if 'combined_text' not in df_core.columns:
                print("Creating combined text field for analysis...")
                text_fields = []
                for idx, row in df_core.iterrows():
                    combined_text = ''
                    if 'title' in df_core.columns and pd.notna(row['title']):
                        combined_text += str(row['title']) + ' '
                    if 'description' in df_core.columns and pd.notna(row['description']):
                        combined_text += str(row['description']) + ' '
                    if 'tags' in df_core.columns and pd.notna(row['tags']):
                        combined_text += str(row['tags']) + ' '
                    if 'all_comments' in df_core.columns and pd.notna(row['all_comments']):
                        combined_text += str(row['all_comments']) + ' '
                    text_fields.append(combined_text.strip())
                df_core['combined_text'] = text_fields
            else:
                print("Using existing 'combined_text' column.")

            print("Cleaning text data...")
            df_core['cleaned_text'] = df_core['combined_text'].apply(clean_text)

            print("Detecting trending tags...")
            trending_tags = detect_trending_tags(df_core)
            print("Identifying product gaps...")
            product_gaps = identify_product_gaps(df_core)
            print("Analyzing trending ingredients...")
            trending_ingredients = get_trending_ingredients(df_core)
        # Provide empty lists if core analysis failed
        if 'trending_tags' not in locals(): trending_tags = []
        if 'product_gaps' not in locals(): product_gaps = []
        if 'trending_ingredients' not in locals(): trending_ingredients = []


        # --- 3. Process New Signal Data ---
        print("\n--- Processing New Signal Data ---")
        # a. Amazon Ratings
        top_amazon_products, top_amazon_brands = analyze_amazon_data(data_dict.get('amazon_ratings', pd.DataFrame()))

        # b. Top Beauty Products 2024
        top_categories, top_brands, successful_products, successful_brands_from_list = analyze_top_products_data(data_dict.get('top_products', pd.DataFrame()))

        # c. Supply Chain Analysis
        top_supply_types = analyze_supply_chain_data(data_dict.get('supply_chain', pd.DataFrame()))


        # --- 4. Save Intermediate Data Analysis ---
        print("\n--- Saving Intermediate Analysis Data ---")
        try:
            intermediate_data_saved = False
            if trending_tags:
                pd.DataFrame(trending_tags, columns=['tag', 'score']).to_csv('trending_tags.csv', index=False)
                print(f"Saved 'trending_tags.csv' ({len(trending_tags)} tags)")
                intermediate_data_saved = True
            if product_gaps:
                pd.DataFrame(product_gaps, columns=['product', 'mentions']).to_csv('product_gaps.csv', index=False)
                print(f"Saved 'product_gaps.csv' ({len(product_gaps)} gaps)")
                intermediate_data_saved = True
            if trending_ingredients:
                pd.DataFrame(trending_ingredients, columns=['ingredient', 'mentions']).to_csv('trending_ingredients.csv', index=False)
                print(f"Saved 'trending_ingredients.csv' ({len(trending_ingredients)} ingredients)")
                intermediate_data_saved = True
            # Save new signal summaries
            if top_amazon_products:
                pd.DataFrame({'product_id': top_amazon_products}).to_csv('top_amazon_products.csv', index=False)
                print(f"Saved 'top_amazon_products.csv' ({len(top_amazon_products)} products)")
                intermediate_data_saved = True
            if top_categories:
                pd.DataFrame({'category': top_categories}).to_csv('top_categories.csv', index=False)
                print(f"Saved 'top_categories.csv' ({len(top_categories)} categories)")
                intermediate_data_saved = True
            if top_brands:
                pd.DataFrame({'brand': top_brands}).to_csv('top_brands.csv', index=False)
                print(f"Saved 'top_brands.csv' ({len(top_brands)} brands)")
                intermediate_data_saved = True
            if successful_products:
                pd.DataFrame({'product_name': successful_products}).to_csv('successful_products.csv', index=False)
                print(f"Saved 'successful_products.csv' ({len(successful_products)} products)")
                intermediate_data_saved = True
            if top_supply_types:
                pd.DataFrame({'product_type': top_supply_types}).to_csv('top_supply_types.csv', index=False)
                print(f"Saved 'top_supply_types.csv' ({len(top_supply_types)} types)")
                intermediate_data_saved = True

            if not intermediate_data_saved:
                print("No intermediate data to save.")

        except Exception as e:
            print(f"Warning: Could not save all intermediate CSV files: {e}")


        # --- 5. Generate Recommendations using DeepSeek via OpenRouter ---
        print("\n--- Generating Final Product Recommendations ---")
        recommendations = generate_deepseek_recommendations(
            product_gaps, trending_ingredients, trending_tags,
            top_amazon_products, top_categories, top_brands,
            successful_products, top_supply_types, successful_brands_from_list
        )

        # --- 6. Generate Revenue Forecasts ---
        print("\n--- Generating Revenue Forecasts ---")
        if recommendations:
            # Convert recommendations to DataFrame for forecasting
            temp_rec_df = pd.DataFrame(recommendations)

            # Apply ML forecasting
            forecasted_df = forecast_product_revenue_and_margin(
                temp_rec_df,
                data_dict.get('top_products', pd.DataFrame()),
                data_dict.get('supply_chain', pd.DataFrame())
            )

            # Add context columns to forecasted data
            forecasted_df['analysis_context_trending_tags'] = str([tag for tag, score in trending_tags[:5]]) if trending_tags else "[]"
            forecasted_df['analysis_context_top_product_gaps'] = str([product for product, count in product_gaps[:5]]) if product_gaps else "[]"
            forecasted_df['analysis_context_top_ingredients'] = str([ingredient for ingredient, count in trending_ingredients[:5]]) if trending_ingredients else "[]"
            forecasted_df['analysis_context_top_categories'] = str(top_categories[:5]) if top_categories else "[]"
            forecasted_df['analysis_context_top_brands'] = str(top_brands[:5]) if top_brands else "[]"
            forecasted_df['analysis_context_supply_types'] = str(top_supply_types[:5]) if top_supply_types else "[]"

            # Save forecasted recommendations
            output_filename = 'beauty_innovation_recommendations_with_forecasts.csv'
            forecasted_df.to_csv(output_filename, index=False)
            print(f"\n--- Analysis Complete! ---")
            print(f"Final recommendations with forecasts saved to '{output_filename}' ({len(forecasted_df)} products)")

            # Display top forecasted products
            print("\n--- Top 5 Forecasted Products by Revenue Potential ---")
            top_forecasted = forecasted_df.nlargest(5, 'forecasted_yearly_revenue')
            for i, (_, product) in enumerate(top_forecasted.iterrows()):
                print(f"{i+1}. {product['product_name']} - {product['category']}")
                print(".0f")
                print(".1f")
                print(f"   Break-even: {product['break_even_months']:.1f} months")
                print(f"   Investment: {product['investment_recommendation']}")
                print(f"   Key Ingredients: {product.get('key_ingredients', 'N/A')}")
                print(f"   Target Market: {product.get('target_market', 'N/A')}\n")
        else:
            print("\n--- Analysis Complete ---")
            print("No final recommendations were generated.")


        # --- 7. Display Sample Results ---
        if recommendations:
            print("\n--- Summary of All Forecasted Products ---")
            print(f"Total Products: {len(forecasted_df)}")
            print(".0f")
            print(".0f")
            print(f"Average Break-even: {forecasted_df['break_even_months'].mean():.1f} months")
            print(f"Top Investment Recommendations: {forecasted_df['investment_recommendation'].value_counts().to_dict()}")
        else:
            print("\nNo recommendations available to display.")

    except Exception as e:
        print(f"An unexpected error occurred in the main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if OPENROUTER_API_KEY == "<OPENROUTER_API_KEY>" or not OPENROUTER_API_KEY:
        print("ERROR: Please replace '<OPENROUTER_API_KEY>' with your actual OpenRouter API key in the script.")
    else:
        main()