"""
Data Preprocessing Module
==========================
Handles auto-detection of dataset types and preprocessing for different data schemas.
Supports: Product catalogs (amazon.csv), Transaction data (customer_shopping_data.csv)
"""

import pandas as pd
import numpy as np
import re
import hashlib
import os
from datetime import datetime
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Smart data preprocessor that auto-detects dataset type and applies
    appropriate transformations for clustering and analytics.
    """
    
    DATASET_TYPES = {
        'product_catalog': ['product', 'rating', 'price', 'category', 'review'],
        'transaction_data': ['invoice', 'customer', 'quantity', 'payment', 'date'],
        'customer_profile': ['customer', 'age', 'gender', 'income', 'segment']
    }
    
    def __init__(self, cache_dir='static/cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.dataset_type = None
        self.original_df = None
        self.processed_df = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        self.text_columns = []
        
    def detect_dataset_type(self, df):
        """Auto-detect the type of dataset based on column names."""
        columns_lower = [col.lower() for col in df.columns]
        col_text = ' '.join(columns_lower)
        
        scores = {}
        for dtype, keywords in self.DATASET_TYPES.items():
            score = sum(1 for kw in keywords if kw in col_text)
            scores[dtype] = score
        
        best_type = max(scores, key=scores.get)
        self.dataset_type = best_type if scores[best_type] >= 2 else 'generic'
        return self.dataset_type
    
    def get_cache_key(self, df):
        """Generate a cache key based on dataframe content."""
        content = df.head(100).to_string() + str(df.shape)
        return hashlib.md5(content.encode()).hexdigest()
    
    def parse_currency(self, value):
        """Convert currency strings to float (handles ₹, $, €)."""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[₹$€£,\s]', '', str(value))
            return float(cleaned)
        except:
            return np.nan
    
    def parse_date(self, date_series, formats=None):
        """Smart date parsing with multiple format attempts."""
        if formats is None:
            formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d']
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_series, format=fmt, errors='coerce')
            except:
                continue
        return pd.to_datetime(date_series, errors='coerce')
    
    def classify_columns(self, df):
        """Classify columns by their data type for appropriate processing."""
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        self.text_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Skip ID columns
            if 'id' in col_lower and '_id' not in col_lower:
                continue
                
            dtype = df[col].dtype
            
            if dtype in ['int64', 'float64']:
                self.numeric_columns.append(col)
            elif dtype == 'object':
                # Check if it's a date
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    first_val = str(sample.iloc[0])
                    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', first_val):
                        self.date_columns.append(col)
                    elif df[col].nunique() < 50:
                        self.categorical_columns.append(col)
                    else:
                        self.text_columns.append(col)
                        
        return {
            'numeric': self.numeric_columns,
            'categorical': self.categorical_columns,
            'date': self.date_columns,
            'text': self.text_columns
        }
    
    def process_product_catalog(self, df):
        """Process product catalog data (like amazon.csv)."""
        processed = pd.DataFrame()
        
        # Extract price columns
        price_cols = [c for c in df.columns if 'price' in c.lower()]
        for col in price_cols:
            processed[col] = df[col].apply(self.parse_currency)
        
        # Extract rating
        rating_cols = [c for c in df.columns if 'rating' in c.lower()]
        for col in rating_cols[:2]:  # Limit to 2 rating columns
            if df[col].dtype == 'object':
                processed[col] = pd.to_numeric(df[col].str.extract(r'([\d.]+)')[0], errors='coerce')
            else:
                processed[col] = df[col]
        
        # Extract rating count
        count_cols = [c for c in df.columns if 'count' in c.lower()]
        for col in count_cols[:1]:
            if df[col].dtype == 'object':
                processed[col] = df[col].str.replace(',', '').astype(float, errors='ignore')
            else:
                processed[col] = df[col]
        
        # Extract discount percentage
        discount_cols = [c for c in df.columns if 'discount' in c.lower()]
        for col in discount_cols[:1]:
            if df[col].dtype == 'object':
                processed[col] = pd.to_numeric(df[col].str.replace('%', ''), errors='coerce')
            else:
                processed[col] = df[col]
        
        # Extract category features
        cat_cols = [c for c in df.columns if 'category' in c.lower()]
        for col in cat_cols[:1]:
            if col in df.columns:
                # Extract main category
                categories = df[col].str.split('|').str[0].fillna('Unknown')
                # One-hot encode top categories
                top_cats = categories.value_counts().head(10).index
                for cat in top_cats:
                    processed[f'cat_{cat[:15]}'] = (categories == cat).astype(int)
        
        processed = processed.dropna(how='all')
        return processed.fillna(processed.median(numeric_only=True))
    
    def process_transaction_data(self, df):
        """Process transaction data (like customer_shopping_data.csv)."""
        processed = pd.DataFrame()
        
        # Numeric columns
        for col in ['quantity', 'price', 'age']:
            matching = [c for c in df.columns if col in c.lower()]
            for m in matching[:1]:
                processed[col] = pd.to_numeric(df[m], errors='coerce')
        
        # Calculate total amount if quantity and price exist
        if 'quantity' in processed.columns and 'price' in processed.columns:
            processed['total_amount'] = processed['quantity'] * processed['price']
        
        # Gender encoding
        gender_cols = [c for c in df.columns if 'gender' in c.lower()]
        for col in gender_cols[:1]:
            processed['is_female'] = (df[col].str.lower() == 'female').astype(int)
        
        # Category encoding (top categories)
        cat_cols = [c for c in df.columns if 'category' in c.lower()]
        for col in cat_cols[:1]:
            top_cats = df[col].value_counts().head(8).index
            for cat in top_cats:
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', str(cat))[:15]
                processed[f'cat_{safe_name}'] = (df[col] == cat).astype(int)
        
        # Payment method encoding
        payment_cols = [c for c in df.columns if 'payment' in c.lower()]
        for col in payment_cols[:1]:
            methods = df[col].value_counts().head(5).index
            for method in methods:
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', str(method))[:12]
                processed[f'pay_{safe_name}'] = (df[col] == method).astype(int)
        
        # Date features
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        for col in date_cols[:1]:
            dates = self.parse_date(df[col])
            if dates.notna().sum() > len(dates) * 0.5:
                processed['month'] = dates.dt.month
                processed['day_of_week'] = dates.dt.dayofweek
                processed['is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        
        processed = processed.dropna(how='all')
        return processed.fillna(processed.median(numeric_only=True))
    
    def process_generic(self, df):
        """Process generic numeric data."""
        numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
        
        # Remove ID-like columns
        cols_to_keep = []
        for col in numeric_df.columns:
            col_lower = col.lower()
            is_id = 'id' in col_lower or col_lower.endswith('_id')
            is_unique = numeric_df[col].nunique() == len(numeric_df)
            if not (is_id or is_unique):
                cols_to_keep.append(col)
        
        if len(cols_to_keep) < 2:
            cols_to_keep = numeric_df.columns.tolist()[:5]
        
        return numeric_df[cols_to_keep].fillna(numeric_df[cols_to_keep].median())
    
    def preprocess(self, df, sample_size=None):
        """
        Main preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            sample_size: Optional sample size for large datasets
            
        Returns:
            Processed DataFrame ready for clustering
        """
        self.original_df = df.copy()
        
        # Sample large datasets
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} rows from {len(self.original_df)} total rows")
        
        # Detect dataset type
        self.detect_dataset_type(df)
        print(f"Detected dataset type: {self.dataset_type}")
        
        # Classify columns
        self.classify_columns(df)
        
        # Apply appropriate processing
        if self.dataset_type == 'product_catalog':
            self.processed_df = self.process_product_catalog(df)
        elif self.dataset_type == 'transaction_data':
            self.processed_df = self.process_transaction_data(df)
        else:
            self.processed_df = self.process_generic(df)
        
        # Ensure we have enough columns
        if self.processed_df.shape[1] < 2:
            print("Warning: Not enough features, falling back to generic processing")
            self.processed_df = self.process_generic(df)
        
        return self.processed_df
    
    def get_feature_names(self):
        """Return the feature names used for clustering."""
        if self.processed_df is not None:
            return self.processed_df.columns.tolist()
        return []
    
    def get_dataset_summary(self):
        """Return a summary of the dataset."""
        if self.original_df is None:
            return {}
        
        return {
            'total_rows': len(self.original_df),
            'total_columns': len(self.original_df.columns),
            'dataset_type': self.dataset_type,
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'features_for_clustering': self.processed_df.shape[1] if self.processed_df is not None else 0
        }
