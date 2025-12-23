"""
Sales Analytics Module
======================
Comprehensive sales and product analytics for e-commerce data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class SalesAnalyzer:
    """
    Sales analytics engine for analyzing revenue, products, and trends.
    """
    
    def __init__(self):
        self.df = None
        self.date_col = None
        self.category_col = None
        self.amount_col = None
        self.quantity_col = None
        self.product_col = None
        
    def auto_detect_columns(self, df):
        """Auto-detect relevant columns from dataframe."""
        columns = df.columns.tolist()
        col_lower = {c: c.lower() for c in columns}
        
        # Detect columns
        for col, lower in col_lower.items():
            if 'date' in lower or 'time' in lower:
                self.date_col = self.date_col or col
            if 'category' in lower:
                self.category_col = self.category_col or col
            if 'amount' in lower or 'price' in lower or 'total' in lower:
                self.amount_col = self.amount_col or col
            if 'quantity' in lower or 'qty' in lower:
                self.quantity_col = self.quantity_col or col
            if 'product' in lower or 'item' in lower:
                self.product_col = self.product_col or col
        
        return self
    
    def load_data(self, df):
        """Load and preprocess data for analysis."""
        self.df = df.copy()
        self.auto_detect_columns(df)
        return self
    
    def get_category_analysis(self, top_n=10):
        """Analyze sales by category."""
        if not self.category_col or self.category_col not in self.df.columns:
            return None
        
        df = self.df.copy()
        
        # Calculate total value
        if self.amount_col and self.quantity_col:
            if self.amount_col in df.columns and self.quantity_col in df.columns:
                df['value'] = pd.to_numeric(df[self.amount_col], errors='coerce') * \
                              pd.to_numeric(df[self.quantity_col], errors='coerce')
            else:
                df['value'] = pd.to_numeric(df.get(self.amount_col, 1), errors='coerce')
        elif self.amount_col and self.amount_col in df.columns:
            df['value'] = pd.to_numeric(df[self.amount_col], errors='coerce')
        else:
            df['value'] = 1
        
        # Quantity
        if self.quantity_col and self.quantity_col in df.columns:
            df['qty'] = pd.to_numeric(df[self.quantity_col], errors='coerce')
        else:
            df['qty'] = 1
        
        # Aggregate by category
        category_stats = df.groupby(self.category_col).agg({
            'value': ['sum', 'mean', 'count'],
            'qty': 'sum'
        }).round(2)
        
        category_stats.columns = ['Total_Revenue', 'Avg_Transaction', 'Transaction_Count', 'Total_Quantity']
        category_stats['Revenue_Percentage'] = (category_stats['Total_Revenue'] / 
                                                  category_stats['Total_Revenue'].sum() * 100).round(1)
        
        return category_stats.sort_values('Total_Revenue', ascending=False).head(top_n)
    
    def get_time_series_data(self, freq='M'):
        """Get time series data for trend analysis."""
        if not self.date_col or self.date_col not in self.df.columns:
            return None
        
        df = self.df.copy()
        
        # Parse dates
        date_formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y']
        dates = None
        
        for fmt in date_formats:
            try:
                dates = pd.to_datetime(df[self.date_col], format=fmt, errors='coerce')
                if dates.notna().sum() > len(dates) * 0.5:
                    break
            except:
                continue
        
        if dates is None:
            dates = pd.to_datetime(df[self.date_col], errors='coerce')
        
        df['parsed_date'] = dates
        df = df.dropna(subset=['parsed_date'])
        
        if len(df) == 0:
            return None
        
        # Calculate value
        if self.amount_col and self.quantity_col:
            if self.amount_col in df.columns and self.quantity_col in df.columns:
                df['value'] = pd.to_numeric(df[self.amount_col], errors='coerce') * \
                              pd.to_numeric(df[self.quantity_col], errors='coerce')
            else:
                df['value'] = pd.to_numeric(df.get(self.amount_col, 1), errors='coerce')
        elif self.amount_col and self.amount_col in df.columns:
            df['value'] = pd.to_numeric(df[self.amount_col], errors='coerce')
        else:
            df['value'] = 1
        
        # Aggregate by time period
        df.set_index('parsed_date', inplace=True)
        
        time_series = df.resample(freq).agg({
            'value': ['sum', 'count']
        })
        time_series.columns = ['Revenue', 'Transactions']
        time_series['Avg_Transaction'] = (time_series['Revenue'] / time_series['Transactions']).round(2)
        
        return time_series.reset_index()
    
    def get_price_distribution(self, bins=20):
        """Get price/amount distribution statistics."""
        if not self.amount_col or self.amount_col not in self.df.columns:
            return None
        
        prices = pd.to_numeric(self.df[self.amount_col], errors='coerce').dropna()
        
        stats = {
            'min': prices.min(),
            'max': prices.max(),
            'mean': prices.mean(),
            'median': prices.median(),
            'std': prices.std(),
            'q25': prices.quantile(0.25),
            'q75': prices.quantile(0.75),
            'count': len(prices)
        }
        
        # Create histogram data
        hist, bin_edges = np.histogram(prices, bins=bins)
        
        return {
            'statistics': stats,
            'histogram': {'counts': hist.tolist(), 'bin_edges': bin_edges.tolist()}
        }
    
    def get_payment_analysis(self):
        """Analyze payment methods."""
        payment_cols = [c for c in self.df.columns if 'payment' in c.lower()]
        
        if not payment_cols:
            return None
        
        payment_col = payment_cols[0]
        df = self.df.copy()
        
        # Calculate value
        if self.amount_col and self.quantity_col:
            if self.amount_col in df.columns and self.quantity_col in df.columns:
                df['value'] = pd.to_numeric(df[self.amount_col], errors='coerce') * \
                              pd.to_numeric(df[self.quantity_col], errors='coerce')
            else:
                df['value'] = pd.to_numeric(df.get(self.amount_col, 1), errors='coerce')
        elif self.amount_col and self.amount_col in df.columns:
            df['value'] = pd.to_numeric(df[self.amount_col], errors='coerce')
        else:
            df['value'] = 1
        
        payment_stats = df.groupby(payment_col).agg({
            'value': ['sum', 'mean', 'count']
        }).round(2)
        
        payment_stats.columns = ['Total_Revenue', 'Avg_Transaction', 'Count']
        payment_stats['Percentage'] = (payment_stats['Count'] / payment_stats['Count'].sum() * 100).round(1)
        
        return payment_stats.sort_values('Total_Revenue', ascending=False)
    
    def get_customer_analysis(self):
        """Analyze customer demographics if available."""
        df = self.df.copy()
        analysis = {}
        
        # Gender analysis
        gender_cols = [c for c in df.columns if 'gender' in c.lower()]
        if gender_cols:
            gender_counts = df[gender_cols[0]].value_counts()
            analysis['gender'] = gender_counts.to_dict()
        
        # Age analysis
        age_cols = [c for c in df.columns if 'age' in c.lower()]
        if age_cols:
            ages = pd.to_numeric(df[age_cols[0]], errors='coerce').dropna()
            # Create age groups
            bins = [0, 25, 35, 45, 55, 65, 100]
            labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
            analysis['age_groups'] = age_groups.value_counts().sort_index().to_dict()
            analysis['age_stats'] = {
                'min': int(ages.min()),
                'max': int(ages.max()),
                'mean': round(ages.mean(), 1),
                'median': int(ages.median())
            }
        
        # Shopping mall/location analysis
        mall_cols = [c for c in df.columns if 'mall' in c.lower() or 'store' in c.lower() or 'location' in c.lower()]
        if mall_cols:
            mall_counts = df[mall_cols[0]].value_counts().head(10)
            analysis['locations'] = mall_counts.to_dict()
        
        return analysis if analysis else None
    
    def get_summary_stats(self):
        """Get overall summary statistics."""
        df = self.df.copy()
        
        summary = {
            'total_records': len(df),
            'columns': len(df.columns)
        }
        
        # Unique customers
        customer_cols = [c for c in df.columns if 'customer' in c.lower()]
        if customer_cols:
            summary['unique_customers'] = df[customer_cols[0]].nunique()
        
        # Date range
        if self.date_col and self.date_col in df.columns:
            dates = pd.to_datetime(df[self.date_col], errors='coerce').dropna()
            if len(dates) > 0:
                summary['date_range'] = {
                    'start': dates.min().strftime('%Y-%m-%d'),
                    'end': dates.max().strftime('%Y-%m-%d')
                }
        
        # Categories
        if self.category_col and self.category_col in df.columns:
            summary['unique_categories'] = df[self.category_col].nunique()
        
        # Total revenue
        if self.amount_col and self.amount_col in df.columns:
            amounts = pd.to_numeric(df[self.amount_col], errors='coerce')
            if self.quantity_col and self.quantity_col in df.columns:
                quantities = pd.to_numeric(df[self.quantity_col], errors='coerce')
                total_revenue = (amounts * quantities).sum()
            else:
                total_revenue = amounts.sum()
            summary['total_revenue'] = round(total_revenue, 2)
        
        return summary
