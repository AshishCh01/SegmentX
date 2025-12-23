"""
RFM Analysis Module
====================
Recency-Frequency-Monetary analysis for customer segmentation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class RFMAnalyzer:
    """
    RFM (Recency, Frequency, Monetary) analyzer for customer segmentation.
    
    RFM is a marketing analysis technique used to identify best customers
    based on their transaction history.
    """
    
    SEGMENT_NAMES = {
        'Champions': (4, 5, 4, 5, 4, 5),       # High R, F, M
        'Loyal Customers': (2, 5, 3, 5, 3, 5), # Medium-High R, High F, High M
        'Potential Loyalists': (3, 5, 1, 3, 1, 3),
        'Recent Customers': (4, 5, 0, 1, 0, 1),
        'Promising': (3, 4, 0, 1, 0, 1),
        'Needs Attention': (2, 3, 2, 3, 2, 3),
        'About to Sleep': (2, 3, 0, 2, 0, 2),
        'At Risk': (0, 2, 2, 5, 2, 5),
        'Cant Lose Them': (0, 1, 4, 5, 4, 5),
        'Hibernating': (1, 2, 1, 2, 1, 2),
        'Lost': (0, 2, 0, 2, 0, 2)
    }
    
    def __init__(self, date_col=None, customer_col=None, amount_col=None, quantity_col=None):
        self.date_col = date_col
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.quantity_col = quantity_col
        self.rfm_df = None
        self.rfm_scores = None
        self.analysis_date = None
        
    def auto_detect_columns(self, df):
        """Auto-detect RFM-relevant columns from dataframe."""
        columns = df.columns.tolist()
        col_lower = {c: c.lower() for c in columns}
        
        # Detect customer column
        if not self.customer_col:
            for col, lower in col_lower.items():
                if 'customer' in lower or 'client' in lower or 'user' in lower:
                    self.customer_col = col
                    break
            if not self.customer_col:
                for col, lower in col_lower.items():
                    if '_id' in lower and 'invoice' not in lower:
                        self.customer_col = col
                        break
        
        # Detect date column
        if not self.date_col:
            for col, lower in col_lower.items():
                if 'date' in lower or 'time' in lower:
                    self.date_col = col
                    break
        
        # Detect amount/price column
        if not self.amount_col:
            for col, lower in col_lower.items():
                if 'amount' in lower or 'total' in lower or 'price' in lower:
                    self.amount_col = col
                    break
        
        # Detect quantity column
        if not self.quantity_col:
            for col, lower in col_lower.items():
                if 'quantity' in lower or 'qty' in lower:
                    self.quantity_col = col
                    break
        
        return {
            'customer_col': self.customer_col,
            'date_col': self.date_col,
            'amount_col': self.amount_col,
            'quantity_col': self.quantity_col
        }
    
    def parse_dates(self, df):
        """Parse date column with multiple format attempts."""
        if self.date_col not in df.columns:
            return None
        
        date_formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y']
        
        for fmt in date_formats:
            try:
                dates = pd.to_datetime(df[self.date_col], format=fmt, errors='coerce')
                if dates.notna().sum() > len(dates) * 0.5:
                    return dates
            except:
                continue
        
        return pd.to_datetime(df[self.date_col], errors='coerce')
    
    def calculate_rfm(self, df, analysis_date=None):
        """
        Calculate RFM values for each customer.
        
        Args:
            df: Transaction dataframe
            analysis_date: Reference date for recency calculation (defaults to max date + 1)
            
        Returns:
            DataFrame with RFM values per customer
        """
        # Auto-detect columns if needed
        self.auto_detect_columns(df)
        
        # Validate required columns
        if not self.customer_col or not self.date_col:
            raise ValueError("Could not detect customer and date columns for RFM analysis")
        
        # Parse dates
        df = df.copy()
        df['parsed_date'] = self.parse_dates(df)
        df = df.dropna(subset=['parsed_date'])
        
        if len(df) == 0:
            raise ValueError("No valid dates found in the data")
        
        # Set analysis date
        if analysis_date is None:
            self.analysis_date = df['parsed_date'].max() + pd.Timedelta(days=1)
        else:
            self.analysis_date = pd.to_datetime(analysis_date)
        
        # Calculate total amount if not available
        if self.amount_col and self.quantity_col:
            if self.amount_col in df.columns and self.quantity_col in df.columns:
                df['transaction_value'] = pd.to_numeric(df[self.amount_col], errors='coerce') * \
                                          pd.to_numeric(df[self.quantity_col], errors='coerce')
            elif self.amount_col in df.columns:
                df['transaction_value'] = pd.to_numeric(df[self.amount_col], errors='coerce')
            else:
                df['transaction_value'] = 1  # Count-based
        elif self.amount_col and self.amount_col in df.columns:
            df['transaction_value'] = pd.to_numeric(df[self.amount_col], errors='coerce')
        else:
            df['transaction_value'] = 1
        
        # Group by customer
        rfm = df.groupby(self.customer_col).agg({
            'parsed_date': 'max',          # Most recent purchase
            self.customer_col: 'count',     # Frequency
            'transaction_value': 'sum'      # Total spend
        }).reset_index(drop=True)
        
        # Rename columns
        rfm = df.groupby(self.customer_col).agg({
            'parsed_date': 'max',
            'transaction_value': ['count', 'sum']
        })
        rfm.columns = ['last_purchase', 'frequency', 'monetary']
        rfm = rfm.reset_index()
        
        # Calculate recency
        rfm['recency'] = (self.analysis_date - rfm['last_purchase']).dt.days
        
        # Store RFM data
        self.rfm_df = rfm[[self.customer_col, 'recency', 'frequency', 'monetary']]
        
        return self.rfm_df
    
    def calculate_rfm_scores(self, n_bins=5):
        """
        Calculate RFM scores (1-5 scale) using quantile binning.
        
        Args:
            n_bins: Number of score bins (default 5)
            
        Returns:
            DataFrame with RFM scores
        """
        if self.rfm_df is None:
            raise ValueError("RFM values not calculated. Call calculate_rfm first.")
        
        rfm = self.rfm_df.copy()
        
        # Score Recency (lower is better, so reverse)
        rfm['R_Score'] = pd.qcut(rfm['recency'].rank(method='first'), q=n_bins, labels=range(n_bins, 0, -1))
        
        # Score Frequency (higher is better)
        rfm['F_Score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=n_bins, labels=range(1, n_bins + 1))
        
        # Score Monetary (higher is better)
        rfm['M_Score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=n_bins, labels=range(1, n_bins + 1))
        
        # Convert to int
        for col in ['R_Score', 'F_Score', 'M_Score']:
            rfm[col] = rfm[col].astype(int)
        
        # Combined RFM Score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        self.rfm_scores = rfm
        return rfm
    
    def assign_segments(self):
        """Assign customer segments based on RFM scores."""
        if self.rfm_scores is None:
            self.calculate_rfm_scores()
        
        rfm = self.rfm_scores.copy()
        
        def get_segment(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'Recent Customers'
            elif r >= 3 and f <= 2 and m <= 2:
                return 'Promising'
            elif r <= 2 and f >= 3:
                return 'At Risk'
            elif r <= 2 and f >= 4 and m >= 4:
                return 'Cant Lose Them'
            elif r <= 2 and f <= 2:
                return 'Lost'
            elif r == 3 and f == 3:
                return 'Needs Attention'
            else:
                return 'Potential Loyalists'
        
        rfm['Segment'] = rfm.apply(get_segment, axis=1)
        self.rfm_scores = rfm
        
        return rfm
    
    def get_segment_summary(self):
        """Get summary statistics for each segment."""
        if self.rfm_scores is None or 'Segment' not in self.rfm_scores.columns:
            self.assign_segments()
        
        summary = self.rfm_scores.groupby('Segment').agg({
            self.customer_col: 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['mean', 'sum'],
            'RFM_Total': 'mean'
        }).round(2)
        
        summary.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue', 'Avg_RFM_Score']
        summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(1)
        
        return summary.sort_values('Avg_RFM_Score', ascending=False)
    
    def get_rfm_for_clustering(self):
        """Get standardized RFM values ready for clustering."""
        if self.rfm_scores is None:
            self.calculate_rfm_scores()
        
        rfm_cluster = self.rfm_scores[['recency', 'frequency', 'monetary', 'R_Score', 'F_Score', 'M_Score']].copy()
        
        # Log transform monetary for better distribution
        rfm_cluster['monetary_log'] = np.log1p(rfm_cluster['monetary'])
        
        return rfm_cluster
