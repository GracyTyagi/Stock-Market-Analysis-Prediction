"""
Data Preprocessing Module for Stock Market Analysis
Handles CSV upload, data cleaning, and basic statistics
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataPreprocessor:
    """Class for preprocessing stock market data"""
    
    def __init__(self):
        self.data = None
        self.stock_name = None
        self.available_stocks = []
    
    def load_csv(self, uploaded_file) -> Tuple[bool, str]:
        """
        Load CSV file and perform initial validation
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Read CSV file
            self.data = pd.read_csv(uploaded_file)
            
            # Check if required columns exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}"
    
    def preprocess_data(self) -> Tuple[bool, str]:
        """
        Preprocess the loaded data:
        - Convert Date column to datetime
        - Handle missing values
        - Sort by date
        - Extract stock names if available
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if self.data is None:
                return False, "No data loaded. Please upload a CSV file first."
            
            # Convert Date column to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            
            # Check for invalid dates
            if self.data['Date'].isnull().any():
                return False, "Some dates could not be parsed. Please check the Date column format."
            
            # Sort data by date
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            
            # Handle missing values in numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Forward fill missing values
            self.data[numeric_columns] = self.data[numeric_columns].fillna(method='ffill')
            
            # Backward fill if any remaining NaN values
            self.data[numeric_columns] = self.data[numeric_columns].fillna(method='bfill')
            
            # Check if there are still missing values
            if self.data[numeric_columns].isnull().any().any():
                return False, "Unable to handle missing values. Please check your data."
            
            # Check for multiple stocks (if 'Name' or 'Symbol' column exists)
            if 'Name' in self.data.columns:
                self.available_stocks = self.data['Name'].unique().tolist()
            elif 'Symbol' in self.data.columns:
                self.available_stocks = self.data['Symbol'].unique().tolist()
            else:
                # Single stock dataset
                self.available_stocks = ['Single Stock']
                self.stock_name = 'Single Stock'
            
            return True, "Data preprocessed successfully"
            
        except Exception as e:
            return False, f"Error preprocessing data: {str(e)}"
    
    def select_stock(self, stock_name: str) -> Tuple[bool, str]:
        """
        Select a specific stock from multi-stock dataset
        
        Args:
            stock_name: Name of the stock to select
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if stock_name == 'Single Stock':
                self.stock_name = 'Single Stock'
                return True, "Single stock dataset selected"
            
            if 'Name' in self.data.columns:
                self.data = self.data[self.data['Name'] == stock_name].copy()
            elif 'Symbol' in self.data.columns:
                self.data = self.data[self.data['Symbol'] == stock_name].copy()
            else:
                return False, "No stock identifier column found in dataset"
            
            self.stock_name = stock_name
            self.data = self.data.reset_index(drop=True)
            
            return True, f"Stock '{stock_name}' selected successfully"
            
        except Exception as e:
            return False, f"Error selecting stock: {str(e)}"
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the preprocessed data
        
        Returns:
            DataFrame or None if no data loaded
        """
        return self.data
    
    def get_basic_statistics(self) -> Optional[pd.DataFrame]:
        """
        Calculate basic statistics for the dataset
        
        Returns:
            DataFrame with statistics or None if no data loaded
        """
        if self.data is None:
            return None
        
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        stats = self.data[numeric_columns].describe()
        
        return stats
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary with dataset information
        """
        if self.data is None:
            return {}
        
        info = {
            'total_rows': len(self.data),
            'date_range': {
                'start': self.data['Date'].min().strftime('%Y-%m-%d'),
                'end': self.data['Date'].max().strftime('%Y-%m-%d')
            },
            'columns': list(self.data.columns),
            'stock_name': self.stock_name,
            'available_stocks': self.available_stocks,
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        return info
    
    def get_latest_data(self) -> Optional[dict]:
        """
        Get the latest (most recent) data point
        
        Returns:
            Dictionary with latest data or None if no data loaded
        """
        if self.data is None:
            return None
        
        latest = self.data.iloc[-1].to_dict()
        
        # Calculate daily change
        if len(self.data) > 1:
            previous = self.data.iloc[-2]
            latest['daily_change'] = latest['Close'] - previous['Close']
            latest['daily_change_pct'] = (latest['daily_change'] / previous['Close']) * 100
        else:
            latest['daily_change'] = 0
            latest['daily_change_pct'] = 0
        
        return latest
