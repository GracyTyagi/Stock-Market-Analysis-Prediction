"""
Database Module for Prediction History
SQLite database operations for storing and retrieving prediction records
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List


class PredictionDatabase:
    """Handle SQLite database operations for prediction history"""
    
    def __init__(self, db_path: str = "prediction_history.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database table if it doesn't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_datetime TEXT NOT NULL,
                    stock_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    trading_signal TEXT NOT NULL,
                    mae REAL NOT NULL,
                    rmse REAL NOT NULL,
                    r2_score REAL NOT NULL
                )
            """)
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            raise
    
    def insert_prediction(
        self,
        stock_name: str,
        model_name: str,
        current_price: float,
        predicted_price: float,
        trading_signal: str,
        mae: float,
        rmse: float,
        r2_score: float
    ) -> bool:
        """
        Insert a new prediction record into the database
        
        Args:
            stock_name: Name of the stock
            model_name: Name of the ML model used
            current_price: Current stock price
            predicted_price: Predicted stock price
            trading_signal: Trading signal (BUY/SELL/HOLD)
            mae: Mean Absolute Error
            rmse: Root Mean Square Error
            r2_score: R² Score
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prediction_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO prediction_history 
                (prediction_datetime, stock_name, model_name, current_price, 
                 predicted_price, trading_signal, mae, rmse, r2_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_datetime,
                stock_name,
                model_name,
                current_price,
                predicted_price,
                trading_signal,
                mae,
                rmse,
                r2_score
            ))
            
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"Error inserting prediction: {e}")
            return False
    
    def get_latest_predictions(self, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Retrieve the latest N predictions from the database
        
        Args:
            limit: Number of records to retrieve
            
        Returns:
            DataFrame with prediction records or None if error
        """
        try:
            query = """
                SELECT 
                    prediction_datetime as "Date & Time",
                    stock_name as "Stock",
                    model_name as "Model",
                    current_price as "Current Price",
                    predicted_price as "Predicted Price",
                    trading_signal as "Signal"
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            return df
            
        except sqlite3.Error as e:
            print(f"Error retrieving predictions: {e}")
            return None
    
    def get_all_predictions(self) -> Optional[pd.DataFrame]:
        """
        Retrieve all predictions from the database
        
        Returns:
            DataFrame with all prediction records or None if error
        """
        try:
            query = """
                SELECT 
                    prediction_datetime as "Date & Time",
                    stock_name as "Stock",
                    model_name as "Model",
                    current_price as "Current Price",
                    predicted_price as "Predicted Price",
                    trading_signal as "Signal",
                    mae as "MAE",
                    rmse as "RMSE",
                    r2_score as "R² Score"
                FROM prediction_history
                ORDER BY id DESC
            """
            
            df = pd.read_sql_query(query, self.conn)
            return df
            
        except sqlite3.Error as e:
            print(f"Error retrieving all predictions: {e}")
            return None
    
    def clear_history(self) -> bool:
        """
        Clear all prediction records from the database
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM prediction_history")
            self.conn.commit()
            return True
            
        except sqlite3.Error as e:
            print(f"Error clearing history: {e}")
            return False
    
    def get_history_count(self) -> int:
        """
        Get the total number of prediction records
        
        Returns:
            Number of records in the database
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prediction_history")
            count = cursor.fetchone()[0]
            return count
            
        except sqlite3.Error as e:
            print(f"Error getting history count: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()
