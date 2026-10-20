"""
Machine Learning Prediction Module for Stock Market Analysis
Implements various ML models for stock price prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Optional, Dict
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. XGBoost model will not be available.")


class StockPredictor:
    """Class for stock price prediction using ML models"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data including indicators
        
        Args:
            data: DataFrame with stock data and technical indicators
        """
        self.data = data.copy()
        self.model = None
        self.feature_columns = []
        self.target_column = 'Close'
        self.scaler = None
        self.predictions = None
        self.metrics = {}
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML models
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Define feature columns
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA50', 'MA200', 'RSI', 'MACD', 'Signal',
            'BB_Upper', 'BB_Lower', 'OBV'
        ]
        
        # Filter only available columns
        available_features = [col for col in self.feature_columns if col in self.data.columns]
        
        if not available_features:
            raise ValueError("No valid feature columns found in data")
        
        # Create features DataFrame
        features = self.data[available_features].copy()
        
        # Handle missing values (forward fill, then backward fill)
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # If still have NaN values, drop them
        features = features.dropna()
        
        # Create target (next day's closing price)
        target = self.data['Close'].shift(-1).loc[features.index]
        
        # Remove last row (no target available)
        features = features[:-1]
        target = target[:-1]
        
        return features, target
    
    def split_data(self, features: pd.DataFrame, target: pd.Series, 
                   test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets using time-series split
        
        Args:
            features: Feature DataFrame
            target: Target Series
            test_size: Proportion of data for testing (default: 0.2)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Time-series split: first 80% for training, last 20% for testing
        split_idx = int(len(features) * (1 - test_size))
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train Linear Regression model
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           n_estimators: int = 100, random_state: int = 42) -> None:
        """
        Train Random Forest Regressor
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees (default: 100)
            random_state: Random seed (default: 42)
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     n_estimators: int = 100, learning_rate: float = 0.1,
                     max_depth: int = 6, random_state: int = 42) -> None:
        """
        Train XGBoost Regressor
        
        Args:
            X_train: Training features
            y_train: Training target
            n_estimators: Number of trees (default: 100)
            learning_rate: Learning rate (default: 0.1)
            max_depth: Maximum tree depth (default: 6)
            random_state: Random seed (default: 42)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it using: pip install xgboost")
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train a model first.")
        
        self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def calculate_metrics(self, y_test: pd.Series, predictions: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_test: Actual values
            predictions: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        return self.metrics
    
    def predict_next_day(self) -> Tuple[float, Dict]:
        """
        Predict next day's closing price using the latest data
        
        Returns:
            Tuple of (predicted_price, metrics)
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train a model first.")
        
        # Get latest data point
        latest_data = self.data.iloc[-1][self.feature_columns].to_frame().T
        
        # Handle missing values
        latest_data = latest_data.fillna(method='ffill').fillna(method='bfill')
        
        # Filter only available columns
        available_features = [col for col in self.feature_columns if col in latest_data.columns]
        latest_data = latest_data[available_features]
        
        # Make prediction
        next_day_price = self.model.predict(latest_data)[0]
        
        return next_day_price, self.metrics
    
    def run_prediction(self, model_type: str = 'random_forest') -> Dict:
        """
        Run complete prediction pipeline
        
        Args:
            model_type: Type of model to use ('linear_regression', 'random_forest', 'xgboost')
            
        Returns:
            Dictionary with prediction results and metrics
        """
        try:
            # Prepare features
            features, target = self.prepare_features()
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(features, target)
            
            # Train model
            if model_type == 'linear_regression':
                self.train_linear_regression(X_train, y_train)
            elif model_type == 'random_forest':
                self.train_random_forest(X_train, y_train)
            elif model_type == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost not available. Using Random Forest instead.")
                self.train_xgboost(X_train, y_train)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Make predictions
            predictions = self.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, predictions)
            
            # Predict next day
            next_day_price, _ = self.predict_next_day()
            
            return {
                'success': True,
                'model_type': model_type,
                'next_day_price': next_day_price,
                'metrics': metrics,
                'predictions': predictions,
                'actual': y_test.values,
                'test_dates': self.data['Date'].iloc[X_test.index].values
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_trading_signal(self, current_price: float, predicted_price: float,
                          rsi_signal: str = 'NEUTRAL', macd_signal: str = 'NEUTRAL',
                          bollinger_position: str = 'MIDDLE') -> str:
        """
        Generate trading signal based on prediction and technical indicators
        
        Args:
            current_price: Current closing price
            predicted_price: Predicted next day price
            rsi_signal: RSI signal ('OVERBOUGHT', 'OVERSOLD', 'NEUTRAL')
            macd_signal: MACD signal ('BULLISH', 'BEARISH', 'NEUTRAL')
            bollinger_position: Bollinger Bands position ('NEAR_UPPER', 'NEAR_LOWER', 'MIDDLE')
            
        Returns:
            Trading signal ('BUY', 'SELL', 'HOLD')
        """
        buy_score = 0
        sell_score = 0
        
        # Price prediction
        if predicted_price > current_price * 1.01:  # Predicted increase > 1%
            buy_score += 2
        elif predicted_price < current_price * 0.99:  # Predicted decrease > 1%
            sell_score += 2
        
        # RSI signal
        if rsi_signal == 'OVERSOLD':
            buy_score += 2
        elif rsi_signal == 'OVERBOUGHT':
            sell_score += 2
        
        # MACD signal
        if macd_signal == 'BULLISH':
            buy_score += 1
        elif macd_signal == 'BEARISH':
            sell_score += 1
        
        # Bollinger Bands position
        if bollinger_position == 'NEAR_LOWER':
            buy_score += 1
        elif bollinger_position == 'NEAR_UPPER':
            sell_score += 1
        
        # Determine final signal
        if buy_score >= 3:
            return 'BUY'
        elif sell_score >= 3:
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model (if available)
        
        Returns:
            DataFrame with feature importance or None if not available
        """
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_columns
            
            # Filter to match actual features used
            available_features = [col for col in feature_names if col in self.data.columns]
            
            df = pd.DataFrame({
                'Feature': available_features[:len(importance)],
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return df
        
        return None
    
    def compare_models(self) -> Optional[pd.DataFrame]:
        """
        Compare all three models (Linear Regression, Random Forest, XGBoost)
        
        Returns:
            DataFrame with model comparison metrics or None if failed
        """
        try:
            # Prepare features
            features, target = self.prepare_features()
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(features, target)
            
            results = []
            
            # Linear Regression
            try:
                self.train_linear_regression(X_train, y_train)
                lr_predictions = self.predict(X_test)
                lr_metrics = self.calculate_metrics(y_test, lr_predictions)
                results.append({
                    'Model': 'Linear Regression',
                    'MAE': lr_metrics['MAE'],
                    'RMSE': lr_metrics['RMSE'],
                    'R2 Score': lr_metrics['R2']
                })
            except Exception as e:
                print(f"Linear Regression failed: {e}")
            
            # Random Forest
            try:
                self.train_random_forest(X_train, y_train)
                rf_predictions = self.predict(X_test)
                rf_metrics = self.calculate_metrics(y_test, rf_predictions)
                results.append({
                    'Model': 'Random Forest',
                    'MAE': rf_metrics['MAE'],
                    'RMSE': rf_metrics['RMSE'],
                    'R2 Score': rf_metrics['R2']
                })
            except Exception as e:
                print(f"Random Forest failed: {e}")
            
            # XGBoost
            if XGBOOST_AVAILABLE:
                try:
                    self.train_xgboost(X_train, y_train)
                    xgb_predictions = self.predict(X_test)
                    xgb_metrics = self.calculate_metrics(y_test, xgb_predictions)
                    results.append({
                        'Model': 'XGBoost',
                        'MAE': xgb_metrics['MAE'],
                        'RMSE': xgb_metrics['RMSE'],
                        'R2 Score': xgb_metrics['R2']
                    })
                except Exception as e:
                    print(f"XGBoost failed: {e}")
            
            if results:
                comparison_df = pd.DataFrame(results)
                # Sort by R2 Score (descending)
                comparison_df = comparison_df.sort_values('R2 Score', ascending=False)
                return comparison_df
            
            return None
            
        except Exception as e:
            print(f"Model comparison failed: {e}")
            return None
    
    def get_prediction_comparison_dataframe(self, model_type: str = 'random_forest') -> Optional[pd.DataFrame]:
        """
        Get a dataframe comparing actual vs predicted prices
        
        Args:
            model_type: Type of model to use
            
        Returns:
            DataFrame with Date, Actual Close Price, Predicted Close Price
        """
        try:
            # Run prediction
            results = self.run_prediction(model_type)
            
            if not results['success']:
                return None
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Date': results['test_dates'],
                'Actual Close Price': results['actual'],
                'Predicted Close Price': results['predictions']
            })
            
            # Add error column
            comparison_df['Error'] = abs(comparison_df['Actual Close Price'] - comparison_df['Predicted Close Price'])
            comparison_df['Error %'] = (comparison_df['Error'] / comparison_df['Actual Close Price']) * 100
            
            return comparison_df
            
        except Exception as e:
            print(f"Failed to get prediction comparison: {e}")
            return None
