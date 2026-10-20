"""
Technical Indicators Module for Stock Market Analysis
Calculates various technical indicators for stock price analysis
"""

import pandas as pd
import numpy as np
from typing import Optional


class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data
        
        Args:
            data: DataFrame with columns Date, Open, High, Low, Close, Volume
        """
        self.data = data.copy()
        self.indicators = {}
    
    def calculate_moving_averages(self, periods: list = [50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages (SMA)
        
        Args:
            periods: List of periods for moving averages
            
        Returns:
            DataFrame with added MA columns
        """
        for period in periods:
            self.data[f'MA{period}'] = self.data['Close'].rolling(window=period).mean()
        
        self.indicators['moving_averages'] = True
        return self.data
    
    def calculate_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            period: Period for RSI calculation (default: 14)
            
        Returns:
            DataFrame with added RSI column
        """
        # Calculate price changes
        delta = self.data['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        self.indicators['rsi'] = True
        return self.data
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            fast_period: Period for fast EMA (default: 12)
            slow_period: Period for slow EMA (default: 26)
            signal_period: Period for signal line (default: 9)
            
        Returns:
            DataFrame with added MACD, Signal, and Histogram columns
        """
        # Calculate EMAs
        ema_fast = self.data['Close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        self.data['MACD'] = ema_fast - ema_slow
        
        # Calculate Signal line
        self.data['Signal'] = self.data['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Histogram
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal']
        
        self.indicators['macd'] = True
        return self.data
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            period: Period for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        # Calculate middle band (SMA)
        self.data['BB_Middle'] = self.data['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std = self.data['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        self.data['BB_Upper'] = self.data['BB_Middle'] + (std * std_dev)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (std * std_dev)
        
        # Calculate bandwidth
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        
        self.indicators['bollinger_bands'] = True
        return self.data
    
    def calculate_volume_ma(self, period: int = 20) -> pd.DataFrame:
        """
        Calculate Volume Moving Average
        
        Args:
            period: Period for volume MA (default: 20)
            
        Returns:
            DataFrame with added Volume MA column
        """
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=period).mean()
        
        self.indicators['volume_ma'] = True
        return self.data
    
    def calculate_obv(self) -> pd.DataFrame:
        """
        Calculate On Balance Volume (OBV)
        
        Returns:
            DataFrame with added OBV column
        """
        # Calculate price direction
        price_change = self.data['Close'].diff()
        
        # Initialize OBV
        obv = [0]
        
        # Calculate OBV values
        for i in range(1, len(self.data)):
            if price_change[i] > 0:
                obv.append(obv[-1] + self.data['Volume'].iloc[i])
            elif price_change[i] < 0:
                obv.append(obv[-1] - self.data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        self.data['OBV'] = obv
        
        self.indicators['obv'] = True
        return self.data
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Returns:
            DataFrame with all indicators calculated
        """
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_volume_ma()
        self.calculate_obv()
        
        return self.data
    
    def get_indicator_data(self, indicator_name: str) -> Optional[pd.Series]:
        """
        Get data for a specific indicator
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Series with indicator data or None if not found
        """
        indicator_map = {
            'MA50': 'MA50',
            'MA200': 'MA200',
            'RSI': 'RSI',
            'MACD': 'MACD',
            'Signal': 'Signal',
            'MACD_Histogram': 'MACD_Histogram',
            'BB_Upper': 'BB_Upper',
            'BB_Middle': 'BB_Middle',
            'BB_Lower': 'BB_Lower',
            'Volume_MA': 'Volume_MA',
            'OBV': 'OBV'
        }
        
        column_name = indicator_map.get(indicator_name)
        
        if column_name and column_name in self.data.columns:
            return self.data[column_name]
        
        return None
    
    def get_rsi_signal(self) -> str:
        """
        Get RSI-based trading signal
        
        Returns:
            'OVERBOUGHT', 'OVERSOLD', or 'NEUTRAL'
        """
        if 'RSI' not in self.data.columns:
            return 'NEUTRAL'
        
        latest_rsi = self.data['RSI'].iloc[-1]
        
        if latest_rsi > 70:
            return 'OVERBOUGHT'
        elif latest_rsi < 30:
            return 'OVERSOLD'
        else:
            return 'NEUTRAL'
    
    def get_macd_signal(self) -> str:
        """
        Get MACD-based trading signal
        
        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if 'MACD' not in self.data.columns or 'Signal' not in self.data.columns:
            return 'NEUTRAL'
        
        latest_macd = self.data['MACD'].iloc[-1]
        latest_signal = self.data['Signal'].iloc[-1]
        
        if latest_macd > latest_signal:
            return 'BULLISH'
        elif latest_macd < latest_signal:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def get_bollinger_position(self) -> str:
        """
        Get price position relative to Bollinger Bands
        
        Returns:
            'NEAR_UPPER', 'NEAR_LOWER', or 'MIDDLE'
        """
        if 'BB_Upper' not in self.data.columns or 'BB_Lower' not in self.data.columns:
            return 'MIDDLE'
        
        latest_price = self.data['Close'].iloc[-1]
        latest_upper = self.data['BB_Upper'].iloc[-1]
        latest_lower = self.data['BB_Lower'].iloc[-1]
        
        band_width = latest_upper - latest_lower
        
        if latest_price > latest_upper - (band_width * 0.1):
            return 'NEAR_UPPER'
        elif latest_price < latest_lower + (band_width * 0.1):
            return 'NEAR_LOWER'
        else:
            return 'MIDDLE'
    
    def get_all_signals(self) -> dict:
        """
        Get all trading signals
        
        Returns:
            Dictionary with all signals
        """
        return {
            'RSI': self.get_rsi_signal(),
            'MACD': self.get_macd_signal(),
            'Bollinger': self.get_bollinger_position()
        }
