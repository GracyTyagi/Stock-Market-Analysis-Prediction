"""
Visualization Module for Stock Market Analysis
Creates interactive charts using Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List


class StockVisualizer:
    """Class for creating stock market visualizations"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data
        
        Args:
            data: DataFrame with stock data and indicators
        """
        self.data = data.copy()
    
    def plot_closing_price(self, title: str = "Closing Price Trend") -> go.Figure:
        """
        Create closing price trend chart
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2E86AB', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_moving_averages(self, title: str = "Moving Averages") -> go.Figure:
        """
        Create moving averages chart
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add closing price
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2E86AB', width=1),
            opacity=0.7
        ))
        
        # Add MA50 if available
        if 'MA50' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['Date'],
                y=self.data['MA50'],
                mode='lines',
                name='MA50',
                line=dict(color='#A23B72', width=2)
            ))
        
        # Add MA200 if available
        if 'MA200' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['Date'],
                y=self.data['MA200'],
                mode='lines',
                name='MA200',
                line=dict(color='#F18F01', width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_rsi(self, title: str = "RSI Indicator") -> go.Figure:
        """
        Create RSI chart with overbought/oversold levels
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        if 'RSI' not in self.data.columns:
            fig.update_layout(
                title="RSI data not available",
                height=300
            )
            return fig
        
        # Add RSI line
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='#C73E1D', width=2)
        ))
        
        # Add overbought level (70)
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="Overbought (70)")
        
        # Add oversold level (30)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                      annotation_text="Oversold (30)")
        
        # Add middle level (50)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                      annotation_text="Neutral (50)")
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_macd(self, title: str = "MACD Indicator") -> go.Figure:
        """
        Create MACD chart with signal line and histogram
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('MACD Line & Signal', 'MACD Histogram'),
            row_heights=[0.7, 0.3]
        )
        
        if 'MACD' not in self.data.columns:
            fig.update_layout(
                title="MACD data not available",
                height=400
            )
            return fig
        
        # Add MACD line
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='#2E86AB', width=2)
        ), row=1, col=1)
        
        # Add Signal line
        if 'Signal' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['Date'],
                y=self.data['Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#F18F01', width=2)
            ), row=1, col=1)
        
        # Add Histogram
        if 'MACD_Histogram' in self.data.columns:
            colors = ['green' if x >= 0 else 'red' for x in self.data['MACD_Histogram']]
            fig.add_trace(go.Bar(
                x=self.data['Date'],
                y=self.data['MACD_Histogram'],
                name='Histogram',
                marker_color=colors
            ), row=2, col=1)
        
        fig.update_layout(
            title=title,
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_bollinger_bands(self, title: str = "Bollinger Bands") -> go.Figure:
        """
        Create Bollinger Bands chart
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        if 'BB_Upper' not in self.data.columns:
            fig.update_layout(
                title="Bollinger Bands data not available",
                height=300
            )
            return fig
        
        # Add closing price
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2E86AB', width=1)
        ))
        
        # Add upper band
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['BB_Upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='gray', width=1),
            fill=None
        ))
        
        # Add lower band
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['BB_Lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.2)'
        ))
        
        # Add middle band
        if 'BB_Middle' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['Date'],
                y=self.data['BB_Middle'],
                mode='lines',
                name='Middle Band (MA20)',
                line=dict(color='red', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_volume(self, title: str = "Volume Analysis") -> go.Figure:
        """
        Create volume chart with moving average
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Add volume bars
        colors = ['green' if self.data['Close'].iloc[i] >= self.data['Open'].iloc[i] 
                  else 'red' for i in range(len(self.data))]
        
        fig.add_trace(go.Bar(
            x=self.data['Date'],
            y=self.data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ))
        
        # Add volume MA if available
        if 'Volume_MA' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data['Date'],
                y=self.data['Volume_MA'],
                mode='lines',
                name='Volume MA',
                line=dict(color='#A23B72', width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Volume',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_obv(self, title: str = "On Balance Volume (OBV)") -> go.Figure:
        """
        Create OBV chart
        
        Args:
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        if 'OBV' not in self.data.columns:
            fig.update_layout(
                title="OBV data not available",
                height=300
            )
            return fig
        
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['OBV'],
            mode='lines',
            name='OBV',
            line=dict(color='#6B4C9A', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='OBV',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_candlestick(self, title: str = "Candlestick Chart", show_ma: bool = False) -> go.Figure:
        """
        Create candlestick chart with optional moving averages
        
        Args:
            title: Chart title
            show_ma: Whether to show moving averages overlay
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Candlestick(
            x=self.data['Date'],
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Add moving averages if requested and available
        if show_ma:
            if 'MA50' in self.data.columns:
                fig.add_trace(go.Scatter(
                    x=self.data['Date'],
                    y=self.data['MA50'],
                    mode='lines',
                    name='MA50',
                    line=dict(color='#2196F3', width=1.5)
                ))
            
            if 'MA200' in self.data.columns:
                fig.add_trace(go.Scatter(
                    x=self.data['Date'],
                    y=self.data['MA200'],
                    mode='lines',
                    name='MA200',
                    line=dict(color='#FF9800', width=1.5)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white',
            height=550,
            xaxis_rangeslider_visible=False,
            dragmode='zoom'
        )
        
        return fig
    
    def plot_prediction_results(self, actual: pd.Series, predicted: pd.Series, 
                                dates: pd.Series = None, title: str = "Prediction Results") -> go.Figure:
        """
        Create prediction comparison chart with dates
        
        Args:
            actual: Actual values
            predicted: Predicted values
            dates: Date series for x-axis (optional)
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Use dates if provided, otherwise use index
        x_axis = dates if dates is not None else range(len(actual))
        
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=actual,
            mode='lines',
            name='Actual Price',
            line=dict(color='#2E86AB', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=predicted,
            mode='lines',
            name='Predicted Price',
            line=dict(color='#C73E1D', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date' if dates is not None else 'Time',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white',
            height=450,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_dashboard_charts(self, selected_indicators: List[str], show_candlestick: bool = False) -> dict:
        """
        Create all selected charts for dashboard
        
        Args:
            selected_indicators: List of selected indicator names
            show_candlestick: Whether to show candlestick chart
            
        Returns:
            Dictionary of chart figures
        """
        charts = {}
        
        # Always show closing price
        charts['Closing Price'] = self.plot_closing_price()
        
        # Candlestick chart if requested
        if show_candlestick:
            charts['Candlestick'] = self.plot_candlestick(show_ma=True)
        
        # Moving averages
        if 'Moving Average' in selected_indicators:
            charts['Moving Averages'] = self.plot_moving_averages()
        
        # RSI
        if 'RSI' in selected_indicators:
            charts['RSI'] = self.plot_rsi()
        
        # MACD
        if 'MACD' in selected_indicators:
            charts['MACD'] = self.plot_macd()
        
        # Bollinger Bands
        if 'Bollinger Bands' in selected_indicators:
            charts['Bollinger Bands'] = self.plot_bollinger_bands()
        
        # Volume
        if 'Volume Analysis' in selected_indicators:
            charts['Volume'] = self.plot_volume()
        
        # OBV
        if 'OBV' in selected_indicators:
            charts['OBV'] = self.plot_obv()
        
        return charts
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame, 
                                title: str = "Feature Importance") -> go.Figure:
        """
        Create horizontal bar chart for feature importance
        
        Args:
            feature_importance_df: DataFrame with Feature and Importance columns
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if feature_importance_df is None or feature_importance_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Feature importance data not available",
                height=300
            )
            return fig
        
        # Sort by importance (descending)
        df_sorted = feature_importance_df.sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=df_sorted['Importance'],
            y=df_sorted['Feature'],
            orientation='h',
            marker=dict(
                color=df_sorted['Importance'],
                colorscale='Viridis',
                colorbar=dict(title="Importance")
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_white',
            height=400,
            margin=dict(l=150, r=20, t=50, b=50)
        )
        
        return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                              title: str = "Model Performance Comparison") -> go.Figure:
        """
        Create bar chart for model comparison
        
        Args:
            comparison_df: DataFrame with Model, MAE, RMSE, R2 Score columns
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        if comparison_df is None or comparison_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Model comparison data not available",
                height=300
            )
            return fig
        
        fig = go.Figure()
        
        # Add MAE bars
        fig.add_trace(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df['MAE'],
            name='MAE',
            marker_color='#2E86AB'
        ))
        
        # Add RMSE bars
        fig.add_trace(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df['RMSE'],
            name='RMSE',
            marker_color='#A23B72'
        ))
        
        # Add R2 Score bars (normalized to same scale)
        fig.add_trace(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df['R2 Score'] * max(comparison_df['MAE'].max(), comparison_df['RMSE'].max()),
            name='R2 Score (scaled)',
            marker_color='#F18F01'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Model',
            yaxis_title='Score',
            template='plotly_white',
            height=400,
            barmode='group'
        )
        
        return fig
