"""
Stock Market Analysis & Prediction Dashboard
Professional Financial Analytics Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from preprocessing import DataPreprocessor
from indicators import TechnicalIndicators
from visualization import StockVisualizer
from prediction import StockPredictor
from database import PredictionDatabase

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = DataPreprocessor()
if 'indicators' not in st.session_state:
    st.session_state.indicators = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'stock_name' not in st.session_state:
    st.session_state.stock_name = None
if 'db' not in st.session_state:
    st.session_state.db = PredictionDatabase()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("### 📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=['csv'],
    help="Upload CSV with Date, Open, High, Low, Close, Volume"
)

if uploaded_file:
    with st.sidebar:
        with st.spinner("Loading..."):
            success, message = st.session_state.preprocessor.load_csv(uploaded_file)
            
            if success:
                success, message = st.session_state.preprocessor.preprocess_data()
                
                if success:
                    st.session_state.data = st.session_state.preprocessor.get_data()
                    st.success("Dataset loaded")
                    
                    info = st.session_state.preprocessor.get_data_info()
                    
                    if len(info['available_stocks']) > 1:
                        st.sidebar.markdown("### 📈 Stock Selection")
                        selected_stock = st.sidebar.selectbox(
                            "Select Stock",
                            info['available_stocks']
                        )
                        
                        if selected_stock != st.session_state.stock_name:
                            success, message = st.session_state.preprocessor.select_stock(selected_stock)
                            if success:
                                st.session_state.data = st.session_state.preprocessor.get_data()
                                st.session_state.stock_name = selected_stock
                    else:
                        st.session_state.stock_name = info['available_stocks'][0]
                else:
                    st.error(message)
            else:
                st.error(message)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Technical Indicators")

indicator_options = [
    "Candlestick",
    "Moving Average",
    "RSI",
    "MACD",
    "Bollinger Bands",
    "Volume",
    "OBV"
]

selected_indicators = st.sidebar.multiselect(
    "Select Indicators",
    indicator_options,
    default=["Candlestick", "Moving Average", "RSI", "MACD", "Bollinger Bands"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Prediction")

model_options = ["Linear Regression", "Random Forest", "XGBoost"]
selected_model = st.sidebar.selectbox(
    "Model",
    model_options,
    index=1
)

run_prediction = st.sidebar.button("Run Prediction")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Analysis")

show_model_comparison = st.sidebar.checkbox("Model Comparison", value=True)
show_feature_importance = st.sidebar.checkbox("Feature Importance", value=True)

# =====================================================
# MAIN CONTENT
# =====================================================

if st.session_state.data is not None:
    # Calculate indicators
    st.session_state.indicators = TechnicalIndicators(st.session_state.data)
    st.session_state.indicators.calculate_all_indicators()
    data_with_indicators = st.session_state.indicators.data
    
    # Initialize visualizer and predictor
    st.session_state.visualizer = StockVisualizer(data_with_indicators)
    st.session_state.predictor = StockPredictor(data_with_indicators)
    
    # Get latest data
    latest_data = st.session_state.preprocessor.get_latest_data()
    
    # =====================================================
    # METRIC CARDS
    # =====================================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = latest_data['Close']
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{latest_data['daily_change_pct']:.2f}%"
        )
    
    with col2:
        daily_change = latest_data['daily_change']
        st.metric(
            "Daily Change",
            f"${daily_change:.2f}",
            delta_color="normal" if daily_change >= 0 else "inverse"
        )
    
    with col3:
        volume = latest_data['Volume']
        st.metric("Volume", f"{volume:,.0f}")
    
    with col4:
        if 'MA50' in data_with_indicators.columns and 'MA200' in data_with_indicators.columns:
            latest_ma50 = data_with_indicators['MA50'].iloc[-1]
            latest_ma200 = data_with_indicators['MA200'].iloc[-1]
            trend = "Uptrend 📈" if latest_ma50 > latest_ma200 else "Downtrend 📉"
        else:
            trend = "N/A"
        st.metric("Trend", trend)
    
    st.markdown("---")
    
    # =====================================================
    # PRICE ANALYSIS SECTION
    # =====================================================
    
    st.header("Price Analysis")
    
    charts = st.session_state.visualizer.create_dashboard_charts(
        selected_indicators, 
        show_candlestick=("Candlestick" in selected_indicators)
    )
    
    if 'Candlestick' in charts:
        st.plotly_chart(charts['Candlestick'], use_container_width=True)
    
    if 'Closing Price' in charts:
        st.plotly_chart(charts['Closing Price'], use_container_width=True)
    
    if 'Moving Averages' in charts:
        st.plotly_chart(charts['Moving Averages'], use_container_width=True)
    
    st.markdown("---")
    
    # =====================================================
    # TECHNICAL INDICATORS SECTION
    # =====================================================
    
    st.header("Technical Indicators")
    
    signals = st.session_state.indicators.get_all_signals()
    
    indicator_tabs = st.tabs(["RSI", "MACD", "Bollinger Bands", "Volume", "OBV"])
    
    with indicator_tabs[0]:
        if 'RSI' in charts:
            st.plotly_chart(charts['RSI'], use_container_width=True)
        rsi_signal = signals['RSI']
        if rsi_signal == 'OVERBOUGHT':
            st.warning(f"RSI: {rsi_signal}")
        elif rsi_signal == 'OVERSOLD':
            st.success(f"RSI: {rsi_signal}")
        else:
            st.info(f"RSI: {rsi_signal}")
    
    with indicator_tabs[1]:
        if 'MACD' in charts:
            st.plotly_chart(charts['MACD'], use_container_width=True)
        macd_signal = signals['MACD']
        if macd_signal == 'BULLISH':
            st.success(f"MACD: {macd_signal}")
        elif macd_signal == 'BEARISH':
            st.warning(f"MACD: {macd_signal}")
        else:
            st.info(f"MACD: {macd_signal}")
    
    with indicator_tabs[2]:
        if 'Bollinger Bands' in charts:
            st.plotly_chart(charts['Bollinger Bands'], use_container_width=True)
        bb_position = signals['Bollinger']
        if bb_position == 'NEAR_UPPER':
            st.warning(f"Bollinger: {bb_position}")
        elif bb_position == 'NEAR_LOWER':
            st.success(f"Bollinger: {bb_position}")
        else:
            st.info(f"Bollinger: {bb_position}")
    
    with indicator_tabs[3]:
        if 'Volume' in charts:
            st.plotly_chart(charts['Volume'], use_container_width=True)
    
    with indicator_tabs[4]:
        if 'OBV' in charts:
            st.plotly_chart(charts['OBV'], use_container_width=True)
    
    st.markdown("---")
    
    # =====================================================
    # PREDICTION SECTION
    # =====================================================
    
    st.header("Prediction")
    
    model_type_map = {
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost"
    }
    
    if run_prediction or True:
        with st.spinner("Running prediction..."):
            prediction_results = st.session_state.predictor.run_prediction(
                model_type_map[selected_model]
            )
        
        if prediction_results['success']:
            next_day_price = prediction_results['next_day_price']
            current_price = latest_data['Close']
            price_change = next_day_price - current_price
            price_change_pct = (price_change / current_price) * 100
            metrics = prediction_results['metrics']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Selected Model", selected_model)
            
            with col2:
                st.metric(
                    "Predicted Price",
                    f"${next_day_price:.2f}",
                    f"{price_change_pct:.2f}%"
                )
            
            with col3:
                st.metric("R² Score", f"{metrics['R2']:.4f}")
            
            st.markdown("---")
            
            # Trading Signal
            trading_signal = st.session_state.predictor.get_trading_signal(
                current_price=current_price,
                predicted_price=next_day_price,
                rsi_signal=signals['RSI'],
                macd_signal=signals['MACD'],
                bollinger_position=signals['Bollinger']
            )
            
            if trading_signal == 'BUY':
                st.success(f"🟢 {trading_signal}")
            elif trading_signal == 'SELL':
                st.error(f"🔴 {trading_signal}")
            else:
                st.warning(f"🟡 {trading_signal}")
            
            st.markdown("---")
            
            # Prediction Chart
            pred_chart = st.session_state.visualizer.plot_prediction_results(
                actual=pd.Series(prediction_results['actual']),
                predicted=pd.Series(prediction_results['predictions']),
                dates=pd.Series(prediction_results['test_dates'])
            )
            st.plotly_chart(pred_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Download Analysis Report
            st.subheader("Analysis Report")
            
            # Generate report data
            report_data = pd.DataFrame({
                "Date & Time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Selected Stock": [st.session_state.stock_name or "Unknown"],
                "Selected Machine Learning Model": [selected_model],
                "Current Price": [f"${current_price:.2f}"],
                "Predicted Price": [f"${next_day_price:.2f}"],
                "Trading Signal": [trading_signal],
                "MAE": [f"{metrics['MAE']:.4f}"],
                "RMSE": [f"{metrics['RMSE']:.4f}"],
                "R² Score": [f"{metrics['R2']:.4f}"]
            })
            
            csv = report_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="⬇ Download Analysis Report",
                data=csv,
                file_name="stock_analysis_report.csv",
                mime="text/csv"
            )
            
            # Insert prediction into database
            st.session_state.db.insert_prediction(
                stock_name=st.session_state.stock_name or "Unknown",
                model_name=selected_model,
                current_price=current_price,
                predicted_price=next_day_price,
                trading_signal=trading_signal,
                mae=metrics['MAE'],
                rmse=metrics['RMSE'],
                r2_score=metrics['R2']
            )
            
        else:
            st.error(f"Prediction failed: {prediction_results['error']}")
    
    st.markdown("---")
    
    # =====================================================
    # MODEL COMPARISON SECTION
    # =====================================================
    
    if show_model_comparison:
        st.header("Model Comparison")
        
        with st.spinner("Comparing models..."):
            model_comparison = st.session_state.predictor.compare_models()
        
        if model_comparison is not None:
            st.dataframe(model_comparison, use_container_width=True)
            
            best_model = model_comparison.iloc[0]
            st.success(f"🏆 Best Model: {best_model['Model']} (R²: {best_model['R2 Score']:.4f})")
        else:
            st.warning("Model comparison not available")
        
        st.markdown("---")
    
    # =====================================================
    # FEATURE IMPORTANCE SECTION
    # =====================================================
    
    if show_feature_importance:
        st.header("Feature Importance")
        
        with st.spinner("Calculating feature importance..."):
            st.session_state.predictor.run_prediction('random_forest')
            feature_importance = st.session_state.predictor.get_feature_importance()
        
        if feature_importance is not None:
            fig = st.session_state.visualizer.plot_feature_importance(feature_importance)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available")
    
    st.markdown("---")
    
    # =====================================================
    # PREDICTION HISTORY SECTION
    # =====================================================
    
    st.header("📜 Prediction History")
    
    # Get latest predictions
    history_df = st.session_state.db.get_latest_predictions(limit=10)
    
    if history_df is not None and not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑 Clear History"):
                if st.session_state.db.clear_history():
                    st.success("History cleared successfully")
                    st.rerun()
                else:
                    st.error("Failed to clear history")
        
        with col2:
            all_history = st.session_state.db.get_all_predictions()
            if all_history is not None and not all_history.empty:
                csv_history = all_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download History CSV",
                    data=csv_history,
                    file_name="prediction_history.csv",
                    mime="text/csv"
                )
    else:
        st.info("No prediction history available. Run a prediction to see history here.")

else:
    # =====================================================
    # LANDING PAGE
    # =====================================================
    
    st.markdown("## 📈 Stock Market Analysis & Prediction Dashboard")
    st.markdown("Analyze historical stock market data using Technical Analysis and Machine Learning.")
    
    st.markdown("---")
    st.info("📁 Upload Dataset from Sidebar")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Market Analysis")
        st.markdown("Analyze historical stock trends using interactive charts.")
    
    with col2:
        st.markdown("### 📈 Technical Indicators")
        st.markdown("RSI, MACD, Bollinger Bands, Moving Averages and OBV.")
    
    with col3:
        st.markdown("### 🤖 Machine Learning")
        st.markdown("Predict stock prices using Linear Regression, Random Forest and XGBoost.")
    
    st.markdown("---")
    st.caption("Educational Project | Not Financial Advice")
