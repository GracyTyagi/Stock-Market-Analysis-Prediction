# 📈 Stock Market Analysis & Prediction Tool

A professional Stock Market Analytics Dashboard built with **Python, Streamlit, Plotly, Scikit-learn, XGBoost, and SQLite**.

The application allows users to upload stock datasets, visualize market trends, calculate technical indicators, compare machine learning models, generate trading signals, and maintain prediction history — all in an interactive dashboard.

> ⚠️ This project is designed for **educational and portfolio purposes only** and does not provide financial advice.

---

## 🚀 Features

### 📂 Dataset Management
- Upload CSV stock datasets
- Automatic data validation and preprocessing
- Missing value handling
- Date conversion & sorting
- Support for single-stock and multi-stock datasets

### 📊 Technical Indicators
- Moving Average (MA50 & MA200)
- RSI (Relative Strength Index)
- MACD
- Bollinger Bands
- Volume Moving Average
- On Balance Volume (OBV)

### 📈 Interactive Visualizations
- Candlestick Chart
- Closing Price Trend
- Moving Average Chart
- RSI Chart
- MACD Chart
- Bollinger Bands
- Volume Analysis
- OBV Analysis
- Actual vs Predicted Price
- Feature Importance

### 🤖 Machine Learning Models
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### 📉 Model Evaluation
- MAE
- RMSE
- R² Score
- MAPE
- Model Comparison Dashboard

### 🎯 Trading Signals
Automatic BUY / SELL / HOLD recommendations based on:
- RSI
- MACD
- Bollinger Bands

### 📜 Prediction History
- SQLite Database
- Prediction History
- Download History CSV
- Download Analysis Report

---

# 📁 Project Structure

```
Stock-Market-Analysis-Tool/
│
├── app.py
├── preprocessing.py
├── indicators.py
├── visualization.py
├── prediction.py
├── database.py
├── requirements.txt
├── README.md
├── run.bat
└── launch.vbs
```

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- XGBoost
- SQLite

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Stock-Market-Analysis-Tool.git
```

## 2. Open Project

```bash
cd Stock-Market-Analysis-Tool
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Application

### Windows

Double-click

```
launch.vbs
```

or

```bash
python -m streamlit run app.py
```

The dashboard will automatically open in your browser.

---

# 📂 Dataset Format

The CSV file must contain:

| Column | Required |
|----------|----------|
| Date | ✅ |
| Open | ✅ |
| High | ✅ |
| Low | ✅ |
| Close | ✅ |
| Volume | ✅ |

Optional:

- Name
- Symbol

Example:

```csv
Date,Open,High,Low,Close,Volume
2024-01-15,150.25,155.50,149.75,154.00,1000000
```

---

# 📖 Workflow

1. Upload CSV dataset
2. Select stock
3. Choose technical indicators
4. Select prediction model
5. Run prediction
6. View charts & trading signals
7. Compare ML models
8. Download analysis report
9. Check prediction history

---

# 📊 Machine Learning Models

| Model | Purpose |
|--------|----------|
| Linear Regression | Baseline Prediction |
| Random Forest | Non-linear Prediction |
| XGBoost | Advanced Boosting Model |

---

# 📈 Technical Indicators

| Indicator | Purpose |
|------------|----------|
| MA50 / MA200 | Trend Analysis |
| RSI | Overbought / Oversold |
| MACD | Momentum |
| Bollinger Bands | Volatility |
| Volume MA | Volume Trend |
| OBV | Price Confirmation |

---

# 📜 Prediction Report

Each prediction includes:

- Current Price
- Predicted Price
- Trading Signal
- MAE
- RMSE
- R² Score
- Date & Time
- Selected Model

Reports can be downloaded as CSV.

---

# 🗄 Prediction History

Every prediction is automatically stored in SQLite.

Stored information:

- Prediction Date
- Selected Model
- Current Price
- Predicted Price
- Trading Signal
- Performance Metrics

---

# ⚠️ Disclaimer

This project is intended for **educational, research, and portfolio purposes only.**

The generated predictions and trading signals should **not** be considered financial advice.

Always perform your own research before making investment decisions.

---

# 👩‍💻 Author

**Gracy Tyagi**

MCA (Data Science)

### Skills

- Python
- Machine Learning
- Data Analytics
- SQL
- Streamlit
- Scikit-learn

GitHub:
https://github.com/YOUR_USERNAME

LinkedIn:
https://linkedin.com/in/YOUR_PROFILE

---

# 📄 License

This project is licensed under the MIT License.

---

# 🙏 Acknowledgements

- Streamlit
- Plotly
- Scikit-learn
- XGBoost
- Kaggle

---

⭐ If you found this project useful, consider giving it a star on GitHub.