# OFFSIDE — UCL Statistical Analyzer (Web App)

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python app.py

# 3. Open in browser
http://localhost:5000
```

## Usage
1. Click **📂 Upload Excel** in the top header
2. Select your `ucl_round_of_16.xlsx` file
3. Navigate using the bottom tab bar

## Pages
| Tab | Contents |
|-----|----------|
| 🏠 HOME | KPI cards, Featured Prediction, Top 10 chart, Club summary table |
| 🎯 PREDICTOR | Team vs Team selector, Win % donut, Comparison chart & table |
| 📊 ANALYTICS | Descriptive stats, CI, Histograms, Box plots, Normality test, Poisson fit, Correlation heatmap |
| 📈 REGRESSION | Simple regression (pick X/Y), prediction tool, Multiple regression, Residuals plot |
| 🔍 PLAYERS | Search/filter/sort all 699 players |
