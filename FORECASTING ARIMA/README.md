# Champagne Sales Forecasting using SARIMA

## Overview

This project focuses on forecasting **Perrin Frères Monthly Champagne Sales** using time series analysis techniques. The dataset consists of monthly sales data from 1964 to 1972. The objective is to analyze trends, seasonality, and variations in sales to predict future sales and support business decision-making.

## Business Problem

The sales of champagne fluctuate due to seasonal demand, economic factors, and consumer behavior. Understanding sales patterns is crucial for optimizing inventory, improving marketing strategies, and ensuring a steady supply chain. Accurate forecasting can help businesses minimize losses due to overstocking or understocking.

## Key Insights

- **Data Preprocessing**:
  - Renamed columns for clarity.
  - Removed unnecessary rows.
  - Converted date columns to proper datetime format for time series analysis.
- **Time Series Decomposition**:
  - Decomposed sales data into **trend, seasonality, and residuals**.
    ![image_anti](https://github.com/Pdeep666/ML_MODEL_DEPLOYMENT/blob/65ab03c7f9242188482f992d90309acf8863d242/FORECASTING%20ARIMA/IMG1.png)
- **Statistical Analysis**:
  - Calculated rolling mean and rolling standard deviation to understand sales variability.
    ![image_anti](https://github.com/Pdeep666/ML_MODEL_DEPLOYMENT/blob/4631bed644be5db6f718eb8355f36e76e59cee68/FORECASTING%20ARIMA/IMG2.png)
  - Performed stationarity tests for model selection.
- **Forecasting Methods**:
  - Applied **ARIMA (AutoRegressive Integrated Moving Average)** for sales prediction.
  - Applied **SARIMA (Seasonal ARIMA) (1,1,1)(1,1,1,12)** to incorporate seasonality and improve accuracy.
  - Used the model to generate **future sales predictions** to help in business planning.

## Tools Used

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **Statsmodels** (for time series decomposition and ARIMA modeling)
- **Google Colab** (for execution and analysis)

## Recommendations

1. **Inventory Management**: Use seasonal trends to ensure stock availability during peak sales periods.
2. **Marketing Strategies**: Increase advertising and promotions before high-demand seasons.
3. **Dynamic Pricing**: Implement price adjustments based on forecasted demand trends.
4. **Supply Chain Optimization**: Plan procurement and distribution to avoid shortages.
5. **Continuous Model Monitoring**: Regularly update forecasting models to maintain accuracy.

## Conclusion

By analyzing and forecasting champagne sales, businesses can make informed decisions to optimize supply chain management, marketing strategies, and financial planning. The use of **time series models** ensures better demand prediction, reducing operational inefficiencies.

## License

This project is intended for educational and analytical purposes. Unauthorized distribution or commercial use of this content is prohibited.

