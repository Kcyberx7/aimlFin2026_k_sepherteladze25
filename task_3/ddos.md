# Task 3: Web Server Log Analysis for DDoS Detection

## 1. Introduction
This analysis identifies DDoS attack intervals within a web server log using **Linear Regression**. By establishing a baseline of normal traffic patterns, we can statistically flag anomalies that indicate an attack.

* **Log File:** [server.log](./server.log)

## 2. Methodology
To solve this task, I performed the following technical steps:
1.  **Parsing:** Used Regular Expressions to extract timestamps from the raw log file.
2.  **Aggregation:** Resampled the data into 1-second intervals to calculate request density.
3.  **Regression Analysis:** Applied a Linear Regression model to find the predicted "normal" traffic trend.
4.  **Anomaly Detection:** Any activity exceeding the regression trend by a threshold of 20+ requests per second was flagged as a DDoS event.

## 3. DDoS Attack Interval
Based on the regression analysis of the provided logs, the DDoS attack activity was detected during the following timeframe:
* **Start Time:** 2024-03-22 18:05:13
* **End Time:** 2024-03-22 18:52:29

## 4. Source Code Fragments
The core logic for the regression analysis and detection is as follows:
```python
# Create regression model
model = LinearRegression()
model.fit(X, y)
traffic['predicted_trend'] = model.predict(X)

# Detect spikes significantly above the trend line
traffic['is_attack'] = traffic['requests'] > (traffic['predicted_trend'] + 20)