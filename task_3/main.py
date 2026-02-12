import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import re

# 1. Load the log file
log_data = []
# This matches your specific log date format: [2024-03-22 18:00:44]
date_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'

with open("server.log", "r") as f:
    for line in f:
        match = re.search(date_pattern, line)
        if match:
            log_data.append(match.group(1))

# 2. Process into a DataFrame
df = pd.DataFrame(log_data, columns=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Count requests per second
df.set_index('timestamp', inplace=True)
traffic = df.resample('1s').size().reset_index(name='requests')
traffic['seconds_passed'] = np.arange(len(traffic))

# 4. Regression Analysis
X = traffic[['seconds_passed']]
y = traffic['requests']
model = LinearRegression()
model.fit(X, y)
traffic['predicted_trend'] = model.predict(X)

# Flagging DDoS: Anything significantly above the trend line
# Adjust the '+ 10' if needed depending on how big the spikes are
traffic['is_attack'] = traffic['requests'] > (traffic['predicted_trend'] + 10)

# 5. Output results
attack_windows = traffic[traffic['is_attack']]
if not attack_windows.empty:
    print(f"DDoS detected between: {attack_windows['timestamp'].min()} and {attack_windows['timestamp'].max()}")
else:
    print("No significant DDoS spikes detected.")

# 6. Create Visualization
plt.figure(figsize=(10, 5))
plt.plot(traffic['timestamp'], traffic['requests'], label='Actual Traffic', color='blue', alpha=0.5)
plt.plot(traffic['timestamp'], traffic['predicted_trend'], label='Regression Trend', color='green', linestyle='--')
plt.scatter(attack_windows['timestamp'], attack_windows['requests'], color='red', s=5, label='Attack Points')
plt.title('DDoS Detection: Request Spikes vs Regression Trend')
plt.legend()
plt.savefig('ddos_plot.png')
plt.show()