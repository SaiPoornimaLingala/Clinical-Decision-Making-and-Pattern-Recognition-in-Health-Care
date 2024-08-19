import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import streamlit as st

# Step 1: Generate Synthetic Heart Rate Data
np.random.seed(42)
time = pd.date_range(start="2023-01-01", periods=100, freq='h')  # Corrected frequency 'h'
heart_rate = np.random.normal(loc=70, scale=5, size=len(time))

# Introduce some anomalies
heart_rate[20:25] = heart_rate[20:25] + 15  # Sudden increase
heart_rate[50:55] = heart_rate[50:55] - 10  # Sudden drop

data = pd.DataFrame({"Time": time, "Heart Rate": heart_rate})

# Step 2: Apply Anomaly Detection using Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
data['Anomaly'] = model.fit_predict(data[['Heart Rate']])

# Mark anomalies in the data
anomalies = data[data['Anomaly'] == -1]

# Step 3: Create a Plot to Visualize the Data and Anomalies
plt.figure(figsize=(10, 6))
plt.plot(data['Time'], data['Heart Rate'], label='Heart Rate')
plt.scatter(anomalies['Time'], anomalies['Heart Rate'], color='red', label='Anomaly')
plt.title('Heart Rate Time-Series with Anomalies')
plt.xlabel('Time')
plt.ylabel('Heart Rate')
plt.legend()
plt.grid(True)

# Step 4: Build the Streamlit Web App
st.title("Heart Rate Anomaly Detection")

# Display the chart
st.pyplot(plt)

# Show the data in a table
st.write("### Heart Rate Data with Anomalies")
st.write(data)

# Highlight the anomalies in the table
st.write("### Anomalies Detected")
st.write(anomalies)

st.write("This is a basic anomaly detection prototype using synthetic heart rate data. Anomalies are detected using the Isolation Forest algorithm and are highlighted in the plot and table above.")
