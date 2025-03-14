# ===============================
# 📌 1. Import Library
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


# ===============================
# 📌 2. Load Data dari Google Cloud Monitoring
# ===============================
# Simulasi Data Metric dari Google Cloud Monitoring (Data Per Jam)
data = {
    'timestamp': pd.date_range(start='2024-03-01', periods=100, freq='H'),  # Data per jam
    'cpu_usage': np.random.uniform(0.2, 0.95, 100),  # Simulasi CPU usage antara 20%-95%
    'memory_usage': np.random.uniform(0.1, 0.9, 100),  # Simulasi Memory usage antara 10%-90%
}

df = pd.DataFrame(data)

# Simpan data ke CSV (opsional)
df.to_csv("cloud_metrics_forecasting.csv", index=False)

print("✅ Data Cloud Metrics berhasil di-generate!")

# ===============================
# 📌 3. Prediksi CPU Usage dengan ARIMA
# ===============================
# Konversi timestamp ke index
df.set_index('timestamp', inplace=True)

# Buat Model ARIMA (p,d,q)
model_arima = ARIMA(df['cpu_usage'], order=(5,1,0))  # (p=5, d=1, q=0)
model_arima_fit = model_arima.fit()

# Prediksi 10 jam ke depan
forecast_arima = model_arima_fit.forecast(steps=10)

# Plot hasil prediksi ARIMA
plt.figure(figsize=(10,5))
plt.plot(df.index, df['cpu_usage'], label="Actual CPU Usage")
plt.plot(pd.date_range(df.index[-1], periods=10, freq='H'), forecast_arima, label="Predicted CPU (ARIMA)", linestyle="dashed", color="red")
plt.title("CPU Usage Forecasting with ARIMA")
plt.xlabel("Timestamp")
plt.ylabel("CPU Usage")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()

# ===============================
# 📌 4. Prediksi CPU Usage dengan FB Prophet
# ===============================
# Format data untuk Prophet (harus 'ds' untuk timestamp & 'y' untuk target)
df_prophet = df.reset_index()[['timestamp', 'cpu_usage']].rename(columns={'timestamp': 'ds', 'cpu_usage': 'y'})

# Buat model Prophet & training
model_prophet = Prophet()
model_prophet.fit(df_prophet)

# Prediksi 10 jam ke depan
future = model_prophet.make_future_dataframe(periods=10, freq='H')
forecast_prophet = model_prophet.predict(future)

# Plot hasil prediksi Prophet
plt.figure(figsize=(10,5))
plt.plot(df.index, df['cpu_usage'], label="Actual CPU Usage")
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label="Predicted CPU (Prophet)", linestyle="dashed", color="green")
plt.title("CPU Usage Forecasting with FB Prophet")
plt.xlabel("Timestamp")
plt.ylabel("CPU Usage")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()

# ===============================
# 📌 5. Prediksi Instance Scaling Berdasarkan Forecasting
# ===============================
def autoscale_forecast(predicted_cpu_usage, target_utilization=0.7, current_instances=5):
    """Menentukan jumlah instance yang direkomendasikan berdasarkan prediksi CPU"""
    recommended_instances = np.ceil((predicted_cpu_usage * current_instances) / target_utilization)
    return int(recommended_instances)

# Ambil hasil prediksi 10 jam ke depan
predicted_cpu = forecast_prophet['yhat'].iloc[-10:].mean()  # Rata-rata 10 jam ke depan
recommended_instances = autoscale_forecast(predicted_cpu, current_instances=5)

print(f"🔮 Predicted CPU Usage (Next 10h Avg): {predicted_cpu:.2f}")
print(f"⚙️ Recommended Instances: {recommended_instances}")

# ===============================
# 📌 6. Simulasi Scaling Decision Berdasarkan Prediksi
# ===============================
def simulate_autoscaling(predicted_cpu):
    """Menentukan apakah perlu scaling berdasarkan prediksi"""
    if predicted_cpu > 0.8:
        return "Scale Up"
    elif predicted_cpu < 0.3:
        return "Scale Down"
    else:
        return "Optimal"

scaling_decision = simulate_autoscaling(predicted_cpu)
print(f"📢 Scaling Decision Based on Forecast: {scaling_decision}")
