# ===============================
# 📌 1. Import Library
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ===============================
# 📌 2. Load Data dari Google Cloud Monitoring
# ===============================
# Simulasi Data Metric dari Google Cloud Monitoring
data = {
    'timestamp': pd.date_range(start='2024-03-01', periods=30, freq='H'),  # Data per jam
    'cpu_usage': np.random.uniform(0.2, 0.95, 30),  # Simulasi CPU usage antara 20%-95%
    'memory_usage': np.random.uniform(0.1, 0.9, 30),  # Simulasi Memory usage antara 10%-90%
    'instances_running': np.random.randint(3, 10, 30)  # Jumlah instance yang aktif
}

df = pd.DataFrame(data)

# Simpan data ke CSV (opsional)
df.to_csv("cloud_metrics_sintetis.csv", index=False)

print("✅ Data Cloud Metrics berhasil di-generate!")

# ===============================
# 📌 3. Kalkulasi Sintetis untuk Cloud Sizing Automation
# ===============================

## 3.1 Aturan Threshold-Based Scaling
def synthetic_scaling_decision(cpu_usage, mem_usage):
    """Menentukan apakah instance perlu scale up atau down berdasarkan threshold"""
    if cpu_usage > 0.8 or mem_usage > 0.8:
        return "Scale Up"
    elif cpu_usage < 0.3 and mem_usage < 0.3:
        return "Scale Down"
    else:
        return "Optimal"

df['scaling_decision'] = df.apply(lambda row: synthetic_scaling_decision(row['cpu_usage'], row['memory_usage']), axis=1)

## 3.2 Moving Average untuk Trend Analysis
df['cpu_avg'] = df['cpu_usage'].rolling(window=5, min_periods=1).mean()
df['mem_avg'] = df['memory_usage'].rolling(window=5, min_periods=1).mean()

## 3.3 Capacity Planning Berdasarkan Peak Usage
def calculate_instance_sizing(peak_cpu_usage, total_instances, target_cpu_utilization=0.7):
    """Menghitung jumlah instance yang direkomendasikan berdasarkan peak usage"""
    recommended_instances = np.ceil((peak_cpu_usage * total_instances) / target_cpu_utilization)
    return int(recommended_instances)

# Gunakan peak CPU dari histori sebagai dasar rekomendasi
peak_cpu_usage = df['cpu_usage'].max()
peak_instances = df['instances_running'].max()
recommended_instances = calculate_instance_sizing(peak_cpu_usage, peak_instances)

print(f"🔥 Peak CPU Usage: {peak_cpu_usage:.2f}")
print(f"🔧 Current Max Instances: {peak_instances}")
print(f"⚡ Recommended Instances: {recommended_instances}")

# ===============================
# 📌 4. Visualisasi Hasil Analisis
# ===============================
plt.figure(figsize=(10,5))
plt.plot(df['timestamp'], df['cpu_usage'], label="CPU Usage", color='b')
plt.plot(df['timestamp'], df['cpu_avg'], label="CPU Moving Avg", color='r', linestyle='dashed')
plt.axhline(y=0.8, color='g', linestyle='--', label="Threshold Scale Up")
plt.axhline(y=0.3, color='orange', linestyle='--', label="Threshold Scale Down")
plt.title("Cloud Resource Usage & Scaling Analysis")
plt.xlabel("Timestamp")
plt.ylabel("CPU Usage")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# ===============================
# 📌 5. Simulasi Pengambilan Keputusan Scaling untuk Instance
# ===============================
def simulate_scaling_recommendation(df):
    """Menampilkan keputusan scaling per timestamp"""
    for index, row in df.iterrows():
        print(f"🕒 {row['timestamp']} | CPU: {row['cpu_usage']:.2f} | Mem: {row['memory_usage']:.2f} | Scaling Decision: {row['scaling_decision']}")

simulate_scaling_recommendation(df)
