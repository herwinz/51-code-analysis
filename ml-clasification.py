import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
import pytz
from google.cloud import monitoring_v3
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 📌 1. Konfigurasi Google Cloud Monitoring (Stackdriver) & Ambil Data
# ===============================
client = monitoring_v3.MetricServiceClient()
project_id = "sikopat"
project_name = f"projects/{project_id}"

# Tentukan rentang waktu pengambilan data (misal: 7 hari terakhir)
wib_timezone = pytz.timezone('Asia/Jakarta')
end_time_utc = datetime.now(timezone.utc)
end_time_wib = end_time_utc.astimezone(wib_timezone)
start_time_utc = end_time_utc - timedelta(days=7)
start_time_wib = start_time_utc.astimezone(wib_timezone)

def get_metric_data(metric_type):
    interval = monitoring_v3.TimeInterval(
        start_time=start_time_utc,
        end_time=end_time_utc,
    )
    results = client.list_time_series(
        name=project_name,
        filter=f'metric.type="{metric_type}"',
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )
    data_list = []
    for series in results:
        for point in series.points:
            timestamp_utc = point.interval.end_time
            timestamp_wib = timestamp_utc.astimezone(wib_timezone)
            instance = series.resource.labels.get("instance_id", "unknown_instance")
            value = point.value.double_value
            data_list.append([instance, timestamp_wib, value])
    return pd.DataFrame(data_list, columns=["instance_id", "timestamp", metric_type.split("/")[-1].replace("percent_used", "memory_usage").replace("utilization", "cpu_utilization")])

# Mengambil data CPU & Memory Usage
cpu_df = get_metric_data("agent.googleapis.com/cpu/utilization")
mem_df = get_metric_data("agent.googleapis.com/memory/percent_used")

cpu_df["timestamp"] = cpu_df["timestamp"].dt.round("s")
mem_df["timestamp"] = mem_df["timestamp"].dt.round("s")

df = pd.merge(cpu_df, mem_df, on=["instance_id", "timestamp"], how="inner")

if not df.empty:
    df.to_csv("gcp_vm_metrics.csv", index=False)
    print("✅ Data berhasil diambil dan disimpan!")
else:
    print("⚠️ No Data was found, CSV not created")

# ===============================
# 📌 2. Load Dataset dari CSV
# ===============================
file_path = "gcp_vm_metrics.csv"
df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ===============================
# 📌 3. Pisahkan Dataset per Instance
# ===============================
instances = df["instance_id"].unique()

for instance in instances:
    df_instance = df[df["instance_id"] == instance]
    
    # Simpan ke CSV per instance
    instance_file = f"metrics_{instance}.csv"
    df_instance.to_csv(instance_file, index=False)
    print(f"✅ Data untuk instance {instance} disimpan dalam {instance_file}")
    
    # ===============================
    # 📌 4. Feature Engineering
    # ===============================
    df_instance["hour"] = df_instance["timestamp"].dt.hour
    df_instance["day_of_week"] = df_instance["timestamp"].dt.dayofweek
    
    def classify_status(cpu, mem): # Fungsi untuk mengklasifikasikan / labeling status VM > label (y)
        if cpu < 0.3 and mem < 0.3:
            return 0  # Jika CPU Usage < 30% dan Memory Usage < 30%, maka statusnya 0 (Underutilized).
        elif cpu > 0.8 or mem > 0.8:
            return 2  # Jika CPU Usage > 80% atau Memory Usage > 80%, maka statusnya 2 (Overutilized).
        else:
            return 1  # Jika tidak memenuhi kondisi di atas, maka statusnya 1 (Optimal).
    
    df_instance["status"] = df_instance.apply(lambda row: classify_status(row["cpu_utilization"], row["memory_usage"]), axis=1)
    
    # Pilih fitur
    X = df_instance[["cpu_utilization", "memory_usage", "hour", "day_of_week"]]
    y = df_instance["status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Data dibagi menjadi 80% training data dan 20% testing data.
    
    # ===============================
    # 📌 5. Training Model Decision Tree
    # ===============================
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    
    print(f"📊 Decision Tree Accuracy for {instance}: {acc_dt:.2f}")
    print("📄 Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
    
    # Confusion Matrix Decision Tree
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - Decision Tree ({instance})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # ===============================
    # 📌 6. Training Model Random Forest
    # ===============================
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"🌲 Random Forest Accuracy for {instance}: {acc_rf:.2f}")
    print("📄 Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
    
    # Confusion Matrix Random Forest
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens")
    plt.title(f"Confusion Matrix - Random Forest ({instance})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # ===============================
    # 📌 7. Fungsi Prediksi Autoscaling VM
    # ===============================
    def predict_vm_sizing(cpu, mem, hour, day_of_week):
        input_data = pd.DataFrame([[cpu, mem, hour, day_of_week]], columns=["cpu_utilization", "memory_usage", "hour", "day_of_week"])
        predicted_class = rf_model.predict(input_data)[0]
        
        if predicted_class == 0:
            return "⬇ Underutilized (Downsize VM)"
        elif predicted_class == 1:
            return "✅ Optimal"
        else:
            return "⬆ Overutilized (Upscale VM)"
    
    # Contoh Prediksi untuk Instance
    cpu_usage = 0.85  # 85% CPU Utilization
    memory_usage = 0.90  # 90% Memory Usage
    hour = 14  # Jam 14 (siang)
    day_of_week = 2  # Hari Rabu
    
    recommendation = predict_vm_sizing(cpu_usage, memory_usage, hour, day_of_week)
    print(f"📢 Scaling Recommendation for {instance}: {recommendation}")
