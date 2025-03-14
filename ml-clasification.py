# ml-classification.py
# ===============================
# 📌 1. Import Library
# ===============================
from google.cloud import monitoring_v3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
import pytz  # Import pytz for timezone handling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 📌 2. Konfigurasi Google Cloud Monitoring (Stackdriver)
# ===============================
# Pastikan sudah mengaktifkan Google Cloud Monitoring API di GCP Console!
# Pastikan juga sudah melakukan gcloud auth application-default login
client = monitoring_v3.MetricServiceClient()
project_id = "sikopat"  # Ganti dengan project ID GCP kamu
project_name = f"projects/{project_id}"

# Tentukan rentang waktu pengambilan data (misal: 7 hari terakhir)
wib_timezone = pytz.timezone('Asia/Jakarta')  # Define the timezone
end_time_utc = datetime.now(timezone.utc)
end_time_wib = end_time_utc.astimezone(wib_timezone)  # convert to wib
start_time_utc = end_time_utc - timedelta(days=7)
start_time_wib = start_time_utc.astimezone(wib_timezone)  # convert to wib

# ===============================
# 📌 3. Fungsi untuk Mengambil Data CPU & Memory dari GCP Monitoring
# ===============================
def get_metric_data(metric_type):
    """Mengambil data metrik CPU/MEMORY dari Google Cloud Monitoring"""
    interval = monitoring_v3.TimeInterval(
        start_time=start_time_utc,  # Keep using UTC for the API, as the API expects it.
        end_time=end_time_utc,  # Keep using UTC for the API, as the API expects it.
    )

    results = client.list_time_series(
        name=project_name,
        filter=f'metric.type="{metric_type}"',
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )

    # Parsing data ke dalam DataFrame
    data_list = []
    for series in results:
        for point in series.points:
            timestamp_utc = point.interval.end_time
            timestamp_wib = timestamp_utc.astimezone(wib_timezone)  # convert to wib
            instance = series.resource.labels.get("instance_id", "unknown_instance")  # Handle missing instance_id label
            value = point.value.double_value  # memory percent is a double.
            data_list.append([instance, timestamp_wib, value])  # store wib timestamp

    return pd.DataFrame(data_list, columns=["instance_id", "timestamp", metric_type.split("/")[-1].replace("percent_used", "memory_usage").replace("utilization", "cpu_utilization")])

# Mengambil data CPU & Memory Usage
cpu_df = get_metric_data("agent.googleapis.com/cpu/utilization")
mem_df = get_metric_data("agent.googleapis.com/memory/percent_used")

print(f"CPU DataFrame size: {len(cpu_df)}")
print(f"Memory DataFrame size: {len(mem_df)}")

# Round timestamps to the nearest second
cpu_df["timestamp"] = cpu_df["timestamp"].dt.round("s")
mem_df["timestamp"] = mem_df["timestamp"].dt.round("s")

# Gabungkan data CPU & Memory berdasarkan instance_id dan timestamp
df = pd.merge(cpu_df, mem_df, on=["instance_id", "timestamp"], how="inner")

print(f"Merged DataFrame size: {len(df)}")

# Simpan ke CSV (opsional)
if not df.empty:
    df.to_csv("gcp_vm_metrics.csv", index=False)
    print("✅ Data berhasil diambil dan disimpan!")
else:
    print("⚠️ No Data was found, CSV not created")

# ===============================
# 📌 4. Preprocessing Data untuk ML
# ===============================
if not df.empty:
    # Load data jika ingin langsung dari CSV
    # df = pd.read_csv("gcp_vm_metrics.csv") #Uncomment if you want to load from csv instead of the dataframe created above.

    # Konversi timestamp menjadi datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Feature Engineering: Tambahkan informasi waktu (jam, hari dalam seminggu)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Normalize memory usage from bytes to percentage (approximate)
    # The agent.googleapis.com/memory/percent_used metric already returns a percentage, so no conversion is needed
    # Check if memory usage is zero and print warning
    if (df['memory_usage'] == 0).all():
        print("⚠️ Warning: All memory usage data is zero. Please check your VM's monitoring agent.")

    # Normalisasi CPU & Memory Usage ke kategori status
    def classify_status(cpu, mem):
        if cpu < 0.3 and mem < 0.3:
            return 0  # Underutilized (bisa scale down)
        elif cpu > 0.8 or mem > 0.8:
            return 2  # Overutilized (perlu scale up)
        else:
            return 1  # Normal

    df["status"] = df.apply(lambda row: classify_status(row["cpu_utilization"], row["memory_usage"]), axis=1)

    # Pilih fitur yang akan digunakan untuk training
    X = df[["cpu_utilization", "memory_usage", "hour", "day_of_week"]]
    y = df["status"]

    # Explicitly set column names for X before splitting
    X.columns = ["cpu_utilization", "memory_usage", "hour", "day_of_week"]

    # Split data menjadi training & testing (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===============================
    # 📌 5. Training Model Decision Tree
    # ===============================
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    print(f"📊 Decision Tree Accuracy: {acc_dt:.2f}")
    print("📄 Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

    # Confusion Matrix Decision Tree
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Decision Tree")
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

    print(f"🌲 Random Forest Accuracy: {acc_rf:.2f}")
    print("📄 Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

    # Confusion Matrix Random Forest
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ===============================
    # 📌 7. Fungsi Prediksi Autoscaling VM
    # ===============================
    def predict_vm_sizing(cpu, mem, hour, day_of_week):
        """Memprediksi kebutuhan scaling berdasarkan CPU & Memory usage"""
        input_data = pd.DataFrame([[cpu, mem, hour, day_of_week]], columns=["cpu_utilization", "memory_usage", "hour", "day_of_week"]) #Create a dataframe with column names
        predicted_class = rf_model.predict(input_data)[0]

        if predicted_class == 0:
            return "⬇ Underutilized (Downsize VM)"
        elif predicted_class == 1:
            return "✅ Optimal"
        else:
            return "⬆ Overutilized (Upscale VM)"

    # Contoh Prediksi untuk VM baru
    cpu_usage = 0.85  # 85% CPU Utilization
    memory_usage = 0.90  # 90% Memory Usage
    hour = 14  # Jam 14 (siang)
    day_of_week = 2  # Hari Rabu

    recommendation = predict_vm_sizing(cpu_usage, memory_usage, hour, day_of_week)
    print(f"📢 Scaling Recommendation: {recommendation}")
else:
    print("Cannot proceed with ML as no data was fetched.")