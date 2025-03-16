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

# 📌 1. Konfigurasi Google Cloud Monitoring (Stackdriver) & Ambil Data
client = monitoring_v3.MetricServiceClient()
project_id = "sikopat" 
project_name = f"projects/{project_id}"

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

# 📌 2. Load Dataset dari CSV
file_path = "gcp_vm_metrics.csv"
df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 📌 3. Pisahkan Dataset per Instance
instances = df["instance_id"].unique()

for instance in instances:
    df_instance = df[df["instance_id"] == instance]
    
    instance_file = f"metrics_{instance}.csv"
    df_instance.to_csv(instance_file, index=False)
    print(f"✅ Data untuk instance {instance} disimpan dalam {instance_file}")
    
    # 📌 4. Feature Engineering
    df_instance["hour"] = df_instance["timestamp"].dt.hour
    df_instance["day_of_week"] = df_instance["timestamp"].dt.dayofweek
    
    def classify_status(cpu, mem):
        if cpu < 0.3 and mem < 0.3:
            return "Underutilized"
        elif cpu > 0.8 or mem > 0.8:
            return "Overutilized"
        else:
            return "Optimal"
    
    df_instance["status"] = df_instance.apply(lambda row: classify_status(row["cpu_utilization"], row["memory_usage"]), axis=1)
    
    X = df_instance[["cpu_utilization", "memory_usage", "hour", "day_of_week"]]
    y = df_instance["status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 📌 5. Training Model Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    df_instance["Decision_Tree_Prediction"] = dt_model.predict(X)
    
    # 📌 6. Training Model Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    df_instance["Random_Forest_Prediction"] = rf_model.predict(X)
    
    # 📌 7. Fungsi Hasil Prediksi Decision Tree
    df_instance["Decision_Tree_Prediction"] = df_instance["Decision_Tree_Prediction"].replace({"0": "Underutilized", "1": "Optimal", "2": "Overutilized"})
    
    # 📌 8. Fungsi Hasil Prediksi Decision Random Forest
    df_instance["Random_Forest_Prediction"] = df_instance["Random_Forest_Prediction"].replace({"0": "Underutilized", "1": "Optimal", "2": "Overutilized"})
    
    df_instance.to_csv(f"predictions_{instance}.csv", index=False)
    print(f"✅ Hasil prediksi diperbarui dengan label teks untuk instance {instance}")
