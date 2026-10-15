
import pandas as pd
import numpy as np
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# 1. อ่านและทำความสะอาดข้อมูล
# =====================================

df = pd.read_csv("Trojan_New.csv")

print("ข้อมูลเริ่มต้น:", df.shape)

# ลบข้อมูลซ้ำ
df = df.drop_duplicates()

# ลบแถวที่ไม่มี Class
df = df.dropna(subset=["Class"])

# สุ่มข้อมูลไม่เกิน 30,000 แถว โดยรักษาสัดส่วน Class
n = min(30000, len(df))

if len(df) > n:
    df, _ = train_test_split(
        df,
        train_size=n,
        stratify=df["Class"],
        random_state=42
    )

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ลบคอลัมน์ที่ไม่ใช้
df = df.drop(
    columns=[
        "Unnamed: 0",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ],
    errors="ignore"
)

# แยก Features และ Target
X = df.drop(columns=["Class"])
y = df["Class"]

# ใช้เฉพาะ Features ที่เป็นตัวเลข
X = X.select_dtypes(include="number")

# แทนค่า Infinity ด้วย NaN
X = X.replace([np.inf, -np.inf], np.nan)

# ลบคอลัมน์ที่ไม่มีข้อมูลเลย
X = X.dropna(axis=1, how="all")

# ลบคอลัมน์ที่มีค่าเดียว
X = X.loc[:, X.nunique(dropna=True) > 1]

print("จำนวนข้อมูลหลังทำความสะอาด:", X.shape)

# =====================================
# 2. แปลง Class เป็นตัวเลข
# =====================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Class ที่พบ:", encoder.classes_)

# =====================================
# 3. แบ่ง Train และ Test
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================
# 4. จัดการค่าว่าง
# =====================================

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# =====================================
# 5. ANOVA Feature Selection
# =====================================

# คำนวณคะแนน ANOVA จากชุด Train เท่านั้น
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    scores, p_values = f_classif(X_train, y_train)

# ตัด Features ที่คะแนนเป็น NaN หรือ -Infinity
valid_indices = np.where(
    ~np.isnan(scores) & (scores != -np.inf)
)[0]

if len(valid_indices) == 0:
    raise ValueError(
        "ไม่พบ Features ที่มีคะแนน ANOVA ใช้งานได้"
    )

# เรียง Features จากคะแนน ANOVA มากไปน้อย
ranked_indices = valid_indices[
    np.argsort(scores[valid_indices])[::-1]
]

print("จำนวน Features ที่ใช้ได้:", len(ranked_indices))

# =====================================
# 6. ทดลอง ANN หลายค่า k
# =====================================

feature_counts = [10, 20, 30, 40, 50, 60, 70]
results = []

for requested_k in feature_counts:

    # จำนวน Features จริงต้องไม่เกินที่มี
    k = min(requested_k, len(ranked_indices))

    # เลือก Features ที่มีคะแนน ANOVA สูงสุด
    selected_indices = ranked_indices[:k]

    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    # ANN ต้องปรับสเกลข้อมูล
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # สร้างโมเดล ANN
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=42
    )

    # ฝึกและจับเวลา
    start_time = time.time()

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    elapsed_time = time.time() - start_time

    # ประเมินผล
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Requested k": requested_k,
        "Features Used": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Time (seconds)": elapsed_time
    })

    print(
        f"k={requested_k} | "
        f"Features={k} | "
        f"Accuracy={accuracy:.4f} | "
        f"F1-score={f1:.4f} | "
        f"Time={elapsed_time:.2f} sec"
    )

# =====================================
# 7. แสดงผลเปรียบเทียบ
# =====================================

results_df = pd.DataFrame(results)

print("\nผลการทดลอง ANN + ANOVA")
print(results_df.round(4).to_string(index=False))

# บันทึกผลเป็น CSV
results_df.to_csv(
    "ANN_ANOVA_results.csv",
    index=False
)

# หา k ที่ให้ F1-score สูงที่สุด
best_result = results_df.loc[
    results_df["F1-score"].idxmax()
]

print("\nผลลัพธ์ที่มี F1-score สูงที่สุด")
print(best_result.to_string())

print("\nบันทึกผลไว้ใน ANN_ANOVA_results.csv แล้ว")
