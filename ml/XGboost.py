
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier

# =====================================
# 1. อ่านและทำความสะอาดข้อมูล
# =====================================

df = pd.read_csv("Trojan_New.csv")

print("ข้อมูลเริ่มต้น:", df.shape)

# ลบข้อมูลซ้ำ
df = df.drop_duplicates()

# ลบแถวที่ไม่มี Class
df = df.dropna(subset=["Class"])

# สุ่มข้อมูลไม่เกิน 30,000 แถว
# โดยรักษาสัดส่วนของแต่ละ Class
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

# =====================================
# 2. แยก Features และ Target
# =====================================

X = df.drop(columns=["Class"])
y = df["Class"]

# ใช้เฉพาะ Features ที่เป็นตัวเลข
X = X.select_dtypes(include="number")

# จัดการ Infinity
X = X.replace([np.inf, -np.inf], np.nan)

# ลบคอลัมน์ที่ไม่มีข้อมูลเลย
X = X.dropna(axis=1, how="all")

# ลบคอลัมน์ที่มีค่าเดียว
X = X.loc[:, X.nunique(dropna=True) > 1]

print("จำนวน Features ที่ใช้ทั้งหมด:", X.shape[1])

# แปลง Class เป็นตัวเลข
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Class:", encoder.classes_)

# =====================================
# 3. แบ่ง Train และ Test
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
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
# 5. สร้างโมเดล XGBoost
# =====================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# =====================================
# 6. ฝึกและทดสอบโมเดล
# =====================================

start_time = time.time()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

elapsed_time = time.time() - start_time

# =====================================
# 7. ประเมินผล
# =====================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)

print("\n===== XGBoost Results =====")
print("Features used:", X_train.shape[1])
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Time:      {elapsed_time:.2f} seconds")

print("\nConfusion Matrix:")
print(cm)
