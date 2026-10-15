
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================
# 1. อ่านข้อมูล
# =====================================

df = pd.read_csv("Trojan_New.csv")

# =====================================
# 2. ทำความสะอาดข้อมูล
# =====================================

df = df.drop_duplicates()
df = df.dropna(subset=["Class"])

# จำกัดข้อมูลไม่เกิน 30,000 แถว
n = min(30000, len(df))

if len(df) > n:
    df, _ = train_test_split(
        df,
        train_size=n,
        stratify=df["Class"],
        random_state=42
    )

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("จำนวนข้อมูล:", len(df))
print("\nจำนวนแต่ละ Class:")
print(df["Class"].value_counts())

# =====================================
# 3. ลบคอลัมน์ที่ไม่ใช้
# =====================================

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
# 4. แยก Features และ Target
# =====================================

X = df.drop(columns=["Class"])
y = df["Class"]

# ใช้เฉพาะ Features ตัวเลข
X = X.select_dtypes(include="number")

# แทน Infinity ด้วย NaN
X = X.replace([np.inf, -np.inf], np.nan)

# ลบคอลัมน์ที่ว่างทั้งหมด
X = X.dropna(axis=1, how="all")

# ลบคอลัมน์ที่มีค่าเดียวตลอด
X = X.loc[:, X.nunique(dropna=True) > 1]

print("\nจำนวน Features ที่ใช้:", X.shape[1])

# =====================================
# 5. แปลง Class เป็นตัวเลข
# =====================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Class:", encoder.classes_)

# =====================================
# 6. แบ่ง Train/Test 80:20
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", len(X_train))
print("Test:", len(X_test))

# =====================================
# 7. สร้าง SVM RBF
# =====================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# =====================================
# 8. ฝึกโมเดล
# =====================================

model.fit(X_train, y_train)

# =====================================
# 9. ทำนายผล
# =====================================

y_pred = model.predict(X_test)

# =====================================
# 10. ประเมินผล
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

print("\n===== SVM RBF Results (No Feature Selection) =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# =====================================
# 11. บันทึกผล
# =====================================

result_df = pd.DataFrame([{
    "Model": "SVM RBF",
    "Feature Selection": "None",
    "Features": X.shape[1],
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
}])

result_df.to_csv(
    "SVM_No_Feature_Selection_results.csv",
    index=False
)

print("\nบันทึกผลใน SVM_No_Feature_Selection_results.csv แล้ว")
