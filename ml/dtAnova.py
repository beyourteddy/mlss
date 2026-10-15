
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
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

X = X.select_dtypes(include="number")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

print("\nจำนวน Features เริ่มต้น:", X.shape[1])

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
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train:", len(X_train))
print("Test:", len(X_test))

# =====================================
# 7. เติมค่าว่างด้วยข้อมูล Train
# =====================================

imputer = SimpleImputer(strategy="median")

X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ลบ Features ที่กลายเป็นค่าคงที่หลังเติมค่าว่าง
valid_columns = X_train_imp.nunique(dropna=True) > 1

X_train_imp = X_train_imp.loc[:, valid_columns]
X_test_imp = X_test_imp.loc[:, valid_columns]

print(
    "\nจำนวน Features หลังทำความสะอาด:",
    X_train_imp.shape[1]
)

# =====================================
# 8. ทดลอง ANOVA Feature Selection
# =====================================

feature_counts = [10, 20, 30, 40, 50, 60, 70]
results = []

available_features = X_train_imp.shape[1]

for requested_k in feature_counts:

    k = min(requested_k, available_features)

    # ANOVA เลือก Features
    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = selector.fit_transform(
        X_train_imp,
        y_train
    )

    X_test_selected = selector.transform(
        X_test_imp
    )

    selected_features = X_train_imp.columns[
        selector.get_support()
    ]

    # =================================
    # 9. Decision Tree
    # =================================

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    # =================================
    # 10. ประเมินผล
    # =================================

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

    results.append({
        "Requested Features": requested_k,
        "Actual Features": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"\n===== Decision Tree + ANOVA: {requested_k} Features =====")
    print("จำนวน Features ที่ใช้จริง:", k)
    print("Features:", selected_features.tolist())
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

# =====================================
# 11. ตารางเปรียบเทียบผล
# =====================================

result_df = pd.DataFrame(results)

print("\n===== Decision Tree + ANOVA Results =====")

if not result_df.empty:

    print(result_df.round(4).to_string(index=False))

    result_df.to_csv(
        "DecisionTree_ANOVA_results.csv",
        index=False
    )

    best_row = result_df.loc[
        result_df["F1-score"].idxmax()
    ]

    print("\n===== ผลที่มี Weighted F1-score สูงสุด =====")
    print("จำนวน Features:", int(best_row["Actual Features"]))
    print("Accuracy:", round(best_row["Accuracy"], 4))
    print("Precision:", round(best_row["Precision"], 4))
    print("Recall:", round(best_row["Recall"], 4))
    print("F1-score:", round(best_row["F1-score"], 4))

    print("\nบันทึกผลใน DecisionTree_ANOVA_results.csv แล้ว")

else:
    print("ไม่มีผลการทดลอง")
