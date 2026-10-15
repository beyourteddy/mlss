
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif

# =====================================
# 1. อ่านและทำความสะอาดข้อมูล
# =====================================

df = pd.read_csv("Trojan_Detection.csv")

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

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ลบคอลัมน์ที่ไม่ต้องการ
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

# จัดการค่า inf และคอลัมน์ว่าง
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

# ลบ Features ที่มีค่าเดียวตลอด
X = X.loc[:, X.nunique(dropna=True) > 1]

print("จำนวนข้อมูล:", len(X))
print("จำนวน Features เริ่มต้น:", X.shape[1])

# เข้ารหัส Class
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# =====================================
# 2. เติมค่าว่าง
# =====================================

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# =====================================
# 3. ANOVA F-test
# =====================================

scores, p_values = f_classif(X_imputed, y_encoded)

anova_df = pd.DataFrame({
    "Feature": X_imputed.columns,
    "ANOVA_Score": scores,
    "P_Value": p_values
})

# ตัดคะแนนที่เป็น NaN หรือ Infinity
anova_df = anova_df.replace(
    [np.inf, -np.inf], np.nan
)

anova_df = anova_df.dropna(
    subset=["ANOVA_Score"]
)

# เรียงคะแนนจากมากไปน้อย
anova_df = anova_df.sort_values(
    by="ANOVA_Score",
    ascending=False
).reset_index(drop=True)

ranked_features = anova_df["Feature"].tolist()

print("\nจำนวน Features หลัง ANOVA:", len(ranked_features))

# =====================================
# 4. Correlation Filtering
# =====================================

# เริ่มจาก Features ที่มี ANOVA Score สูงก่อน
# ตัด Feature ที่สัมพันธ์กับ Feature ที่เลือกแล้ว
# มากกว่า threshold ที่กำหนด

CORR_THRESHOLD = 0.90

selected_features = []

for feature in ranked_features:

    keep_feature = True

    for selected in selected_features:

        corr = X_imputed[feature].corr(
            X_imputed[selected]
        )

        if pd.notna(corr) and abs(corr) > CORR_THRESHOLD:
            keep_feature = False
            break

    if keep_feature:
        selected_features.append(feature)

print("จำนวน Features หลัง Correlation:",
      len(selected_features))

# =====================================
# 5. เตรียมคะแนนสำหรับ Scree Plot
# =====================================

score_map = anova_df.set_index(
    "Feature"
)["ANOVA_Score"]

selected_scores = np.array([
    score_map[feature]
    for feature in selected_features
])

# Scree Plot ใช้คะแนน ANOVA แบบเรียงลำดับ
# เพื่อดูบริเวณที่คะแนนเริ่มลดลงช้าลง
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(selected_scores) + 1),
    selected_scores,
    marker="o",
    markersize=3
)

plt.xlabel("Number of Features (k)")
plt.ylabel("ANOVA F-score")
plt.title("Hybrid Feature Selection: ANOVA + Correlation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Hybrid_Scree_Plot.png", dpi=300)
plt.show()

# =====================================
# 6. ประมาณค่า k ด้วย Elbow Method
# =====================================

# ปรับคะแนนให้อยู่ในช่วง 0 ถึง 1
# แล้วหาระยะห่างจากเส้นตรงที่เชื่อม
# จุดแรกกับจุดสุดท้ายของกราฟ

if len(selected_scores) >= 3:

    scores_norm = (
        (selected_scores - selected_scores.min())
        / (selected_scores.max() - selected_scores.min() + 1e-12)
    )

    x_norm = np.linspace(0, 1, len(scores_norm))
    y_norm = scores_norm

    # เส้นตรงระหว่างจุดเริ่มต้นกับจุดสุดท้าย
    y_line = (
        y_norm[0]
        + (y_norm[-1] - y_norm[0]) * x_norm
    )

    distances = np.abs(y_norm - y_line)

    # ไม่นับจุดปลายทั้งสอง
    distances[0] = 0
    distances[-1] = 0

    best_k = int(np.argmax(distances)) + 1

else:
    best_k = len(selected_scores)

print("\n===== ผลลัพธ์ Hybrid Feature Selection =====")
print("ค่า k ที่ประมาณจาก Elbow:", best_k)

final_features = selected_features[:best_k]

print("จำนวน Features ที่เลือก:", len(final_features))
print("\nรายชื่อ Features:")

for i, feature in enumerate(final_features, start=1):
    print(f"{i}. {feature}")

# =====================================
# 7. บันทึกผล
# =====================================

anova_df.to_csv(
    "Hybrid_ANOVA_Ranking.csv",
    index=False
)

pd.DataFrame({
    "Selected_Feature": final_features
}).to_csv(
    "Hybrid_Selected_Features.csv",
    index=False
)

# บันทึกข้อมูลที่เลือก Features แล้ว
df_selected = X_imputed[final_features].copy()
df_selected["Class"] = y.values

df_selected.to_csv(
    "Hybrid_Trojan_Selected_Data.csv",
    index=False
)

print("\nบันทึกไฟล์เรียบร้อย:")
print("- Hybrid_Scree_Plot.png")
print("- Hybrid_ANOVA_Ranking.csv")
print("- Hybrid_Selected_Features.csv")
print("- Hybrid_Trojan_Selected_Data.csv")
