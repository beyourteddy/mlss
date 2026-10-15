import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    f1_score, ConfusionMatrixDisplay, confusion_matrix
)

tj = pd.read_csv("trojan_30000.csv")

#print("Dataset shape:", tj.shape)
#print("Class distribution:")
#print(tj["Class"].value_counts())

# แยก Features และ Target
X = tj.drop("Class", axis=1)
Y = tj["Class"]

# แบ่ง Train และ Test ก่อนเลือก Features
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=0,
    stratify=Y
)

# เลือก 10 Features ที่มีคะแนนสูงสุด
selector = SelectKBest(score_func=f_classif, k=10)

X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)

# แสดงชื่อ Features ที่ถูกเลือก
selected_features = X.columns[selector.get_support()]

print("Selected Features:")
for feature in selected_features:
    print(feature)

