import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

df = pd.read_csv('../creditcard.csv')
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Доля мошеннических в train: {y_train.mean():.5f}")
print(f"Доля мошеннических в test:  {y_test.mean():.5f}")

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)
f1_bag = f1_score(y_test, y_pred_bag, average='macro')
print(f"\nBagging - macro F1: {f1_bag:.4f}")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')
print(f"RandomForest — macro F1: {f1_rf:.4f}")

lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_train, y_train)
y_pred_lgb = lgbm.predict(X_test)
f1_lgb = f1_score(y_test, y_pred_lgb, average='macro')
print(f"LightGBM — macro F1: {f1_lgb:.4f}")

print("\n=== Итоговое сравнение ===")
print(f"Bagging:      {f1_bag:.4f}")
print(f"RandomForest: {f1_rf:.4f}")
print(f"LightGBM:     {f1_lgb:.4f}")

print("\nClassification report LightGBM:")
print(classification_report(y_test, y_pred_lgb, digits=4))