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