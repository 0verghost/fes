import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

df = pd.read_csv('../creditcard.csv')
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

lgb_params = {
    'num_leaves': [8, 53],
    'min_child_samples': [64, 90]
}
lgbm = lgb.LGBMClassifier(
    class_weight='balanced', random_state=42, n_jobs=1, verbosity=-1
)
lgb_search = GridSearchCV(
    lgbm,
    lgb_params,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
lgb_search.fit(X_train, y_train)
lgb_best = lgb_search.best_estimator_
f1_lgb = f1_score(y_test, lgb_best.predict(X_test), average='macro')

print("=== Macro F1 ===")
print(f"LightGBM: {f1_lgb:.4f}")


models = {'LightGBM': (lgb_best, f1_lgb)}
model = lgb_best

y_pred_best = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Легальные (0)', 'Мошеннические (1)'])
# disp.plot(cmap=plt.cm.Blues, values_format='d')
# plt.title(f'Матрица ошибок — {'LightGBM'}')
# plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")