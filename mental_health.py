import pandas as pd

df = pd.read_csv("Maternal Health Risk Data Set.csv")

print(df.shape)
print(df.dtypes)
print(df.isna().sum())

print(df["RiskLevel"].value_counts())
print(df["RiskLevel"].value_counts(normalize=True).round(3))

print("Duplicate rows:", df.duplicated().sum())
print(df.describe().T)
print(df.groupby("RiskLevel").mean(numeric_only=True).round(1))

import numpy as np
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, recall_score

feat = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
df["y"] = df["RiskLevel"].map({"low risk": 0, "mid risk": 1, "high risk": 2})

# 1. impossible heart rate wale rows
print(df[df["HeartRate"] < 40])
df = df[df["HeartRate"] >= 40].copy()

# 2. duplicates hata ke
d = df.drop_duplicates().copy()
print("After cleaning:", d.shape)
print(d["y"].value_counts().sort_index())
print("Same features but different label:", d.duplicated(subset=feat).sum())

# 3. duplicates ka asar (leakage ka saboot)
def rf_acc(data):
    r = cross_validate(RandomForestClassifier(n_estimators=300, random_state=42),
                       data[feat], data["y"],
                       cv=StratifiedKFold(5, shuffle=True, random_state=42),
                       scoring="accuracy")
    return r["test_score"].mean()

print("RF accuracy WITH duplicates:    %.3f" % rf_acc(df))
print("RF accuracy WITHOUT duplicates: %.3f" % rf_acc(d))

# 4. train/test split (test set abhi mat chhoona)
X, y = d[feat], d["y"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", Xtr.shape, "Test:", Xte.shape)

# 5. models ka muqabla (sirf train par, repeated CV)
high_recall = make_scorer(lambda a, b: recall_score(a, b, labels=[2], average="macro"))
scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro", "high_recall": high_recall}
models = {
    "Dummy (baseline)": DummyClassifier(strategy="prior"),
    "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
for name, m in models.items():
    r = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring)
    print(f"\n{name}")
    for k in scoring:
        s = r[f"test_{k}"]
        print(f"  {k:12s} {s.mean():.3f} (+/- {s.std():.3f})")


import joblib
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance

names = ["low", "mid", "high"]
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                            class_weight="balanced", random_state=42)
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)

# 1. test set par natija
print(classification_report(yte, pred, target_names=names, digits=3))
print(confusion_matrix(yte, pred))

# 2. bootstrap confidence interval (test set chhota hai)
rng = np.random.default_rng(42)
yt = yte.values
rec, f1s = [], []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))
    rec.append(recall_score(yt[i], pred[i], labels=[2], average="macro"))
    f1s.append(f1_score(yt[i], pred[i], average="macro"))
print("High-risk recall 95%% CI: %.2f to %.2f" % tuple(np.percentile(rec, [2.5, 97.5])))
print("Macro-F1 95%% CI: %.2f to %.2f" % tuple(np.percentile(f1s, [2.5, 97.5])))

# 3. kaun si features zaroori hain
pi = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=42, scoring="f1_macro")
imp = pd.DataFrame({"feature": feat,
                    "perm_importance": pi.importances_mean.round(3),
                    "perm_std": pi.importances_std.round(3),
                    "rf_importance": rf.feature_importances_.round(3)})
print(imp.sort_values("perm_importance", ascending=False))

# 4. high-risk ke liye threshold (sirf train ke out-of-fold se)
oof = cross_val_predict(rf, Xtr, ytr,
                        cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba")
actual = (ytr.values == 2).astype(int)
print("\nThreshold analysis (train, out-of-fold):")
for t in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    p = (oof[:, 2] >= t).astype(int)
    print(f"t={t:.2f}  recall={recall_score(actual, p):.3f}  "
          f"precision={precision_score(actual, p, zero_division=0):.3f}  flagged={p.sum()}")

# 5. save
joblib.dump(rf, "maternal_rf.joblib")
d.to_csv("maternal_clean.csv", index=False)
print("Saved.")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

T = 0.30
p_high = rf.predict_proba(Xte)[:, 2]
alert = (p_high >= T).astype(int)
actual_high = (yte.values == 2).astype(int)

tn, fp, fn, tp = confusion_matrix(actual_high, alert).ravel()
print(f"\nHigh-risk alert at t={T}: TP={tp} FN={fn} FP={fp} TN={tn}")
print(f"recall={tp/(tp+fn):.3f}  precision={tp/(tp+fp):.3f}")

# bootstrap CI
rec_b, pre_b = [], []
for _ in range(1000):
    i = rng.integers(0, len(actual_high), len(actual_high))
    a, p_ = actual_high[i], alert[i]
    if a.sum() == 0 or p_.sum() == 0:
        continue
    rec_b.append(recall_score(a, p_))
    pre_b.append(precision_score(a, p_))
print("Recall 95%% CI: %.2f to %.2f" % tuple(np.percentile(rec_b, [2.5, 97.5])))
print("Precision 95%% CI: %.2f to %.2f" % tuple(np.percentile(pre_b, [2.5, 97.5])))

# kya missed cases (agar hain) asal mein kaun se the?
missed = (actual_high == 1) & (alert == 0)
print("Missed high-risk cases:\n", Xte[missed].assign(p_high=p_high[missed].round(2)))

# --- figures ---
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=names, ax=ax, cmap="Blues")
ax.set_title("Confusion matrix (test set)")
plt.tight_layout(); plt.savefig("cm.png", dpi=150); plt.close()

order = imp.sort_values("perm_importance")
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.barh(order["feature"], order["perm_importance"], xerr=order["perm_std"])
ax.set_xlabel("Permutation importance (drop in macro-F1)")
plt.tight_layout(); plt.savefig("importance.png", dpi=150); plt.close()

ts = np.arange(0.10, 0.65, 0.05)
r = [recall_score(actual, (oof[:, 2] >= t).astype(int)) for t in ts]
pr = [precision_score(actual, (oof[:, 2] >= t).astype(int), zero_division=0) for t in ts]
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(ts, r, marker="o", label="Recall")
ax.plot(ts, pr, marker="o", label="Precision")
ax.axvline(T, color="gray", linestyle="--", label=f"chosen t={T}")
ax.set_xlabel("Threshold for high-risk alert"); ax.legend()
plt.tight_layout(); plt.savefig("threshold.png", dpi=150); plt.close()
print("Saved: cm.png, importance.png, threshold.png")