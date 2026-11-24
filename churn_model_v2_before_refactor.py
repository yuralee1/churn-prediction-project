"""
Churn prediction models for the IBM Telco Customer Churn dataset (course project, Stage 5-6).

Final version (v2). Started from an AI-generated draft (see churn_model_v1_ai_draft.py) and
revised after testing. Main changes from v1:
  * reads telco_clean.csv so that ALL scaling/encoding happens inside the pipeline and is fit
    on training folds only (v1 used a file that was already scaled on the full data)
  * hyperparameters tuned with 5-fold stratified CV on the training set only
  * tree models regularized (v1 trees memorized the training data)
  * decision threshold chosen from training-set out-of-fold predictions, not from the test set
  * removed num_addon_services (exact sum of six existing flags) and TotalCharges (tenure x
    MonthlyCharges, r = 0.9996) to eliminate multicollinearity; CV AUC changed by only ~0.001
  * saves a metrics table, figures, and model coefficients/importances
  * built-in sanity checks (run_checks) that must pass before results are written

Usage:  python churn_model.py [path_to_telco_clean.csv] [output_dir]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve, fbeta_score)

SEED = 42
BETA = 2.0                       # threshold chosen to maximize F-beta (beta > 1 favors recall)
BLUE, ORANGE = "#1F4E79", "#C55A11"
NUMERIC = ["tenure", "MonthlyCharges"]
DROP = ["customerID", "gender", "PhoneService", "TotalCharges"]   # identifier, two no-signal variables (EDA), and a redundant one


# ----------------------------------------------------------------------------- data
def load_data(path):
    """Load the cleaned table and return features X and the 0/1 target y."""
    df = pd.read_csv(path)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=DROP + ["Churn"])
    return X, y


def make_preprocessor(X):
    """Scale numeric columns; one-hot encode the rest (binary columns become a single 0/1 column)."""
    cat = [c for c in X.columns if c not in NUMERIC]
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)])


# ----------------------------------------------------------------------------- models
def build_searches(X, cv):
    """One GridSearchCV per model; every model is a full pipeline so preprocessing is fit per fold."""
    def pipe(model):
        return Pipeline([("prep", make_preprocessor(X)), ("model", model)])
    specs = {
        "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=2000),
                                {"model__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                          {"model__max_depth": [3, 4, 5, 6], "model__min_samples_leaf": [20, 50, 100]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
                          {"model__max_depth": [6, 8, 12], "model__min_samples_leaf": [5, 20]}),
    }
    return {n: GridSearchCV(pipe(m), g, cv=cv, scoring="roc_auc", n_jobs=-1) for n, (m, g) in specs.items()}


def choose_threshold(estimator, Xtr, ytr, cv, beta=BETA):
    """Pick the probability cut-off that maximizes F-beta on out-of-fold TRAINING predictions."""
    proba = cross_val_predict(estimator, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(ytr, proba >= t, beta=beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"threshold": round(thr, 2), "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred), "recall": recall_score(y_true, pred),
            "f1": f1_score(y_true, pred), "roc_auc": roc_auc_score(y_true, proba)}


# ----------------------------------------------------------------------------- checks
def run_checks(X, y, Xtr, Xte, ytr, yte, fitted, results):
    """Sanity checks. Raises AssertionError if something is wrong."""
    assert X.isna().sum().sum() == 0, "missing values in features"
    assert not any(c in X.columns for c in DROP + ["Churn"]), "dropped/target column still in X"
    assert len(Xtr) + len(Xte) == len(X) and set(Xtr.index).isdisjoint(Xte.index), "train/test overlap"
    assert abs(ytr.mean() - y.mean()) < 0.005 and abs(yte.mean() - y.mean()) < 0.005, "split is not stratified"
    for name, gs in fitted.items():                                   # scaler must have been fit on training rows only
        sc = gs.best_estimator_.named_steps["prep"].named_transformers_["num"]
        assert np.allclose(sc.mean_, Xtr[NUMERIC].mean().values), f"{name}: scaler not fit on training data"
    assert (results["roc_auc"] > 0.5).all(), "a model is no better than chance"
    assert (results["recall"] <= 1).all() and (results["precision"] <= 1).all()
    print("All sanity checks passed.")


# ----------------------------------------------------------------------------- figures
def plot_evaluation(yte, probas, thresholds, best, outdir):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for (name, p), col in zip(probas.items(), [BLUE, ORANGE, "#7F7F7F"]):
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves (test set)")

    cm = confusion_matrix(yte, probas[best] >= thresholds[best]); pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax[1].imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0f}%)", ha="center", va="center", fontsize=11,
                       color="white" if pct[i, j] > 50 else "black")
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"],
              xlabel="Predicted", ylabel="Actual", title=f"{best}, threshold {thresholds[best]:.2f}")

    for (name, p), col in zip(probas.items(), [BLUE, ORANGE, "#7F7F7F"]):
        pr, rc, _ = precision_recall_curve(yte, p)
        ax[2].plot(rc, pr, color=col, lw=2, label=name)
    ax[2].axhline(yte.mean(), color="k", ls="--", lw=1, label="Baseline (churn rate)")
    ax[2].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)"); ax[2].legend(fontsize=8)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "model_evaluation.png"), dpi=200); plt.close()


def feature_tables(fitted):
    names = fitted["Random Forest"].best_estimator_.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    lr = fitted["Logistic Regression"].best_estimator_.named_steps["model"]
    rf = fitted["Random Forest"].best_estimator_.named_steps["model"]
    odds = pd.DataFrame({"feature": names, "coefficient": lr.coef_[0], "odds_ratio": np.exp(lr.coef_[0])})
    imp = pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
    return odds.sort_values("coefficient", key=abs, ascending=False), imp.sort_values("importance", ascending=False)


def plot_features(odds, imp, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    top = imp.head(10).iloc[::-1]
    ax[0].barh(top.feature, top.importance, color=BLUE); ax[0].set_title("Random Forest: top 10 importances")
    o = odds.head(10).iloc[::-1]
    ax[1].barh(o.feature, o.coefficient, color=[ORANGE if c > 0 else BLUE for c in o.coefficient])
    ax[1].axvline(0, color="k", lw=.8); ax[1].set_title("Logistic Regression: top 10 coefficients\n(orange raises churn risk)")
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "feature_drivers.png"), dpi=200); plt.close()


# ----------------------------------------------------------------------------- main
def main(path="telco_clean.csv", outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    X, y = load_data(path)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    fitted, probas, thresholds, rows = {}, {}, {}, []
    for name, search in build_searches(X, cv).items():
        search.fit(Xtr, ytr)                                          # tuning uses training data only
        fitted[name] = search
        thr = choose_threshold(search.best_estimator_, Xtr, ytr, cv)
        thresholds[name] = thr
        probas[name] = search.predict_proba(Xte)[:, 1]
        train_auc = roc_auc_score(ytr, search.predict_proba(Xtr)[:, 1])
        for label, t in [("0.50", 0.5), (f"tuned (F{BETA:g})", thr)]:
            rows.append({"model": name, "operating_point": label, **metrics(yte, probas[name], t),
                         "cv_auc": search.best_score_, "train_auc": train_auc,
                         "best_params": {k.split("__")[1]: v for k, v in search.best_params_.items()}})
    results = pd.DataFrame(rows)
    run_checks(X, y, Xtr, Xte, ytr, yte, fitted, results)

    show = results.drop(columns="best_params").round(3)
    print("\nTest-set results (train n={}, test n={}, churn rate {:.1%})".format(len(Xtr), len(Xte), y.mean()))
    print(show.to_string(index=False))
    results.to_csv(os.path.join(outdir, "model_results.csv"), index=False)

    best = max(fitted, key=lambda n: fitted[n].best_score_)            # chosen by CV AUC, not by test score
    print(f"\nBest model by cross-validated ROC-AUC: {best}")
    plot_evaluation(yte, probas, thresholds, best, outdir)
    odds, imp = feature_tables(fitted)
    odds.to_csv(os.path.join(outdir, "logistic_odds_ratios.csv"), index=False)
    imp.to_csv(os.path.join(outdir, "rf_importances.csv"), index=False)
    plot_features(odds, imp, outdir)
    print("Top 5 RF importances:\n", imp.head(5).round(3).to_string(index=False))
    return results, odds, imp


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
