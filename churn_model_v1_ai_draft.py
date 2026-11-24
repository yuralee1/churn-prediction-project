"""AI-generated first draft (v1) from Prompt 2. Kept unmodified for documentation."""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['customerID', 'avg_monthly_spend', 'is_male', 'PhoneService'])
    X = df.drop(columns='Churn'); y = df['Churn']
    return X, y

def build_models():
    return {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(class_weight='balanced', max_iter=1000))]),
        'Decision Tree': Pipeline([('scaler', StandardScaler()), ('model', DecisionTreeClassifier(class_weight='balanced', random_state=42))]),
        'Random Forest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(class_weight='balanced', random_state=42))]),
    }

def evaluate(models, Xtr, Xte, ytr, yte):
    rows = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte); proba = m.predict_proba(Xte)[:, 1]
        rows.append({'model': name, 'accuracy': accuracy_score(yte, pred), 'precision': precision_score(yte, pred),
                     'recall': recall_score(yte, pred), 'f1': f1_score(yte, pred), 'roc_auc': roc_auc_score(yte, proba),
                     'train_recall': recall_score(ytr, m.predict(Xtr))})
    return pd.DataFrame(rows).sort_values('recall', ascending=False)

if __name__ == '__main__':
    X, y = load_data('telco_model_ready.csv')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(evaluate(build_models(), Xtr, Xte, ytr, yte).round(3).to_string(index=False))
