"""Data preparation for IBM Telco Customer Churn (Stages 2-3)."""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

raw = pd.read_csv('raw.csv')
log = {}
log['raw_shape'] = raw.shape

# --- Stage 2: preparation / cleaning ---
df = raw.copy()
df.columns = [c.strip() for c in df.columns]
log['dup_rows'] = int(df.duplicated().sum())
log['dup_ids'] = int(df.customerID.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# TotalCharges stored as text: blanks for tenure==0 customers
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
log['totalcharges_missing'] = int(df.TotalCharges.isna().sum())
log['missing_all_tenure0'] = bool((df[df.TotalCharges.isna()].tenure == 0).all())
log['missing_all_not_churned'] = bool((df[df.TotalCharges.isna()].Churn == 'No').all())

# --- Missing value treatment ---
# Brand-new customers (tenure 0) have not been billed yet -> TotalCharges = 0
df.loc[df.TotalCharges.isna() & (df.tenure == 0), 'TotalCharges'] = 0.0
log['missing_after'] = int(df.isna().sum().sum())

# Consistency: collapse 'No internet service' / 'No phone service' to 'No'
svc = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for c in svc: df[c] = df[c].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

# Types
df['SeniorCitizen'] = df['SeniorCitizen'].map({0:'No',1:'Yes'})
cat_cols = df.select_dtypes(include=['str','object']).columns.drop('customerID')
for c in cat_cols: df[c] = df[c].astype('category')
log['clean_dtypes'] = df.dtypes.astype(str).to_dict()

# Outlier check (IQR)
out = {}
for c in ['tenure','MonthlyCharges','TotalCharges']:
    q1,q3 = df[c].quantile([.25,.75]); i = q3-q1
    out[c] = int(((df[c]<q1-1.5*i)|(df[c]>q3+1.5*i)).sum())
log['iqr_outliers'] = out

df.to_csv('telco_clean.csv', index=False)

# --- Stage 3: preprocessing / feature engineering ---
m = df.copy()
m['Churn'] = (m['Churn']=='Yes').astype(int)
binary = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','PaperlessBilling']+svc
for c in binary:
    m[c] = (m[c].astype(str).isin(['Yes','Male'])).astype(int)
m = m.rename(columns={'gender':'is_male'})

# engineered features
m['num_addon_services'] = m[svc].sum(axis=1)
m['avg_monthly_spend'] = np.where(m.tenure>0, m.TotalCharges/m.tenure.replace(0,np.nan), m.MonthlyCharges)
m['tenure_group'] = pd.cut(df.tenure,[-1,12,24,48,72],labels=['0-12','13-24','25-48','49-72']).astype(str)

m = pd.get_dummies(m, columns=['InternetService','Contract','PaymentMethod','tenure_group'], drop_first=True, dtype=int)
ids = m.pop('customerID')
y = m.pop('Churn')
num = ['tenure','MonthlyCharges','TotalCharges','avg_monthly_spend','num_addon_services']
m[num] = StandardScaler().fit_transform(m[num])  # fit on full data for documentation only; refit on train split at modeling
out_df = pd.concat([ids,m,y],axis=1)
out_df.to_csv('telco_model_ready.csv', index=False)
log['final_shape'] = out_df.shape
log['final_cols'] = out_df.columns.tolist()
log['class_balance'] = y.value_counts().to_dict()
log['corr_top'] = out_df.drop(columns='customerID').corr()['Churn'].drop('Churn').sort_values(key=abs,ascending=False).head(8).round(3).to_dict()
print(log)
