
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDataset Info:")
    print(df.info())
    fraud_dist = df['isFraud'].value_counts(normalize=True)
    print("\nFraud distribution:\n", fraud_dist)
    return df

def preprocess_features(df):
    df_processed = df.copy()
    # Feature engineering
    df_processed['balance_change_orig'] = df_processed['oldbalanceOrg'] - df_processed['newbalanceOrig']
    df_processed['balance_change_dest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
    
    # Label encoding
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        if col != 'isFraud':
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Scaling
    X = df_processed.drop('isFraud', axis=1)
    y = df_processed['isFraud']
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = RobustScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, scaler, label_encoders

def train_xgboost_model(X_train, X_test, y_train, y_test):
    total = len(y_train)
    neg_cases = sum(y_train == 0)
    pos_cases = sum(y_train == 1)
    scale_pos_weight = neg_cases / pos_cases
    print(f"\nClass distribution:\nNegative: {neg_cases}\nPositive: {pos_cases}\nWeight: {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba > threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC: {roc_auc:.4f}\nPR-AUC: {pr_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    return roc_auc, pr_auc, cm

def plot_feature_importance(model, X):
    importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.show()

def main():
    df = load_and_preprocess_data('dataSet.csv')
    X, y, scaler, label_encoders = preprocess_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model, y_pred, y_pred_proba = train_xgboost_model(X_train, X_test, y_train, y_test)
    roc_auc, pr_auc, cm = evaluate_model(y_test, y_pred_proba)
    plot_feature_importance(model, X)
    
    # Save artifacts
    joblib.dump(model, 'xgboost_model.pkl')
    joblib.dump(scaler, 'robust_scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    # Export test data for Tableau
    test_data = X_test.copy()
    test_data['isFraud'] = y_test
    test_data['fraud_probability'] = y_pred_proba
    test_data.to_csv('test_data_with_predictions.csv', index=False)
    
    return model, scaler, label_encoders

if __name__ == "__main__":
    model, scaler, label_encoders = main()