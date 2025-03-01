import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to generate dummy transactional data
def generate_dummy_data(n=5000):
    np.random.seed(42)
    random.seed(42)
    
    customer_ids = np.arange(1, 501)  # 500 unique customers
    data = []

    for _ in range(n):
        cust_id = np.random.choice(customer_ids)
        txn_amount = round(np.random.uniform(10, 10000), 2)
        txn_type = np.random.choice(["Deposit", "Withdrawal", "Transfer", "Online Purchase"])
        txn_country = np.random.choice(["USA", "UK", "India", "China", "UAE", "Germany"])
        txn_time = np.random.choice(["Morning", "Afternoon", "Evening", "Night"])
        is_suspicious = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraudulent transactions

        data.append([cust_id, txn_amount, txn_type, txn_country, txn_time, is_suspicious])

    df = pd.DataFrame(data, columns=["CustomerID", "Amount", "TxnType", "Country", "TimeOfDay", "Suspicious"])
    
    return df

# Generate the dataset
df = generate_dummy_data()
print(df.head())

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=["TxnType", "Country", "TimeOfDay"], drop_first=True)

# Split dataset into features and target
X = df_encoded.drop(columns=["Suspicious", "CustomerID"])  # Features
y = df_encoded["Suspicious"]  # Target variable

# Standardize numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Performance:")
print(classification_report(y_test, y_pred_dt))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))

# Train K-Means to identify anomalies
kmeans = KMeans(n_clusters=2, random_state=42)
df_encoded["Cluster"] = kmeans.fit_predict(X_scaled)

# Assign cluster labels as suspicious or non-suspicious
cluster_suspicion_map = {0: "Low-Risk", 1: "High-Risk"}
df_encoded["ClusterLabel"] = df_encoded["Cluster"].map(cluster_suspicion_map)

print(df_encoded[["Cluster", "ClusterLabel"]].value_counts())
