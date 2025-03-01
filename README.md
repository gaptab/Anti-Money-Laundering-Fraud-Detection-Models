# Anti-Money-Laundering-Fraud-Detection-Models

**Data Generation (Simulated Transactions)**

We generate financial transactions for 500 customers with 5,000 transactions in total. Each transaction contains:

Customer ID – Unique identifier for each customer.

Transaction Amount – Random values between $10 and $10,000.

Transaction Type – Can be Deposit, Withdrawal, Transfer, or Online Purchase.

Transaction Country – Simulates global transactions from countries like USA, UK, India, China, UAE, and Germany.

Transaction Time – Categorized into Morning, Afternoon, Evening, and Night.

Suspicious Flag – 98% of transactions are legitimate, and 2% are fraudulent, simulating real-world fraud rates.

This dataset mimics real AML and fraud detection challenges.

**Feature Engineering (Preparing Data for ML Models)**

Categorical Encoding – Converts transaction type, country, and time of day into numerical values.

Feature Scaling – Standardizes numerical values to improve model performance.

Train-Test Split – Splits data into training (70%) and testing (30%) sets to evaluate models.

**Machine Learning Models for Fraud Detection**

We apply three ML models to detect suspicious transactions:

**Decision Tree Classifier (Supervised Model)**

Builds a tree-based decision structure to classify whether a transaction is suspicious or not.

Learns rules like "If the transaction is large and international, then it may be fraud."

Outputs a classification report with accuracy and fraud detection performance.

**Random Forest Classifier (Supervised Model)**

An ensemble method that builds multiple Decision Trees and averages their results.

More robust than a single Decision Tree and reduces overfitting.

Used for high-accuracy fraud detection in banks.

**K-Means Clustering (Unsupervised Model for Anomaly Detection)**

Groups transactions into "high-risk" and "low-risk" clusters.

Detects anomalies without labeled fraud cases, useful for new fraud patterns
