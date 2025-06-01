import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

data = pd.read_csv("customer.csv")

# uncomment the line bellow to use nomic-embed-text-v2-moe
# embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True, device="cuda")

# comment the line below to use nomic-embed-text-v2-moe
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

X = embed_model.encode(data["utterance"].tolist()) 
y = data["category"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_log))

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf))
