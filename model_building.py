import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("tweets_cleaned.csv")
df = df[df['Sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
df['Sentiment_Label'] = df['Sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Tweet']).toarray()
y = df['Sentiment_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, log_preds))

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, rf_preds))
