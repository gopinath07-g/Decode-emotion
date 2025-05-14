import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("tweets.csv")
df.dropna(subset=['Tweet'], inplace=True)
df['User'] = df['User'].fillna('Unknown')
df.drop_duplicates(inplace=True)

if 'Likes' in df.columns:
    upper_limit = df['Likes'].quantile(0.99)
    df['Likes'] = np.where(df['Likes'] > upper_limit, upper_limit, df['Likes'])

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df['Tweet'] = df['Tweet'].astype(str)
df['User'] = df['User'].astype(str)

le = LabelEncoder()
df['User_Label'] = le.fit_transform(df['User'])

if 'Likes' in df.columns:
    scaler = StandardScaler()
    df['Likes_Standardized'] = scaler.fit_transform(df[['Likes']])

df.to_csv("tweets_cleaned.csv", index=False)
print("Preprocessing complete.")
