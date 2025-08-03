import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

df = pd.read_csv("clv_data.csv")
X = df.drop("CLV", axis=1)
y = df["CLV"]

cat_features = ["Gender", "ProductCategory"]
num_features = [col for col in X.columns if col not in cat_features]


preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first"), cat_features)
])

model = Pipeline([
    ("pre", preprocessor),
    ("reg", ElasticNet(alpha=1.0, l1_ratio=0.5))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "elasticnet_clv_model.pkl")
print(" Model trained and saved!")
