import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
data = pd.read_csv("breast_cancer.csv")

# Drop unnecessary columns
for col in ['id', 'Unnamed: 32']:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Encode target column
data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1})

# Select ONLY important features
selected_features = [
    'tumor_radius',
    'cell_texture_irregularity',
    'tumor_boundary_length',
    'tumor_area',
    'tumor_edge_smoothness',
    'tumor_cell_density',
    'tumor_concavity'
]

X = data[selected_features]
y = data['diagnosis']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save files
pickle.dump(model, open("cancer_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(imputer, open("imputer.pkl", "wb"))

print("✅ Model trained successfully with 7 important features!")
