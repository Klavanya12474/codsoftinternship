import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Fill missing age values
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Convert Sex column to numbers
encoder = LabelEncoder()
data['Sex'] = encoder.fit_transform(data['Sex'])

# Features
X = data[['Pclass','Sex','Age','SibSp','Parch','Fare']]

# Target
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)