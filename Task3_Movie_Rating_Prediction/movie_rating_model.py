import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# Remove missing values
data = data.dropna(subset=['Year','Votes','Rating'])

# Clean Year column (remove brackets like "(2016)")
data['Year'] = data['Year'].astype(str).str.extract('(\d{4})')
data['Year'] = data['Year'].astype(int)

# Clean Votes column (remove commas)
data['Votes'] = data['Votes'].astype(str).str.replace(',', '')
data['Votes'] = data['Votes'].astype(int)

# Features
X = data[['Year','Votes']]

# Target
y = data['Rating']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Movie Rating Prediction model trained successfully!")