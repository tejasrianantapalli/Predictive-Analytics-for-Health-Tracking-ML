import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Sample dataset: User food intake and corresponding calorie intake
data = {
    'Protein(g)': [50, 70, 65, 80, 45, 90, 60],
    'Carbs(g)': [200, 220, 250, 180, 190, 210, 230],
    'Fats(g)': [70, 60, 80, 90, 50, 85, 75],
    'Calories': [2000, 2300, 2500, 2200, 1800, 2600, 2400],
    'Deficiency': [0, 1, 0, 1, 0, 1, 0]  # 0 = No Deficiency, 1 = Deficiency Detected
}

df = pd.DataFrame(data)

# --- Calorie Prediction (Regression) ---
X = df[['Protein(g)', 'Carbs(g)', 'Fats(g)']]
y = df['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
calorie_prediction = model.predict([[60, 200, 70]])  # Example input
print(f"Predicted Calories: {calorie_prediction[0]:.2f}")

# --- Nutrient Deficiency Detection (Classification) ---
X_class = df[['Protein(g)', 'Carbs(g)', 'Fats(g)']]
y_class = df['Deficiency']
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
deficiency_prediction = classifier.predict([[60, 200, 70]])  # Example input
print(f"Nutrient Deficiency Detected: {'Yes' if deficiency_prediction[0] == 1 else 'No'}")

# --- Meal Recommendation (Clustering) ---
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X)
df['Meal Plan Cluster'] = labels
print("\nMeal Plan Clustering:\n", df[['Protein(g)', 'Carbs(g)', 'Fats(g)', 'Meal Plan Cluster']])
