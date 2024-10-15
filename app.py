import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('ad_data.csv')

print(data.isnull().sum())
data = data.dropna()

data['Impressions'].replace(0, 1, inplace=True)  # Replace zero Impressions with 1
data['Spend'] = pd.to_numeric(data['Spend'], errors='coerce')

# data = data[data['Spend'] >= 0]

data['CTR'] = (data['Clicks'] / data['Impressions']) * 100
data['Conversion_Rate'] = (data['Conversions'] / data['Clicks']) * 100
data['CPC'] = data['Spend'] / data['Clicks']
data['CPA'] = data['Spend'] / data['Conversions']
data['ROAS'] = data['Revenue'] / data['Spend']
data['CPM'] = (data['Spend'] / data['Impressions']) * 1000

X = data[['Impressions', 'Spend', 'CTR']]  # Features
y = data['Clicks']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Clicks')
plt.ylabel('Predicted Clicks')
plt.title('Actual vs Predicted Clicks')
plt.legend()
plt.tight_layout()
plt.show()

future_data = pd.DataFrame({
    'Impressions': [50000],
    'Spend': [2000],
    'CTR': [3.5]
})
future_clicks = model.predict(future_data)
print(f"Predicted future clicks: {future_clicks[0]}")
