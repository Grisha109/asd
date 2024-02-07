import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Загрузка исторических данных
data = pd.read_csv('historical_data.csv')

# Создание задержки на цену BTCUSDT, если это необходимо
data['BTCUSDT_lagged'] = data['BTCUSDT'].shift(1)

# Определение зависимой и независимой переменных
X = data['BTCUSDT_lagged'].values.reshape(-1, 1)
y = data['ETHUSDT'].values.reshape(-1, 1)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели на обучающих данных
model.fit(X_train, y_train)

# Вывод коэффициентов модели
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_)

# Применение модели на тестовых данных
y_pred = model.predict(X_test)

# Оценка качества модели с помощью R2
r2 = r2_score(y_test, y_pred)
print('R2 Score:', r2)

# Построение графика фактической и предсказанной цены ETHUSDT
plt.plot(range(len(y_test)), y_test, label='Actual')
plt.plot(range(len(y_test)), y_pred, label='Predicted')
plt.legend()
plt.xlabel('Time')
plt.ylabel('ETHUSDT Price')
plt.show()
