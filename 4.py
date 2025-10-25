import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

X_glucose = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
y_hemoglobin = np.array([130, 140, 147, 150, 151, 149, 145, 138, 130, 120])


X = X_glucose.reshape(-1, 1)
y = y_hemoglobin


poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)


model = LinearRegression()
model.fit(X_poly, y)


a = model.intercept_  
b1 = model.coef_[0]   
b2 = model.coef_[1]   


y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)



print("="*50)
print("     АНАЛІЗ ОТРИМАНОЇ РЕГРЕСІЙНОЇ МОДЕЛІ")
print("="*50)


print("\n1. Функція Регресії (Рівняння)")
print(f"Модель: Y = a + b1*X + b2*X^2")
print(f"Обчислені параметри:")
print(f"  a (Вільний член) = {a:.4f}")
print(f"  b1 (при X)       = {b1:.4f}")
print(f"  b2 (при X^2)     = {b2:.4f}")
print("\nФІНАЛЬНЕ МАТЕМАТИЧНЕ РІВНЯННЯ:")
print(f"  Гемоглобін = {a:.4f} + {b1:.4f} * (Глюкоза) {b2:.4f} * (Глюкоза)^2")
print("-" * 50)



print("\n2. Вплив Змінних (Аналіз)")
print(f"Коефіцієнт b2 (при X^2) дорівнює {b2:.4f}.")
if b2 < 0:
    print("Оскільки b2 ВІД'ЄМНИЙ, форма зв'язку - це ПАРАБОЛА,")
    print("гілки якої спрямовані ВНИЗ (∩). ")
    print("Це означає, що вплив глюкози на гемоглобін нелінійний:")
    print("  1. Спочатку (при низькій глюкозі) її зростання ПОЗИТИВНО впливає на гемоглобін (він росте).")
    print("  2. Після досягнення точки піку, подальше зростання глюкози НЕГАТИВНО впливає на гемоглобін (він знижується).")
elif b2 > 0:
    print("Оскільки b2 ДОДАТНІЙ, форма зв'язку - це ПАРАБОЛА,")
    print("гілки якої спрямовані ВГОРУ (∪).")
    print("Вплив глюкози: гемоглобін спочатку знижується, досягає мінімуму, а потім починає зростати.")
else:
    print("Оскільки b2 = 0, зв'язок є лінійним, а не параболічним.")
print("-" * 50)



print("\n 3. Оцінка Якості Моделі")
print(f"Коефіцієнт детермінації (R^2) = {r2:.4f}")
print(f"Це означає, що побудована модель пояснює {r2*100:.2f}%")
print("всієї мінливості (варіації) рівня гемоглобіну.")
if r2 > 0.95:
    print("Оцінка якості: ВІДМІННА. Модель дуже добре описує дані.")
elif r2 > 0.8:
    print("Оцінка якості: ДУЖЕ ДОБРА. Модель добре описує дані.")
elif r2 > 0.6:
    print("Оцінка якості: ЗАДОВІЛЬНА. Модель пояснює більшу частину даних.")
else:
    print("Оцінка якості: СЛАБКА. Модель погано описує дані.")
print("="*50)