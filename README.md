Завдання 1.

Розробити алгоритм і програму для побудови дерев рішень. Виконати опрацювання набору тестових даних з метою синтезу дерева рішень. Проаналізувати побудоване дерево рішень.

Побудова дерева рішень на Python
1. Опис завдання та методологія

Мета роботи:
Розробити програму на Python для побудови (синтезу) дерева рішень, виконати опрацювання тестового набору даних та провести детальний аналіз отриманої моделі класифікації.

Використані інструменти

Мова програмування: Python 3

Основна бібліотека: scikit-learn – для побудови дерева рішень, розділення даних та оцінки моделі

Допоміжні бібліотеки:

matplotlib – для візуалізації дерева

pandas, numpy – для роботи з даними

Хід виконання роботи

1. Завантаження даних:
Використано класичний набір даних Iris (load_iris з scikit-learn), який містить 150 зразків трьох класів – setosa, versicolor, virginica – та 4 ознаки (довжина і ширина пелюсток та чашолистків).

2. Розділення вибірки:

Тренувальний набір: 105 зразків (70%)

Тестовий набір: 45 зразків (30%)

3. Синтез дерева рішень:
Використано модель:

DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)

    Критерій розділення – ентропія (Information Gain)

    Навчання моделі виконано методом .fit(X_train, y_train)

4. Аналіз моделі:

    Отримано текстові правила за допомогою export_text()

    Побудовано візуальне представлення дерева за допомогою plot_tree()

    Проведено оцінку ефективності:

        Точність (Accuracy)

        Матриця помилок (Confusion Matrix)

        Метрики Precision, Recall, F1-score

2. Код програми
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn import metrics


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Розмір тренувального набору: {X_train.shape[0]} зразків")
print(f"Розмір тестового набору: {X_test.shape[0]} зразків")
print("-" * 30)


clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)


clf.fit(X_train, y_train)

print("Дерево рішень успішно побудовано.")
print("-" * 30)


print("Текстове представлення дерева рішень:")
text_representation = export_text(clf, feature_names=feature_names)
print(text_representation)
print("-" * 30)


print("Візуалізація дерева рішень (дивіться у окремому вікні)...")
plt.figure(figsize=(15, 10))
plot_tree(clf, 
          feature_names=feature_names, 
          class_names=target_names, 
          filled=True, 
          rounded=True) 

plt.show()


y_pred = clf.predict(X_test)


accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Точність (Accuracy) на тестових даних: {accuracy:.2f}")


print("\nМатриця помилок:")
print(metrics.confusion_matrix(y_test, y_pred))


print("\nДетальний звіт по класифікації:")
print(metrics.classification_report(y_test, y_pred, target_names=target_names))



3. Результати виконання роботи
<img width="1206" height="821" alt="изображение" src="https://github.com/user-attachments/assets/b1661950-d6ef-40f3-8fd5-faf4c30afb8f" />

Рисунок 1. Графічна візуалізація дерева рішень
На рисунку показано структуру дерева, побудованого на 105 зразках тренувальної вибірки. Кожен вузол містить:

Rule – правило розділення, обране за критерієм приросту інформації

entropy – показник "нечистоти" вузла (0.0 – чистий вузол)

samples – кількість зразків у вузлі

value – розподіл зразків по класах

class – домінуючий клас у вузлі







<img width="424" height="692" alt="изображение" src="https://github.com/user-attachments/assets/3b9b3fa1-fca9-4feb-b512-ceb4cd436e00" />


Рисунок 2. Текстове представлення дерева рішень та оцінка ефективності

Текстове представлення: правила "ЯКЩО-ТОДІ", що відображають логіку моделі

Метрики ефективності:

Accuracy: ~0.96 (96%)

Confusion Matrix: показує лише одну неправильну класифікацію з 45 тестових зразків

Classification Report: високі показники precision, recall та F1-score для всіх класів

