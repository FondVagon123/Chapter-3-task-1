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



