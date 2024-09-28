import numpy as np
import matplotlib.pyplot as plt
# mlxtend у нас на машине конечно же не установлен, поэтому начинаем с установки
# pip install mlxtend
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_iris, load_diabetes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Описываем свой класс для метода К-ближайших соседей на основании расчёта Эвклидового расстояния
class KNearestNeighbors:
    def __init__(self, n_neighbors=5, regression=False):
        self.n_neighbors = n_neighbors
        self.regression = regression

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def _euclidean_distances(self, x_test_i):
        return np.sqrt(np.sum((self.X_train - x_test_i) ** 2, axis=1))

    def _make_prediction(self, x_test_i):
        distances = self._euclidean_distances(x_test_i)   # distances to all neighbors
        k_nearest_indexes = np.argsort(distances)[:self.n_neighbors]
        targets = self.y_train[k_nearest_indexes]   # k-nearest neighbors target values

        return np.mean(targets) if self.regression else np.bincount(targets).argmax()

    def predict(self, X_test):
        return np.array([self._make_prediction(x) for x in X_test])

# Описываем функцию для отрисовки графика поверхности решений
def decision_boundary_plot(X, y, X_train, y_train, clf, feature_indexes, title=None):
    feature1_name, feature2_name = X.columns[feature_indexes]
    X_feature_columns = X.values[:, feature_indexes]
    X_train_feature_columns = X_train[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train)
    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)
    plt.show()

# Сначала разбираемся с Ирисами Фишера
X1, y1 = load_iris(return_X_y=True, as_frame=True)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1.values, y1.values, random_state=0)
print(X1, y1, sep='\n')

# Берём наш кассификатор, для которого мы писали класс выше
knn_clf = KNearestNeighbors()
# Обучаем модель
knn_clf.fit(X1_train, y1_train)
# Получаем прогнозы
knn_clf_pred_res = knn_clf.predict(X1_test)
# Считаем метрику ACC
knn_clf_accuracy = accuracy_score(y1_test, knn_clf_pred_res)
print(f'KNN classifier accuracy: {knn_clf_accuracy:}')
# KNN classifier accuracy: 0.9736842105263158 — достаточно точно
# print(knn_clf_pred_res)

# Теперь проверим тоже самое, но уже со встроенным классификатором SKLearn
sk_knn_clf = KNeighborsClassifier()
sk_knn_clf.fit(X1_train, y1_train)
sk_knn_clf_pred_res = sk_knn_clf.predict(X1_test)
sk_knn_clf_accuracy = accuracy_score(y1_test, sk_knn_clf_pred_res)

print(f'sk KNN classifier accuracy: {sk_knn_clf_accuracy:}')
# sk KNN classifier accuracy: 0.9736842105263158 — то же самое.
print(sk_knn_clf_pred_res)

# Построим график пространства решений по двум параметрам: petal width и petal length
# Можно по другим парам, например sepal width и sepal length, но график будет визуально менее понятным
feature_indexes = [2, 3]
title1 = 'График пространства решений (predict surface) для KNeighborsClassifier'
decision_boundary_plot(X1, y1, X1_train, y1_train, sk_knn_clf, feature_indexes, title1)

# Теперь поработаем с диабетиками
# Загрузим данные
X2, y2 = load_diabetes(return_X_y=True, as_frame=True)
# Разобъем на обучающую и тестовую выборку. Так как не передаём test_size соотношение будет 75 / 25
X2_train, X2_test, y2_train, y2_test = train_test_split(X2.values, y2.values, random_state=0)
print(X2, y2, sep='\n')

# Берём наш регрессов, для которого мы писали класс выше (для этого передаём аргумент regression = True)
knn_reg = KNearestNeighbors(regression=True)
# Обучаем модель
knn_reg.fit(X2_train, y2_train)
# Получаем прогнозы
knn_reg_pred_res = knn_reg.predict(X2_test)
# Считаем Коэффициент детерминации R^2
knn_reg_r2 = r2_score(y2_test, knn_reg_pred_res)

print(f'KNN regressor R2 score: {knn_reg_r2}')
#KNN regressor R2 score: 0.18912404854026388
# Хотелось бы единичку, но нет: ~0.18 — как вилами по воде
print(knn_reg_pred_res)

# Теперь проверим тоже самое, но уже со встроенным классификатором SKLearn
sk_knn_reg = KNeighborsRegressor()
sk_knn_reg.fit(X2_train, y2_train)
sk_knn_reg_pred_res = sk_knn_reg.predict(X2_test)
sk_knn_reg_r2 = r2_score(y2_test, sk_knn_reg_pred_res)

print(f'sk KNN regressor R2 score: {sk_knn_reg_r2}')
# sk KNN regressor R2 score: 0.18912404854026388 — то же самое.
print(sk_knn_reg_pred_res)

# Ну и построим график регрессора, по X - будет фактические значения прогрессирование заболевания, а по Y - то что напредсказывала модель
title2 = 'График прогнозов для KNeighborsRegressor'
plt.figure(figsize=(10, 6))
# Добавим точки на график
plt.scatter(y2_test, sk_knn_reg_pred_res, color='blue', label='Прогноз vs Факт')
# Добавим идеальную прямую обучения — чем ближе к ней точки, тем точне прогноз
plt.plot([min(y2_test), max(y2_test)], [min(y2_test), max(y2_test)], color='red', linewidth=2, label='Идеальная прямая обучения')
plt.title('KNN Regression: Прогноз vs Факт')
plt.xlabel('Фактическое прогрессирование заболевания')
plt.ylabel('Предсказанное прогрессирование заболевания')
plt.legend()
plt.show()
# Парочку прогнозов легло почти идеально, но в целом так себе