# Cначала нужно постаить sklearn
# На pip install sklearn среда ругается, поэтому лезем в документацию и смотрим нормальное имя модуля в репозитории: https://github.com/scikit-learn/sklearn-pypi-package
# pip install scikit-learn
# Импорт всех нужных библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Тут пришлось добавить ConfusionMatrixDisplay, а то матрица ошибок смотрится в консоли грустно
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# А чтобы в матрице ошибок были подписи для категорийных признков, нужен будет ещё и Encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#Загружаем данные из CSV набор данных с ирисами Фишера
data = pd.read_csv('./Iris.csv')

# Проверяем, что что-то прочиталось 
#print(data.head(2))

# Убираем лишние столбцы
data.drop('Id', axis=1, inplace=True)

# Выделяем все признкаи
X = data.iloc[:,:-1].values
# И метки
y = data['Species']

encoder = LabelEncoder()
encoder.fit(y)
print(encoder.classes_)

# Сплитом делим 80 / 20 данные на учебные с суффиксом _train и тестовые с суффиксом _test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=25)

# На всякий пожарный выводим учебную часть выборки
#print(X_train)
#print(y_train)

# Дальше берем два разных метода классификации: Метод опорных векторов (SVC) и Метод К-ближайших соседей (KNN).
SVC_model = SVC()
KNN_model = KNeighborsClassifier(n_neighbors=5)

#Обучаем обе модели на учебных наборах
SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)

# Теперь, когда модели обучились, проверяем точность прогнозова на тестовых данных
SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)

# Считаем ACC
print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))

# Матрицы неточностей
SVC_confusion_matrix = confusion_matrix(SVC_prediction, y_test)
KNN_confusion_matrix = confusion_matrix(KNN_prediction, y_test)

disp = ConfusionMatrixDisplay(confusion_matrix=SVC_confusion_matrix, display_labels=encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix SVC')
plt.show()

# Тут видно, что модель SVC определила один Iris virginica как Iris-versicolor, а остальные записи были определены верно

disp = ConfusionMatrixDisplay(confusion_matrix=KNN_confusion_matrix, display_labels=encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix KNN')
plt.show()

# А тут видно, что модель KNN определила два Iris virginica как Iris-versicolor, а остальные записи были определены верно, что менее точно, чем у SVC

# Отчёты о классификации
print(classification_report(SVC_prediction, y_test))
print(classification_report(KNN_prediction, y_test))

# В целом, кажется, что модель классификации на основе метода опорных векторов в случае с Ирисами Фишера работает лучше.