# Importamos las bibliotecas necesarias
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Para guardar y cargar modelos preentrenados

# Cargar el conjunto de datos Iris
iris = datasets.load_iris()
X = iris.data  # Características de las flores
y = iris.target  # Especies de flores

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar las características (opcional, pero mejora el rendimiento de muchos modelos)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar un clasificador SVM (Support Vector Machine)
clf = SVC(kernel='linear')  # Usamos un kernel lineal para simplificar
clf.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo (opcional)
joblib.dump(clf, 'iris_model.pkl')  # Guarda el modelo en un archivo
joblib.dump(scaler, 'scaler.pkl')   # Guarda el escalador (si lo usas)

# Ahora, si quieres cargar el modelo preentrenado y usarlo para predicciones:
# Cargar el modelo y el escalador preentrenados
clf_loaded = joblib.load('iris_model.pkl')
scaler_loaded = joblib.load('scaler.pkl')

# Hacer predicciones sobre el conjunto de prueba usando el modelo cargado
y_pred = clf_loaded.predict(X_test)

# Evaluar el modelo
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
