import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Função para treinar e mostrar resultados
def run_models():
    # Carregar o dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)

    # SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)

    # Rede Neural
    model = Sequential([
        Dense(16, input_dim=4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, verbose=0)
    nn_acc = model.evaluate(X_test, y_test, verbose=0)[1]

    # Exibir resultados
    results = (
        f"KNN Accuracy: {knn_acc:.2f}\n"
        f"SVM Accuracy: {svm_acc:.2f}\n"
        f"Neural Network Accuracy: {nn_acc:.2f}"
    )
    messagebox.showinfo("Resultados dos Modelos", results)

# Criar a janela principal
root = tk.Tk()
root.title("Classificação de Flores - Iris Dataset")

# Adicionar botão para rodar os modelos
btn_run = tk.Button(root, text="Executar Modelos", command=run_models, padx=20, pady=10)
btn_run.pack(pady=20)

# Iniciar o loop da interface
root.mainloop()
