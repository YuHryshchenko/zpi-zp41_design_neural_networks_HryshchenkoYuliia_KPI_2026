# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  ЛАБОРАТОРНА РОБОТА №2                                                                              ║
# ║  Реалізація та дослідження базових архітектур нейронних мереж для моделювання функцій двох змінних  ║
# ║  Предмет: Проектування та реалізація програмних систем з нейронними мережами                        ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════╝
#
# Завдання:
#   За допомогою нейронних мереж змоделювати функцію двох змінних.
#
# Залежності:
#   pip install tensorflow numpy scikit-learn matplotlib

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 1 – Вступний приклад
# Демонструє базову роботу з Python / matplotlib у Kaggle Notebook
# ══════════════════════════════════════════════════════════════════════════════

import os
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Input, Add
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# --- 1.1 Визначення та налаштування середовища ---
CURRENT_LAB = "lab02"

def is_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if is_kaggle():
    print("Running on Kaggle")
    BASE_DIR = ""
else:
    print("Running locally")
    ABSOLUTE_PATH = os.getcwd()
    BASE_DIR = os.path.join(ABSOLUTE_PATH, CURRENT_LAB, "")
    os.makedirs(BASE_DIR, exist_ok=True)

# URL змінено на raw-формат для коректного завантаження бінарних файлів
RAW_BASE_URL = "https://raw.githubusercontent.com/YuHryshchenko/zpi-zp41_design_neural_networks_HryshchenkoYuliia_KPI_2026/main/lab02/"

# Словник для співставлення назви моделі та імені її файлу на GitHub
MODEL_FILENAMES = {
    "FeedForward (10 нейронів)": "ff_10.keras",
    "FeedForward (20 нейронів)": "ff_20.keras",
    "Cascade (20 нейронів)": "cascade_20.keras",
    "Cascade (2x10 нейронів)": "cascade_2x10.keras",
    "Elman (15 нейронів)": "elman_15.keras",
    "Elman (3x5 нейронів)": "elman_3x5.keras"
}

# Функція для генерації навчальних даних
def generate_data(n_samples=1000):
    X = np.random.uniform(0, 10, (n_samples, 2))
    Y = X[:, 0]**2 + X[:, 1]**2
    return X, Y

# Генерація даних
X, Y = generate_data()
X_train, X_test = X[:800], X[800:]
Y_train, Y_test = Y[:800], Y[800:]

# Реалізація Cascade Forward
inputs = Input(shape=(2,))
hidden1 = Dense(20, activation='relu')(inputs)
output = Dense(1)(hidden1)
output_cascade = Dense(1)(inputs)
final_output = Add()([output, output_cascade])
cascade_model_1 = Model(inputs=inputs, outputs=final_output)

inputs_2 = Input(shape=(2,))
hidden1_2 = Dense(10, activation='relu')(inputs_2)
hidden2_2 = Dense(10, activation='relu')(hidden1_2)
output_2 = Dense(1)(hidden2_2)
output_cascade_2 = Dense(1)(inputs_2)
final_output_2 = Add()([output_2, output_cascade_2])
cascade_model_2 = Model(inputs=inputs_2, outputs=final_output_2)

# Перелік неініціалізованих архітектур моделей
models = {
    "FeedForward (10 нейронів)": Sequential([Input(shape=(2,)), Dense(10, activation='relu'), Dense(1)]),
    "FeedForward (20 нейронів)": Sequential([Input(shape=(2,)), Dense(20, activation='relu'), Dense(1)]),
    "Cascade (20 нейронів)": cascade_model_1,
    "Cascade (2x10 нейронів)": cascade_model_2,
    "Elman (15 нейронів)": Sequential([Input(shape=(2, 1)), SimpleRNN(15, activation='relu'), Dense(1)]),
    "Elman (3x5 нейронів)": Sequential([
        Input(shape=(2, 1)),
        SimpleRNN(5, activation='relu', return_sequences=True),
        SimpleRNN(5, activation='relu', return_sequences=True),
        SimpleRNN(5, activation='relu'),
        Dense(1)
    ])
}

# Словники для збереження результатів
histories = {}
errors = {}
predictions = {}
r2_scores = {}

# Основний цикл перевірки, завантаження, скачування або навчання
for name, model in models.items():
    print(f"\nОбробка моделі: {name}...")
    
    filename = MODEL_FILENAMES[name]
    model_path = os.path.join(BASE_DIR, filename) if BASE_DIR else filename
    model_url = RAW_BASE_URL + filename
    
    # Рекурентні мережі (RNN) вимагають зміненої розмірності вхідних даних
    is_rnn = "Elman" in name
    X_tr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)) if is_rnn else X_train
    X_te = X_test.reshape((X_test.shape[0], X_test.shape[1], 1)) if is_rnn else X_test

    loaded_successfully = False

    # 1. Перевірка локального файлу
    if os.path.exists(model_path):
        print(f"[ІНФО] Знайдено локальну модель. Завантаження...")
        try:
            model = tf.keras.models.load_model(model_path)
            loaded_successfully = True
        except Exception as e:
            print(f"[ПОМИЛКА] Не вдалося завантажити локальний файл ({e}).")

    # 2. Спроба скачування з GitHub
    if not loaded_successfully:
        print(f"[ІНФО] Спроба завантаження з GitHub...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("[ІНФО] Успішно завантажено з GitHub!")
            model = tf.keras.models.load_model(model_path)
            loaded_successfully = True
        except Exception as e:
            print(f"[ІНФО] Не вдалося завантажити з GitHub ({e}).")

    # 3. Навчання, якщо файл не знайдено або пошкоджено
    if not loaded_successfully:
        print(f"[ІНФО] Починаємо навчання моделі з нуля...")
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            X_tr, Y_train, 
            epochs=100, 
            batch_size=10, 
            verbose=0, 
            validation_data=(X_te, Y_test)
        )
        histories[name] = history
        print(f"[ІНФО] Збереження навченої моделі у {model_path}...")
        model.save(model_path)

    # 4. Оцінка моделі
    Y_pred = model.predict(X_te, verbose=0).flatten()
    error = np.mean(np.abs((Y_test - Y_pred) / (Y_test + 1e-8)))
    r2 = r2_score(Y_test, Y_pred)
    
    errors[name] = error
    predictions[name] = Y_pred
    r2_scores[name] = r2

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 2 – Візуалізація результатів
# ══════════════════════════════════════════════════════════════════════════════

# Графік MSE малюється лише для моделей, що навчалися у поточній сесії
if histories:
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=name)

    plt.title("Зміна MSE під час навчання")
    plt.xlabel("Епохи")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    
    mse_plot_path = os.path.join(BASE_DIR, "mse_history.png") if BASE_DIR else "mse_history.png"
    plt.savefig(mse_plot_path, bbox_inches="tight")
    plt.show()
else:
    print("\n[ІНФО] Графік MSE пропущено (всі моделі завантажено з файлів).")

# Візуалізація та порівняння передбачень
plt.figure(figsize=(10, 5))
for name, Y_pred in predictions.items():
    plt.scatter(Y_test, Y_pred, label=name, alpha=0.5)

# Ідеальна лінія (де передбачення збігаються з реальністю)
min_val = min(Y_test)
max_val = max(Y_test)
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Ідеальна лінія", linewidth=2)

plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")
plt.legend()
plt.title("Порівняння передбачень різних моделей")
plt.grid(True)

pred_plot_path = os.path.join(BASE_DIR, "predictions_comparison.png") if BASE_DIR else "predictions_comparison.png"
plt.savefig(pred_plot_path, bbox_inches="tight")
plt.show()

# Виведення результатів та помилок
print("\n=== Результати дослідження ===")
for name, error in errors.items():
    print(f"{name}: середня відносна помилка = {error:.4f}, R² = {r2_scores[name]:.4f}")
