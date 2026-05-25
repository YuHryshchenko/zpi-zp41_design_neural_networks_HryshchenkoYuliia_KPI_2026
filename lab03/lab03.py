# ╔═════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  ЛАБОРАТОРНА РОБОТА №3                                                                              ║
# ║  Нейронної мережі прямого розповсюдження для розпізнавання зображення                               ║
# ║  Предмет: Проектування та реалізація програмних систем з нейронними мережами                        ║
# ╚═════════════════════════════════════════════════════════════════════════════════════════════════════╝
#
# Завдання:
#   Написати програму що реалізує нейронну мережу прямого розповсюдження для розпізнавання рукописних цифр.
#
# Залежності:
#   pip install tensorflow numpy scikit-learn matplotlib pillow seaborn

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 1 – Додавання залежностей
# ══════════════════════════════════════════════════════════════════════════════

import os
import urllib.request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image, ImageOps

# --- 1.1 Визначення та налаштування середовища ---
CURRENT_LAB = "lab03"

def is_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if is_kaggle():
    print("Running on Kaggle")
    BASE_DIR = ""
else:
    print("Running locally")
    ABSOLUTE_PATH = os.getcwd()
    BASE_DIR = ABSOLUTE_PATH + "/" + CURRENT_LAB + "/"
    # Створення локальної директорії перед збереженням файлів
    os.makedirs(BASE_DIR, exist_ok=True)

EPOCHS_NUM = 10

# --- 1.2 Завантаження датасету MNIST з локальним кешуванням ---
DATA_FILENAME = "mnist.npz"
data_path = os.path.join(BASE_DIR, DATA_FILENAME) if BASE_DIR else DATA_FILENAME

if os.path.exists(data_path):
    print(f"[ІНФО] Знайдено локальний датасет: {data_path}. Завантаження...")
    with np.load(data_path) as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
else:
    print("[ІНФО] Локального датасету не знайдено. Завантаження через Інтернет...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"[ІНФО] Збереження датасету локально у {data_path}...")
    np.savez(data_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# Збереження оригінальних тестових міток та зображень для подальшої візуалізації
y_test_original = y_test.copy()
x_test_images = x_test.copy()

# 2. Нормалізація зображень (приведення пікселів до діапазону 0..1)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 3. Перетворення 28x28 зображень у вектори довжиною 784
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 4. Перетворення міток у формат one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# ── Логіка завантаження, скачування або навчання моделі ─────────────────────────
MODEL_FILENAME = "mnist_model.keras"
model_path = os.path.join(BASE_DIR, MODEL_FILENAME) if BASE_DIR else MODEL_FILENAME
RAW_MODEL_URL = f"https://raw.githubusercontent.com/YuHryshchenko/zpi-zp41_design_neural_networks_HryshchenkoYuliia_KPI_2026/main/lab03/{MODEL_FILENAME}"

history = None
loaded_successfully = False

# 1. Перевірка локального файлу
if os.path.exists(model_path):
    print(f"[ІНФО] Знайдено локальну модель: {model_path}. Завантаження...")
    try:
        model = keras.models.load_model(model_path)
        loaded_successfully = True
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося завантажити локальний файл ({e}).")

# 2. Спроба скачування з GitHub
if not loaded_successfully:
    print(f"[ІНФО] Локальну модель не знайдено. Спроба завантаження з GitHub...")
    try:
        urllib.request.urlretrieve(RAW_MODEL_URL, model_path)
        print("[ІНФО] Модель успішно завантажено з GitHub!")
        model = keras.models.load_model(model_path)
        loaded_successfully = True
    except Exception as e:
        print(f"[ІНФО] Не вдалося завантажити модель з GitHub ({e}).")

# 3. Навчання, якщо файл відсутній або сталася помилка
if not loaded_successfully:
    print("[ІНФО] Починаємо створення та навчання моделі з нуля...")
    
    # 5. Створення моделі нейромережі прямого розповсюдження
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    # 6. Компіляція моделі
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # 7. Тренування моделі
    print("Початок навчання моделі...")
    history = model.fit(x_train, y_train, epochs=EPOCHS_NUM, batch_size=32, validation_data=(x_test, y_test))
    
    print(f"[ІНФО] Збереження навченої моделі у {model_path}...")
    model.save(model_path)

# 8. Оцінка моделі на тестових даних
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТочність на тестових даних (Accuracy): {accuracy * 100:.2f}%")

# 9. Візуалізація графіка навчання (лише якщо модель тренувалася в цій сесії)
if history is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Точність на train')
    plt.plot(history.history['val_accuracy'], label='Точність на test')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.legend()
    plt.title('Зміна точності під час навчання')
    plt.grid(True)
    
    acc_plot_path = os.path.join(BASE_DIR, 'accuracy_history_lab3.png') if BASE_DIR else 'accuracy_history_lab3.png'
    plt.savefig(acc_plot_path, dpi=300)
    plt.show()
    print(f"[ІНФО] Графік точності збережено у: {acc_plot_path}")
else:
    print("[ІНФО] Графік історії навчання пропущено (модель завантажено з готового файлу).")

# =====================================================================
# ЧАСТИНА 2 – ДОДАТКОВІ ЗАВДАННЯ ЗГІДНО З ВИМОГАМИ ЛАБОРАТОРНОЇ
# =====================================================================

# Отримання передбачень для тестового набору
y_pred_probs = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# 10. Побудова матриці помилок (Confusion Matrix)
cm = confusion_matrix(y_test_original, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Передбачені класи')
plt.ylabel('Очікувані класи')
plt.title('Матриця плутанини (Confusion Matrix) для 10 класів')

cm_plot_path = os.path.join(BASE_DIR, 'confusion_matrix_lab3.png') if BASE_DIR else 'confusion_matrix_lab3.png'
plt.savefig(cm_plot_path, dpi=300)
plt.show()
print(f"[ІНФО] Матрицю плутанини збережено у: {cm_plot_path}")

# 11. Обчислення accuracy, precision, recall та F-Score
print("\n--- Звіт про класифікацію (Precision, Recall, F1-Score) ---")
print(classification_report(y_test_original, y_pred_classes, digits=4))

# Візуалізація кількох випадкових передбачень
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test_images[i], cmap='gray')
    plt.title(f"Передбачено: {y_pred_classes[i]}\nСправжня: {y_test_original[i]}")
    plt.axis('off')
plt.tight_layout()

sample_plot_path = os.path.join(BASE_DIR, 'predictions_sample_lab3.png') if BASE_DIR else 'predictions_sample_lab3.png'
plt.savefig(sample_plot_path, dpi=300)
plt.show()
print(f"[ІНФО] Графік прикладів передбачень збережено у: {sample_plot_path}")

# 12. Функція розпізнавання власноруч написаних цифр
def predict_custom_image(image_path, model):
    """
    Функція приймає шлях до зображення, обробляє його відповідно до формату MNIST
    (градієнт сірого, 28x28 пікселів, інверсія) та повертає передбачення.
    """
    try:
        # Завантаження зображення у відтінках сірого (L)
        img = Image.open(image_path).convert('L')
        
        # Увага: MNIST використовує світлі цифри на чорному тлі. 
        # Якщо ваша картинка - чорна цифра на білому папері, розкоментуйте наступний рядок:
        # img = ImageOps.invert(img)
        
        # Зміна розміру до потрібних 28x28 пікселів
        img = img.resize((28, 28))
        
        # Перетворення у масив та нормалізація (0-1)
        img_array = np.array(img).astype("float32") / 255.0
        
        # Візуалізація того, що саме «побачить» нейромережа
        plt.figure(figsize=(3, 3))
        plt.imshow(img_array, cmap='gray')
        plt.title("Оброблене зображення")
        plt.axis('off')
        
        custom_img_plot_path = os.path.join(BASE_DIR, 'custom_image_processed.png') if BASE_DIR else 'custom_image_processed.png'
        plt.savefig(custom_img_plot_path, dpi=300)
        plt.show()
        print(f"[ІНФО] Графік обробленого користувацького зображення збережено у: {custom_img_plot_path}")
        
        # Перетворення вектора (reshape) під вхідний шар
        img_array = img_array.reshape(1, 784)
        
        # Передбачення
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        print(f"Передбачена цифра: {predicted_digit} (Впевненість мережі: {confidence:.2f}%)")
        return predicted_digit
        
    except Exception as e:
        print(f"Помилка при обробці зображення: {e}")

# Щоб протестувати свою цифру, збережіть її як "my_digit.png" та розкоментуйте рядок нижче:
# predict_custom_image('my_digit.png', model)