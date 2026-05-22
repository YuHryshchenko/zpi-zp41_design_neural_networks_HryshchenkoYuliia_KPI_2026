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
#   pip install tensorflow numpy scikit-learn matplotlib

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 1 – Додавання залежностей
# ══════════════════════════════════════════════════════════════════════════════

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
    # Set Kaggle-specific paths
    BASE_DIR = ""
else:
    print("Running locally")
    # Set local paths
    ABSOLUTE_PATH = os.getcwd()
    BASE_DIR = ABSOLUTE_PATH + "/" + CURRENT_LAB + "/"

# 1.2 Завантаження датасету MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# 8. Оцінка моделі на тестових даних
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТочність на тестових даних (Accuracy): {accuracy * 100:.2f}%")

# 9. Візуалізація графіка навчання
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Точність на train')
plt.plot(history.history['val_accuracy'], label='Точність на test')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.title('Зміна точності під час навчання')
plt.grid(True)
plt.savefig('accuracy_history_lab3.png', dpi=300)
plt.show()

# =====================================================================
# ЧАСТИНА 2 – ДОДАТКОВІ ЗАВДАННЯ ЗГІДНО З ВИМОГАМИ ЛАБОРАТОРНОЇ
# =====================================================================

# Отримання передбачень для тестового набору
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# 10. Побудова матриці помилок (Confusion Matrix)
cm = confusion_matrix(y_test_original, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Передбачені класи')
plt.ylabel('Очікувані класи')
plt.title('Матриця плутанини (Confusion Matrix) для 10 класів')
plt.savefig('confusion_matrix_lab3.png', dpi=300)
plt.show()

# 11. Обчислення accuracy, precision, recall та F-Score
print("\n--- Звіт про класифікацію (Precision, Recall, F1-Score) ---")
# scikit-learn автоматично розраховує мікро-, макро- та зважені оцінки
print(classification_report(y_test_original, y_pred_classes, digits=4))

# Візуалізація кількох випадкових передбачень
plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test_images[i], cmap='gray')
    plt.title(f"Передбачено: {y_pred_classes[i]}\nСправжня: {y_test_original[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('predictions_sample_lab3.png', dpi=300)
plt.show()

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
        plt.show()
        
        # Перетворення вектора (reshape) під вхідний шар
        img_array = img_array.reshape(1, 784)
        
        # Передбачення
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        print(f"Передбачена цифра: {predicted_digit} (Впевненість мережі: {confidence:.2f}%)")
        return predicted_digit
        
    except Exception as e:
        print(f"Помилка при обробці зображення: {e}")

# Щоб протестувати свою цифру, збережіть її як "my_digit.png" та розкоментуйте рядок нижче:
# predict_custom_image('my_digit.png', model)