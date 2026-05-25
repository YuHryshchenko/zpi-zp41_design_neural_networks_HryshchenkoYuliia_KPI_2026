# ============================================================
# ЛАБОРАТОРНА РОБОТА №4
# Реалізація та дослідження згорткової нейронної мережі AlexNet
# для класифікації зображень (набір даних Animals10)
# ============================================================

# ============================================================
# КРОК 1. Встановлення та завантаження датасету
# ============================================================

import os
import time
import random
import urllib.request
import kaggle
# from kaggle.api.kaggle_api_extended import KaggleApi as api
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import (
    Conv2D as C2D, MaxPooling2D as MP2D,
    Flatten as FL, Dense as D, Dropout as DR,
    BatchNormalization as BN, Input as INP,
    GlobalAveragePooling2D as GAP
)

# --- 1.1 Визначення та налаштування середовища ---
CURRENT_LAB = "lab04"

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
    os.makedirs(BASE_DIR, exist_ok=True)

# Встановлюємо kaggle
# ! pip install kaggle

# Монтуємо Google Drive
# from google.colab import drive
# drive.mount("/content/drive")
# Mounted at /content/drive

# Робимо папку та копіюємо в неї файл з API key
# ! mkdir ~/.kaggle
# ! cp /content/drive/MyDrive/kaggle/kaggle.json ~/.kaggle/kaggle.json

# Обмежуємо доступ до файлу
# ! chmod 600 ~/.kaggle/kaggle.json

# Завантажуємо датасет
# kaggle datasets download alessiocorrado99/animals10

# Розпаковуємо архів
# unzip animals10.zip

#kaggle.api.authenticate()

# Download and unzip
#kaggle.api.dataset_download_files(dataset="alessiocorrado99/animals10", path='.', unzip=True)

# ============================================================
# КРОК 2. Розділяємо дані на навчальні та тестові
# ============================================================

NUM_EPOCHS=10

img_size = (227, 227)
batch_size = 32
extract_path = BASE_DIR + "/raw-img"

datagen = ImageDataGenerator(
    rescale=1.0 / 255,       # Нормалізація
    validation_split=0.2     # 20% даних під валідацію
)

train_generator = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Found 20947 images belonging to 10 classes.
# Found 5232 images belonging to 10 classes.


# ============================================================
# КРОК 3 & КРОК 4. Управління архітектурою та Навчання (AlexNet)
# ============================================================

MODEL_FILENAME = "alexnet_model.keras"
model_path = os.path.join(BASE_DIR, MODEL_FILENAME) if BASE_DIR else MODEL_FILENAME
RAW_MODEL_URL = f"https://raw.githubusercontent.com/YuHryshchenko/zpi-zp41_design_neural_networks_HryshchenkoYuliia_KPI_2026/main/lab04/{MODEL_FILENAME}"

history = None
loaded_successfully = False

# 1. Перевірка локального файлу
if os.path.exists(model_path):
    print(f"[ІНФО] Знайдено локальну модель AlexNet: {model_path}. Завантаження...")
    try:
        model = tf.keras.models.load_model(model_path)
        loaded_successfully = True
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося завантажити локальний файл ({e}).")

# 2. Спроба скачування з GitHub
if not loaded_successfully:
    print(f"[ІНФО] Локальну модель не знайдено. Спроба завантаження з GitHub...")
    try:
        urllib.request.urlretrieve(RAW_MODEL_URL, model_path)
        print("[ІНФО] Модель AlexNet успішно завантажено з GitHub!")
        model = tf.keras.models.load_model(model_path)
        loaded_successfully = True
    except Exception as e:
        print(f"[ІНФО] Не вдалося завантажити оригінальну модель з GitHub ({e}).")

# 3. Навчання, якщо файл відсутній або сталася помилка
if not loaded_successfully:
    print("[ІНФО] Починаємо створення та навчання оригінальної моделі з нуля...")
    model = Sequential([
        Input(shape=(227, 227, 3)),

        # Блок 1 — перший згортковий шар
        Conv2D(96, (11, 11), strides=4, activation='relu'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        # Блок 2 — другий згортковий шар
        Conv2D(256, (5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        # Блок 3 — три послідовних згорткових шари
        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        # Повнозв'язні шари
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')   # 10 класів
    ])

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        epochs=NUM_EPOCHS,
        validation_data=val_generator
    )
    
    print(f"[ІНФО] Збереження навченої моделі у {model_path}...")
    model.save(model_path)

print("\n=== Зведення оригінальної моделі ===")
print(model.summary())

# ============================================================
# КРОК 5. Виводимо точність та втрати на валідації
# ============================================================

loss, accuracy = model.evaluate(val_generator)
print(f"Точність на валідації: {accuracy:.4f}")
print(f"Втрати (Loss): {loss:.4f}")


# ============================================================
# КРОК 6. Будуємо графіки точності та втрати
# ============================================================

if history is not None:
    acc       = history.history['accuracy']
    val_acc   = history.history['val_accuracy']
    loss_hist = history.history['loss']
    val_loss  = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc,     label='Точність на тренуванні')
    plt.plot(epochs_range, val_acc, label='Точність на валідації')
    plt.legend()
    plt.title('Графік точності')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss_hist, label='Втрати на тренуванні')
    plt.plot(epochs_range, val_loss,  label='Втрати на валідації')
    plt.legend()
    plt.title('Графік втрат')

    plt.tight_layout()
    metrics_plot_path = os.path.join(BASE_DIR, 'alexnet_training_metrics.png') if BASE_DIR else 'alexnet_training_metrics.png'
    plt.savefig(metrics_plot_path, dpi=300)
    plt.show()
    print(f"[ІНФО] Метрики навчання збережено у: {metrics_plot_path}")
else:
    print("[ІНФО] Графік історії навчання пропущено (оригінальну модель завантажено з файлу).")


# ============================================================
# КРОК 7. Будуємо матрицю помилок та обраховуємо метрики
#         (accuracy, precision, recall, F-Score)
# ============================================================

# Отримуємо передбачення для всієї валідаційної вибірки
val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

class_names = list(val_generator.class_indices.keys())

# --- Матриця помилок ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel('Передбачений клас')
plt.ylabel('Справжній клас')
plt.title('Матриця помилок (Confusion Matrix)')
plt.tight_layout()

cm_plot_path = os.path.join(BASE_DIR, 'alexnet_confusion_matrix.png') if BASE_DIR else 'alexnet_confusion_matrix.png'
plt.savefig(cm_plot_path, dpi=300)
plt.show()
print(f"[ІНФО] Матрицю помилок збережено у: {cm_plot_path}")

# --- Метрики ---
acc_score  = accuracy_score(y_true, y_pred)
prec_score = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec_score  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1         = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"\nAccuracy  : {acc_score:.4f}")
print(f"Precision : {prec_score:.4f}")
print(f"Recall    : {rec_score:.4f}")
print(f"F-Score   : {f1:.4f}")

print("\nДетальний звіт по класах:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


# ============================================================
# КРОК 8. Перевіряємо розпізнавання на випадковому зображенні
#         з набору даних
# ============================================================

class_names_list = list(train_generator.class_indices.keys())

random_class      = random.choice(class_names_list)
random_image_path = random.choice(os.listdir(f"{extract_path}/{random_class}"))
img_path          = f"{extract_path}/{random_class}/{random_image_path}"

img       = image.load_img(img_path, target_size=(227, 227))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions     = model.predict(img_array)
print(predictions)
predicted_class = class_names_list[np.argmax(predictions)]

plt.imshow(img)
plt.axis('off')
plt.title(f"Очікуваний: {random_class}\nПередбачений: {predicted_class}")

rand_sample_path = os.path.join(BASE_DIR, 'alexnet_random_prediction.png') if BASE_DIR else 'alexnet_random_prediction.png'
plt.savefig(rand_sample_path, dpi=300)
plt.show()
print(f"[ІНФО] Тестове випадкове передбачення збережено у: {rand_sample_path}")


# ============================================================
# КРОК 9. Тестування моделі на зображеннях НЕ з набору даних
#         (написати функцію розпізнавання зображень)
# ============================================================

def predict_image(img_path: str, model, class_names: list) -> str:
    """
    Розпізнає клас зображення за допомогою навченої моделі AlexNet.
    """
    img       = image.load_img(img_path, target_size=(227, 227))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds           = model.predict(img_array, verbose=0)
    predicted_idx   = np.argmax(preds)
    predicted_class = class_names[predicted_idx]
    confidence      = preds[0][predicted_idx] * 100

    # Відображення результату
    plt.imshow(img)
    plt.axis('off')
    plt.title(
        f"Передбачений клас: {predicted_class}\n"
        f"Впевненість: {confidence:.2f}%"
    )
    
    custom_pred_path = os.path.join(BASE_DIR, 'alexnet_custom_prediction.png') if BASE_DIR else 'alexnet_custom_prediction.png'
    plt.savefig(custom_pred_path, dpi=300)
    plt.show()
    print(f"[ІНФО] Користувацьке передбачення збережено у: {custom_pred_path}")

    print(f"Передбачений клас : {predicted_class}")
    print(f"Впевненість       : {confidence:.2f}%")
    print("Розподіл ймовірностей по класах:")
    for name, prob in sorted(
        zip(class_names, preds[0]), key=lambda x: -x[1]
    ):
        print(f"  {name:<15} {prob * 100:.2f}%")

    return predicted_class


# Приклад виклику функції з довільним зображенням (не з датасету)
# predict_image("/content/drive/MyDrive/test_cat.jpg", model, class_names_list)


# ============================================================
# КРОК 10. Пакетна класифікація 256 зображень → CSV
# ============================================================

num_images  = 256
batch_size_infer = 128
output_csv  = "classification_results.csv"
output_csv_path = os.path.join(BASE_DIR, output_csv) if BASE_DIR else output_csv

# Збираємо всі шляхи до зображень
all_images = []
for class_name in os.listdir(extract_path):
    class_dir = os.path.join(extract_path, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            all_images.append((img_path, class_name))

selected_images = random.sample(all_images, num_images)


def load_batch(image_data):
    """Завантажує батч зображень у масив numpy."""
    images, paths, true_classes = [], [], []
    for img_path, true_class in image_data:
        img       = image.load_img(img_path, target_size=(227, 227))
        img_array = image.img_to_array(img) / 255.0
        images.append(img_array)
        paths.append(img_path)
        true_classes.append(true_class)
    return np.array(images), paths, true_classes


results = []
for i in range(0, num_images, batch_size_infer):
    batch_data                              = selected_images[i:i + batch_size_infer]
    batch_images, batch_paths, batch_true   = load_batch(batch_data)
    predictions_batch                       = model.predict(batch_images)
    predicted_classes                       = np.argmax(predictions_batch, axis=1)

    for j in range(len(batch_paths)):
        results.append([
            batch_paths[j],
            batch_true[j],
            class_names_list[predicted_classes[j]]
        ])

df = pd.DataFrame(
    results,
    columns=["Шлях до файлу", "Справжній клас", "Розпізнаний клас"]
)
df.to_csv(output_csv_path, index=False, encoding="utf-8")
print(f"Результати збережені у {output_csv_path}")

# Показуємо перші кілька рядків
print(df.head(10).to_string(index=False))


# ============================================================
# ДОДАТКОВЕ ЗАВДАННЯ (+1 бал)
# Оптимізована архітектура AlexNet (AlexNet-Lite)
# Порівняння з оригінальною за точністю та обчислювальною
# складністю (кількість параметрів, пам'ять, швидкість)
# ============================================================

MODEL_OPT_FILENAME = "alexnet_lite_model.keras"
model_opt_path = os.path.join(BASE_DIR, MODEL_OPT_FILENAME) if BASE_DIR else MODEL_OPT_FILENAME
RAW_MODEL_OPT_URL = f"https://raw.githubusercontent.com/YuHryshchenko/zpi-zp41_design_neural_networks_HryshchenkoYuliia_KPI_2026/main/lab04/{MODEL_OPT_FILENAME}"

history_opt = None
loaded_opt_successfully = False

# 1. Перевірка локального файлу Lite
if os.path.exists(model_opt_path):
    print(f"[ІНФО] Знайдено локальну модель AlexNet-Lite: {model_opt_path}. Завантаження...")
    try:
        model_opt = tf.keras.models.load_model(model_opt_path)
        loaded_opt_successfully = True
    except Exception as e:
        print(f"[ПОМИЛКА] Не вдалося завантажити локальний файл Lite ({e}).")

# 2. Спроба скачування Lite з GitHub
if not loaded_opt_successfully:
    print(f"[ІНФО] Локальну модель Lite не знайдено. Спроба завантаження з GitHub...")
    try:
        urllib.request.urlretrieve(RAW_MODEL_OPT_URL, model_opt_path)
        print("[ІНФО] Модель AlexNet-Lite успішно завантажено з GitHub!")
        model_opt = tf.keras.models.load_model(model_opt_path)
        loaded_opt_successfully = True
    except Exception as e:
        print(f"[ІНФО] Не вдалося завантажити оптимізовану модель з GitHub ({e}).")

# 3. Навчання Lite, якщо файл відсутній або сталася помилка
if not loaded_opt_successfully:
    print("[ІНФО] Починаємо створення та навчання AlexNet-Lite з нуля...")
    model_opt = Seq([
        INP(shape=(227, 227, 3)),

        C2D(64, (11, 11), strides=4, activation='relu'),
        BN(),
        MP2D((3, 3), strides=2),

        C2D(128, (5, 5), activation='relu', padding='same'),
        BN(),
        MP2D((3, 3), strides=2),

        C2D(192, (3, 3), activation='relu', padding='same'),
        C2D(192, (3, 3), activation='relu', padding='same'),
        C2D(128, (3, 3), activation='relu', padding='same'),
        BN(),
        MP2D((3, 3), strides=2),

        GAP(),                       # замість Flatten → набагато менше параметрів

        D(512, activation='relu'),
        DR(0.5),
        D(10, activation='softmax')
    ], name="AlexNet_Lite")

    optimizer_opt = Adam(learning_rate=0.0001)
    model_opt.compile(
        optimizer=optimizer_opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nНавчання AlexNet-Lite...")
    train_generator.reset()
    val_generator.reset()

    history_opt = model_opt.fit(
        train_generator,
        epochs=2,
        validation_data=val_generator
    )
    
    print(f"[ІНФО] Збереження навченої моделі Lite у {model_opt_path}...")
    model_opt.save(model_opt_path)

print("\n=== Оригінальна AlexNet ===")
model.summary()

print("\n=== Оптимізована AlexNet-Lite ===")
model_opt.summary()

# --- Порівняння кількості параметрів ---
orig_params = model.count_params()
opt_params  = model_opt.count_params()
print(f"\nОригінальна AlexNet  — параметрів: {orig_params:,}")
print(f"AlexNet-Lite         — параметрів: {opt_params:,}")
print(f"Зменшення            : {orig_params / opt_params:.1f}x")

# --- Очікувана оцінка оптимізованої моделі ---
loss_opt, acc_opt = model_opt.evaluate(val_generator)
print(f"\nAlexNet-Lite — Точність на валідації : {acc_opt:.4f}")
print(f"AlexNet-Lite — Втрати (Loss)         : {loss_opt:.4f}")

# --- Швидкість інференсу (час на 1 зображення) ---
sample_img = np.random.rand(1, 227, 227, 3).astype(np.float32)

t0 = time.perf_counter()
for _ in range(100):
    model.predict(sample_img, verbose=0)
t_orig = (time.perf_counter() - t0) / 100 * 1000

t0 = time.perf_counter()
for _ in range(100):
    model_opt.predict(sample_img, verbose=0)
t_opt = (time.perf_counter() - t0) / 100 * 1000

print(f"\nШвидкість інференсу:")
print(f"  Оригінальна AlexNet  : {t_orig:.2f} мс / зображення")
print(f"  AlexNet-Lite         : {t_opt:.2f} мс / зображення")
print(f"  Прискорення          : {t_orig / t_opt:.2f}x")

# --- Порівняльний графік ---
if history is not None and history_opt is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['val_accuracy'],     label='AlexNet (оригінал)')
    axes[0].plot(history_opt.history['val_accuracy'], label='AlexNet-Lite (оптим.)')
    axes[0].set_title('Точність на валідації')
    axes[0].set_xlabel('Епоха')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['val_loss'],     label='AlexNet (оригінал)')
    axes[1].plot(history_opt.history['val_loss'], label='AlexNet-Lite (оптим.)')
    axes[1].set_title('Втрати на валідації')
    axes[1].set_xlabel('Епоха')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.suptitle('Порівняння оригінальної AlexNet та AlexNet-Lite', fontsize=14)
    plt.tight_layout()
    
    comp_plot_path = os.path.join(BASE_DIR, 'alexnet_comparison_history.png') if BASE_DIR else 'alexnet_comparison_history.png'
    plt.savefig(comp_plot_path, dpi=300)
    plt.show()
    print(f"[ІНФО] Порівняльний графік моделей збережено у: {comp_plot_path}")
else:
    print("[ІНФО] Порівняльний графік пропущено (одна або обидві моделі завантажені з готового файлу).")

# --- Зведена таблиця порівняння ---
comparison = {
    "Модель"          : ["AlexNet (оригінал)", "AlexNet-Lite (оптим.)"],
    "Параметрів"      : [f"{orig_params:,}", f"{opt_params:,}"],
    "Val Accuracy"    : [f"{accuracy:.4f}", f"{acc_opt:.4f}"],
    "Val Loss"        : [f"{loss:.4f}", f"{loss_opt:.4f}"],
    "Час інференсу"   : [f"{t_orig:.2f} мс", f"{t_opt:.2f} мс"],
}
df_cmp = pd.DataFrame(comparison)
print("\n=== Зведена таблиця порівняння ===")
print(df_cmp.to_string(index=False))
