# ============================================================
# ЛАБОРАТОРНА РОБОТА №6
# Реалізація та дослідження згорткової нейронної мережі Xception
# для обробки відео (бінарна класифікація: логотип / не логотип)
# ============================================================
# Задача: розпізнавання логотипу бренду (наприклад, BMW) на відео.
# Архітектура Xception реалізована пошарово (завантаження готової
# моделі ЗАБОРОНЕНО; завантаження ваг ImageNet дозволяється).
# ============================================================

import os
import cv2
import time
import glob
import random
import zipfile
import datetime
import numpy as np
import pandas as pd
import urllib.request
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
# from google.colab import drive
from scipy.ndimage import median_filter
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

# ============================================================
# КРОК 1. Завантаження даних з GitHub та автоматична 
# генерація фонових зображень і датасету
# ============================================================

# --- 1.1 Завантаження архіву з логотипами ---
GITHUB_URL_BASE = "https://github.com/YuHryshchenko/zpi-zp41_design_neural_networks_HryshchenkoYuliia_KPI_2026/raw/main/lab06/"
LOGOS_GITHUB_URL = GITHUB_URL_BASE + "raw-img.zip"
VIDEO_GITHUB_URL = GITHUB_URL_BASE + "logo_video.mp4"
VIDEO_DIR  = 'video'
EXTRACT_DIR = "raw-img"
VIDEO_PATH  = 'logo_video.mp4'
ZIP_PATH = "raw_img.zip"

# --- 1.2 Автоматичне завантаження фонових зображень ---
if not os.path.exists(EXTRACT_DIR):
    print("Завантаження архіву з логотипами GitHub...")
    urllib.request.urlretrieve(LOGOS_GITHUB_URL, ZIP_PATH)
    
    print("Розпакування архіву...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("Розпакування завершено!")
else:
    print("Архів з логотипами вже завантажено.")

# --- 1.3 Автоматичне завантаження відео ---
if not os.path.exists(VIDEO_DIR):
    print("Завантаження відео...")
    urllib.request.urlretrieve(VIDEO_GITHUB_URL, VIDEO_PATH)
    print("Відео завантажено!")
else:
    print("Відео вже завантажено.")

# --- 2. Автоматичне завантаження фонових зображень ---
background_dir = os.path.join(EXTRACT_DIR, 'backgrounds')
os.makedirs(background_dir, exist_ok=True)

# Використовуємо сервіс Picsum для стабільного отримання 5 різних фонів розміром 800x600
bg_urls = [
    "https://picsum.photos/seed/city/800/600",
    "https://picsum.photos/seed/nature/800/600",
    "https://picsum.photos/seed/road/800/600",
    "https://picsum.photos/seed/room/800/600",
    "https://picsum.photos/seed/abstract/800/600"
]

print("Перевірка та завантаження фонових зображень...")
for i, url in enumerate(bg_urls):
    bg_file_path = os.path.join(background_dir, f"bg_{i}.jpg")
    if not os.path.exists(bg_file_path):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req) as response, open(bg_file_path, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Не вдалося завантажити фон {url}: {e}")

# --- Налаштування шляхів ---
logo_dir   = os.path.join(EXTRACT_DIR, 'raw-img/toyota')
output_dir = './dataset'

splits  = ['train', 'val', 'test']
classes = ['positive', 'negative']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# --- Аугментація логотипу ---
def augment_logo(logo):
    """Випадковий поворот, масштабування та прозорість логотипу."""
    angle = random.uniform(-30, 30)
    logo  = logo.rotate(angle, expand=True)

    scale = random.uniform(0.5, 1.5)
    w, h  = logo.size
    logo  = logo.resize((int(w * scale), int(h * scale)))

    alpha = random.uniform(0.6, 1.0) # Трохи зменшив прозорість для кращої видимості
    logo.putalpha(int(255 * alpha))
    return logo

# --- Збираємо списки файлів ---
logo_files = [os.path.join(logo_dir, f) for f in os.listdir(logo_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
background_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Перевірка на всякий випадок, щоб уникнути помилки IndexError, якщо картинки не завантажаться
if not background_files:
    raise ValueError("Помилка: Немає фонових зображень у папці backgrounds! Перевірте інтернет-з'єднання або завантажте кілька картинок туди вручну.")
if not logo_files:
    raise ValueError("Помилка: Немає зображень логотипів у папці logos! Перевірте архів.")

total_positive = 1000
total_negative = 1000
split_ratios   = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def get_split_name(idx, total):
    val_thresh  = int(total * split_ratios['train'])
    test_thresh = int(total * (split_ratios['train'] + split_ratios['val']))
    if idx < val_thresh: return 'train'
    elif idx < test_thresh: return 'val'
    else: return 'test'

# ── Підрахунок вже наявних зображень ──────────────────────────
def count_dataset_images(output_dir, splits, classes):
    """Повертає загальну кількість вже наявних зображень у датасеті."""
    total = 0
    for split in splits:
        for cls in classes:
            folder = os.path.join(output_dir, split, cls)
            if os.path.isdir(folder):
                total += len([f for f in os.listdir(folder)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return total

existing_images = count_dataset_images(output_dir, splits, classes)
expected_images = total_positive + total_negative   # 2000

if existing_images >= expected_images:
    print(f"Датасет вже існує ({existing_images} зображень). Генерацію пропущено.")
else:
    print(f"Знайдено {existing_images}/{expected_images} зображень. Починаємо генерацію...")

    # --- Генерація позитивних зразків (логотип на фоні) ---
    # Визначаємо, скільки вже є в кожній split/class, щоб не перезаписувати
    pos_existing = sum(
        len(os.listdir(os.path.join(output_dir, s, 'positive')))
        for s in splits if os.path.isdir(os.path.join(output_dir, s, 'positive'))
    )
    neg_existing = sum(
        len(os.listdir(os.path.join(output_dir, s, 'negative')))
        for s in splits if os.path.isdir(os.path.join(output_dir, s, 'negative'))
    )

    if pos_existing < total_positive:
        i    = pos_existing          # продовжуємо нумерацію з того місця, де зупинились
        pbar = tqdm(total=total_positive - pos_existing, desc='Generating positive samples')

        while i < total_positive:
            bg_path   = random.choice(background_files)
            logo_path = random.choice(logo_files)
            try:
                bg   = Image.open(bg_path).convert('RGB')
                logo = Image.open(logo_path).convert('RGBA')
            except Exception:
                continue

            crop_w, crop_h = 300, 300
            if bg.width > crop_w and bg.height > crop_h:
                cx = random.randint(0, bg.width - crop_w)
                cy = random.randint(0, bg.height - crop_h)
                bg = bg.crop((cx, cy, cx + crop_w, cy + crop_h))

            logo = augment_logo(logo)

            if logo.width > bg.width or logo.height > bg.height:
                logo.thumbnail((bg.width, bg.height))

            max_x = bg.width  - logo.width
            max_y = bg.height - logo.height
            if max_x < 0 or max_y < 0:
                continue

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            bg.paste(logo, (x, y), logo)

            split = get_split_name(i, total_positive)
            bg.save(os.path.join(output_dir, split, 'positive', f'pos_{i}.jpg'))
            i += 1
            pbar.update(1)
        pbar.close()
    else:
        print(f"Позитивні зразки вже існують ({pos_existing}). Пропускаємо.")

    if neg_existing < total_negative:
        i    = neg_existing
        pbar = tqdm(total=total_negative - neg_existing, desc='Generating negative samples')

        while i < total_negative:
            bg_path = random.choice(background_files)
            try:
                bg = Image.open(bg_path).convert('RGB')
            except Exception:
                continue

            crop_w, crop_h = 300, 300
            if bg.width > crop_w and bg.height > crop_h:
                cx = random.randint(0, bg.width - crop_w)
                cy = random.randint(0, bg.height - crop_h)
                bg = bg.crop((cx, cy, cx + crop_w, cy + crop_h))

            split = get_split_name(i, total_negative)
            bg.save(os.path.join(output_dir, split, 'negative', f'neg_{i}.jpg'))
            i += 1
            pbar.update(1)
        pbar.close()
    else:
        print(f"Негативні зразки вже існують ({neg_existing}). Пропускаємо.")

    print("Датасет успішно згенеровано!")

# -- Опціонально монтуємо Google Drive та генеруємо датасет звідти ---
# drive.mount('/content/drive')
# Mounted at /content/drive  (або Drive already mounted at ...)

# ------------------------ Налаштування шляхів -----------------------
#logo_dir       = '/content/drive/MyDrive/logos'   # PNG-зображення логотипу
#background_dir = '/content/data'                  # Фонові зображення
#output_dir     = '/content/dataset'               # Вихідний датасет

#splits  = ['train', 'val', 'test']
#classes = ['positive', 'negative']

# Створюємо структуру папок
# for split in splits:
#    for cls in classes:
#        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# ----------------- Аугментація логотипу -----------------------------
def augment_logo(logo):
    """Випадковий поворот, масштабування та прозорість логотипу."""
    angle = random.uniform(-30, 30)
    logo  = logo.rotate(angle, expand=True)

    scale = random.uniform(0.5, 1.5)
    w, h  = logo.size
    logo  = logo.resize((int(w * scale), int(h * scale)))

    alpha = random.uniform(0.5, 1.0)
    logo.putalpha(int(255 * alpha))
    return logo

# --------------------- Збираємо списки файлів -------------------------
logo_files = [
    os.path.join(logo_dir, f)
    for f in os.listdir(logo_dir)
    if f.endswith(('.png', '.jpg', '.jpeg'))
]

background_files = [
    os.path.join(background_dir, f)
    for f in os.listdir(background_dir)
    if f.endswith(('.png', '.jpg', '.jpeg'))
]

total_positive = 1000
total_negative = 1000
split_ratios   = {'train': 0.7, 'val': 0.15, 'test': 0.15}

def get_split_name(idx, total):
    """Визначає, до якого split належить зображення за індексом."""
    val_thresh  = int(total * split_ratios['train'])
    test_thresh = int(total * (split_ratios['train'] + split_ratios['val']))
    if idx < val_thresh:
        return 'train'
    elif idx < test_thresh:
        return 'val'
    else:
        return 'test'

# ------ Генерація позитивних зразків (логотип на фоні) ------
i    = 0
pbar = tqdm(total=total_positive, desc='Generating positive samples')

while i < total_positive:
    bg_path   = random.choice(background_files)
    logo_path = random.choice(logo_files)
    try:
        bg   = Image.open(bg_path).convert('RGB')
        logo = Image.open(logo_path).convert('RGBA')
    except Exception as e:
        print(f"Error opening image: {e}")
        continue

    logo = augment_logo(logo)

    if logo.width > bg.width or logo.height > bg.height:
        logo.thumbnail((bg.width, bg.height))

    max_x = bg.width  - logo.width
    max_y = bg.height - logo.height
    if max_x < 0 or max_y < 0:
        continue

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    bg.paste(logo, (x, y), logo)

    split = get_split_name(i, total_positive)
    bg.save(os.path.join(output_dir, split, 'positive', f'pos_{i}.jpg'))
    i += 1
    pbar.update(1)

pbar.close()

# ------ Генерація негативних зразків (тільки фон) ------
i    = 0
pbar = tqdm(total=total_negative, desc='Generating negative samples')

while i < total_negative:
    bg_path = random.choice(background_files)
    try:
        bg = Image.open(bg_path).convert('RGB')
    except Exception as e:
        print(f"Error opening background image: {e}")
        continue

    split = get_split_name(i, total_negative)
    bg.save(os.path.join(output_dir, split, 'negative', f'neg_{i}.jpg'))
    i += 1
    pbar.update(1)

pbar.close()

# Generating positive samples: 100%|██████████| 1000/1000 [02:51<00:00, 5.84it/s]
# Generating negative samples: 100%|██████████| 1000/1000 [00:00<00:00, 1433.40it/s]

# ============================================================
# КРОК 2. Готуємо дані (ImageDataGenerator + train/val/test)
# ============================================================

dataset_dir = 'dataset'
img_size    = (150, 150)
batch_size  = 32

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = datagen.flow_from_directory(
    directory=os.path.join(dataset_dir, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Found 1400 images belonging to 2 classes.
# Found  300 images belonging to 2 classes.
# Found  300 images belonging to 2 classes.

# ============================================================
# КРОК 3. Будуємо архітектуру Xception (пошарово, без готової
#         моделі з бібліотеки) та навчаємо
# ============================================================

def build_xception(input_shape=(150, 150, 3), num_classes=1):
    """
    Спрощена архітектура Xception для бінарної класифікації.

    Ключові елементи, що відповідають оригінальній Xception:
      - Початковий звичайний Conv2D (entry flow)
      - SeparableConv2D з residual-з'єднаннями (depthwise separable conv)
      - BatchNormalization після кожної згортки
      - GlobalAveragePooling2D замість Flatten
      - Dropout + Dense для класифікатора
      - Вихід — «сирі» логіти (from_logits=True → без Softmax/Sigmoid)
    """
    inputs = layers.Input(shape=input_shape)

    # --- Entry flow ---
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # --- Блок 1: SeparableConv + residual ---
    skip = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same')(x)
    skip = layers.BatchNormalization()(skip)

    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.add([x, skip])

    # --- Блок 2: SeparableConv + residual ---
    skip = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    skip = layers.BatchNormalization()(skip)

    x = layers.SeparableConv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = layers.add([x, skip])

    # --- Middle flow ---
    x = layers.SeparableConv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # --- Exit flow ---
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes)(x)   # логіти (без активації)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# ── Завантаження моделі якщо вона вже збережена ───────────────

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Шукаємо будь-який .keras файл у папці model/
existing_model_files = sorted(glob.glob(os.path.join(MODEL_DIR, 'xception_*.keras')))

if existing_model_files:
    # Беремо найновіший файл (сортування за іменем → останній timestamp)
    model_load_path = existing_model_files[-1]
    print(f"Знайдено збережену модель: {model_load_path}")
    print("Завантажуємо модель, навчання пропущено.")
    model = tf.keras.models.load_model(model_load_path)
    model.summary()
    history = None   # history недоступна при завантаженні
else:
    print("Збереженої моделі не знайдено. Починаємо навчання...")
    model = build_xception()
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[early_stop]
    )
# ── Кінець завантаження моделі ──────────────────────────────

# Epoch 1/30  ...  Epoch 30/30
# Restoring model weights from the end of the best epoch: 27.

# ============================================================
# КРОК 4. Перевіряємо модель на тестових даних
# ============================================================

print("\nEvaluating on test data...")

test_logits = model.predict(test_generator)
test_probs  = tf.nn.sigmoid(test_logits).numpy().flatten()
test_preds  = (test_probs > 0.5).astype(int)
test_labels = test_generator.labels

test_acc = np.mean(test_preds == test_labels)
print(f"Test accuracy: {test_acc:.4f}")
print("Test prediction stats:")
print(f"  Min: {test_probs.min():.4f}, Max: {test_probs.max():.4f}")
print(f"  Mean: {test_probs.mean():.4f}, Std: {test_probs.std():.4f}")

# Test accuracy: 0.9633
# Test prediction stats:
#   Min: 0.0001, Max: 1.0000
#   Mean: 0.5149, Std: 0.4650

# ============================================================
# КРОК 5. Матриця помилок + accuracy, precision, recall, F-Score
# ============================================================

class_names = list(train_generator.class_indices.keys())  # ['negative', 'positive']

# --- Матриця помилок ---
cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(6, 5))
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
plt.savefig('confusion_matrix.png')
plt.show()

# --- Метрики ---
acc_score  = accuracy_score(test_labels, test_preds)
prec_score = precision_score(test_labels, test_preds, zero_division=0)
rec_score  = recall_score(test_labels, test_preds, zero_division=0)
f1         = f1_score(test_labels, test_preds, zero_division=0)

print(f"\nAccuracy  : {acc_score:.4f}")
print(f"Precision : {prec_score:.4f}")
print(f"Recall    : {rec_score:.4f}")
print(f"F-Score   : {f1:.4f}")
print("\nДетальний звіт:")
print(classification_report(test_labels, test_preds,
                             target_names=class_names, zero_division=0))

# ============================================================
# КРОК 6. Графіки: розподіл передбачень та навчання
# ============================================================

# --- Гістограма передбачень ---
plt.figure(figsize=(10, 5))
plt.hist(test_probs, bins=20, alpha=0.7)
plt.title('Test Predictions Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.grid(alpha=0.3)
plt.savefig('test_predictions.png')
plt.show()

# --- Графіки точності та втрат (лише якщо модель щойно навчалась) ---
if history is not None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
else:
    print("Графіки навчання недоступні (модель завантажена з файлу).")


# ============================================================
# КРОК 7. Перевіряємо модель на випадкових тестових зображеннях
# ============================================================

num_samples      = 5
num_test_samples = len(test_generator.labels)
random_indices   = random.sample(range(num_test_samples), num_samples)

plt.figure(figsize=(15, 10))
for i, index in enumerate(random_indices):
    image_path = test_generator.filepaths[index]
    true_label = test_generator.labels[index]

    img       = tf.keras.utils.load_img(image_path, target_size=model.input_shape[1:3])
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions   = model.predict(img_array)
    probabilities = tf.nn.sigmoid(predictions).numpy().flatten()

    predicted_class = 1 if probabilities[0] > 0.5 else 0
    predicted_label = class_names[predicted_class]
    true_label_name = class_names[true_label]

    plt.subplot(1, num_samples, i + 1)
    plt.imshow(img)
    plt.title(
        f"True: {true_label_name}\n"
        f"Predicted: {predicted_label} ({probabilities[0]:.2f})"
    )
    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================
# КРОК 8. Зберігаємо модель
# ============================================================

now           = datetime.datetime.now()
timestamp     = now.strftime("%Y%m%d_%H%M%S")
base_path     = 'model'
file_name     = f'xception_{timestamp}.keras'
model_save_path = os.path.join(base_path, file_name)

# ── ЗМІНА 3: зберігаємо лише якщо модель щойно навчалась ───────────────
if history is not None:
    model.save(model_save_path)
    print(f"Модель успішно збережено за шляхом: {model_save_path}")
else:
    print(f"Модель завантажена з файлу, повторне збереження пропущено.")
# ── кінець ЗМІНИ 3 ─────────────────────────────────────────────────────

# ============================================================
# КРОК 9. Тестуємо модель на відео
#
# Результат: графік впевненості по кадрах +
#            вивід часових інтервалів появи логотипу
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

# Отримуємо FPS для перетворення кадрів у час
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

batch_size_video   = 32
frames             = []
frames_ids         = []
frame_predictions  = []
frame_numbers      = []
frame_count        = 0

def preprocess_frame(frame):
    """Масштабуємо кадр до (150×150) і нормалізуємо до [0, 1]."""
    resized = cv2.resize(frame, (150, 150))
    return resized.astype(np.float32) / 255.0

def process_batch(batch_frames, batch_ids):
    """Передбачає клас для батчу кадрів (повертає ймовірності через sigmoid)."""
    batch_np    = np.array(batch_frames)
    predictions = model.predict(batch_np, verbose=0)
    for i in range(len(predictions)):
        score = float(1 / (1 + np.exp(-predictions[i][0])))   # sigmoid вручну
        frame_predictions.append(score)
        frame_numbers.append(batch_ids[i])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(preprocess_frame(frame))
    frames_ids.append(frame_count)

    if len(frames) == batch_size_video:
        process_batch(frames, frames_ids)
        frames     = []
        frames_ids = []

    frame_count += 1

if frames:
    process_batch(frames, frames_ids)

cap.release()

# --- Графік передбачень по кадрах ---
plt.figure(figsize=(12, 6))
plt.plot(frame_numbers, frame_predictions, label='Confidence', color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.xlabel('Frame Number')
plt.ylabel('Prediction Score')
plt.title('Model Predictions Over Video Frames')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('video_predictions.png')
plt.show()

# --- Визначаємо часові інтервали появи логотипу ---
THRESHOLD = 0.5
logo_detected = [p >= THRESHOLD for p in frame_predictions]

intervals = []
in_segment = False
seg_start  = 0

for idx, detected in enumerate(logo_detected):
    if detected and not in_segment:
        in_segment = True
        seg_start  = frame_numbers[idx]
    elif not detected and in_segment:
        in_segment = False
        seg_end = frame_numbers[idx - 1]
        intervals.append((seg_start, seg_end))

if in_segment:
    intervals.append((seg_start, frame_numbers[-1]))

print("\nЧасові інтервали появи логотипу (кадри → секунди):")
if intervals:
    for start_f, end_f in intervals:
        t_start = start_f / fps
        t_end   = end_f   / fps
        print(f"  Кадри {start_f:4d}–{end_f:4d}  →  "
              f"{t_start:.2f}s – {t_end:.2f}s")
else:
    print("  Логотип не виявлено жодного разу.")


# ============================================================
# ДОДАТКОВЕ ЗАВДАННЯ (+1 бал)
# Підвищення ефективності обробки відео:
#   A) Батчева обробка (порівняння зі покадровою)
#   B) Пост-обробка (медіанне згладжування + заповнення прогалин)
#   C) Порівняння результатів
# ============================================================

# ---- A) ПОФРЕЙМОВА (baseline) vs. БАТЧЕВА обробка ----

cap_a = cv2.VideoCapture(VIDEO_PATH)
all_raw_frames = []

while True:
    ret, fr = cap_a.read()
    if not ret:
        break
    all_raw_frames.append(preprocess_frame(fr))
cap_a.release()

all_raw_frames = np.array(all_raw_frames)
total_frames   = len(all_raw_frames)

# Покадрово
t0 = time.perf_counter()
preds_single = []
for fr in all_raw_frames:
    p = model.predict(fr[np.newaxis], verbose=0)
    preds_single.append(float(1 / (1 + np.exp(-p[0][0]))))
t_single = time.perf_counter() - t0

# Батчево (batch=32)
BATCH = 32
t0 = time.perf_counter()
preds_batch = []
for i in range(0, total_frames, BATCH):
    chunk = all_raw_frames[i:i + BATCH]
    p     = model.predict(chunk, verbose=0)
    for val in p:
        preds_batch.append(float(1 / (1 + np.exp(-val[0]))))
t_batch = time.perf_counter() - t0

print(f"\n=== A) Швидкість обробки відео ===")
print(f"  Покадрово : {t_single:.2f}s  ({total_frames / t_single:.1f} fps)")
print(f"  Батч ({BATCH}) : {t_batch:.2f}s  ({total_frames / t_batch:.1f} fps)")
print(f"  Прискорення: {t_single / t_batch:.2f}x")


# ---- B) ПОСТ-ОБРОБКА результатів ----

preds_arr = np.array(preds_batch)

# 1. Медіанне згладжування (усуває поодинокі хибні спрацьовування)
MEDIAN_WIN = 9
preds_smooth = median_filter(preds_arr, size=MEDIAN_WIN)

# 2. Заповнення коротких прогалин (< GAP_FILL кадрів) між позитивними сегментами
GAP_FILL     = 15   # кадрів
binary_raw   = (preds_arr    >= THRESHOLD).astype(int)
binary_smooth = (preds_smooth >= THRESHOLD).astype(int)

# Заповнення прогалин у двійковій масці
def fill_gaps(binary_mask, max_gap):
    """Заповнює відрізки з нулів довжиною ≤ max_gap між двома одиницями."""
    filled = binary_mask.copy()
    i = 0
    while i < len(filled):
        if filled[i] == 0:
            j = i
            while j < len(filled) and filled[j] == 0:
                j += 1
            if (i > 0 and j < len(filled) and j - i <= max_gap):
                filled[i:j] = 1
            i = j
        else:
            i += 1
    return filled

binary_filled = fill_gaps(binary_smooth, GAP_FILL)


def extract_intervals(binary_mask, fps_val):
    """Повертає список часових інтервалів (кадр_початок, кадр_кінець, t_start, t_end)."""
    segs, active, start = [], False, 0
    for idx, val in enumerate(binary_mask):
        if val and not active:
            active, start = True, idx
        elif not val and active:
            active = False
            segs.append((start, idx - 1,
                         start / fps_val, (idx - 1) / fps_val))
    if active:
        segs.append((start, len(binary_mask) - 1,
                     start / fps_val, (len(binary_mask) - 1) / fps_val))
    return segs

frames_axis = list(range(total_frames))
segs_raw    = extract_intervals(binary_raw,    fps)
segs_smooth = extract_intervals(binary_filled, fps)

print(f"\n=== B) Пост-обробка: кількість сегментів ===")
print(f"  Без пост-обробки         : {len(segs_raw)} сегментів")
print(f"  Медіана + заповнення ({GAP_FILL}): {len(segs_smooth)} сегментів")

print("\nСегменти після пост-обробки:")
for s in segs_smooth:
    print(f"  Кадри {s[0]:4d}–{s[1]:4d}  →  {s[2]:.2f}s – {s[3]:.2f}s")

# ---- C) ПОРІВНЯЛЬНИЙ ГРАФІК ----

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(frames_axis, preds_arr, color='steelblue', linewidth=0.8,
             label='Без обробки')
axes[0].axhline(THRESHOLD, color='red', linestyle='--', alpha=0.7,
                label=f'Поріг = {THRESHOLD}')
axes[0].set_ylabel('Впевненість')
axes[0].set_title('Передбачення: без пост-обробки')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(frames_axis, preds_smooth, color='darkorange', linewidth=0.8,
             label=f'Медіана (вікно={MEDIAN_WIN})')
axes[1].axhline(THRESHOLD, color='red', linestyle='--', alpha=0.7)
axes[1].set_ylabel('Впевненість')
axes[1].set_title('Після медіанного згладжування')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

axes[2].fill_between(frames_axis, binary_filled, step='mid',
                     alpha=0.6, color='green', label='Логотип виявлено')
axes[2].set_ylabel('Виявлено (0/1)')
axes[2].set_xlabel('Номер кадру')
axes[2].set_title(f'Фінальна маска (медіана + заповнення прогалин ≤ {GAP_FILL} кадрів)')
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.suptitle('Порівняння методів обробки відео', fontsize=14)
plt.tight_layout()
plt.savefig('video_postprocessing_comparison.png')
plt.show()

# ---- Зведена таблиця ----

df_compare = pd.DataFrame({
    'Метод'         : ['Покадрово (без пост-обробки)',
                       f'Батч={BATCH} + медіана + заповнення прогалин'],
    'Час (сек)'     : [f'{t_single:.2f}', f'{t_batch:.2f}'],
    'FPS'           : [f'{total_frames / t_single:.1f}',
                       f'{total_frames / t_batch:.1f}'],
    'Сегментів'     : [len(segs_raw), len(segs_smooth)],
})
print("\n=== C) Зведена таблиця порівняння ===")
print(df_compare.to_string(index=False))