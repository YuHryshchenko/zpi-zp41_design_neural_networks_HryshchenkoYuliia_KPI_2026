# ============================================================
# ЛАБОРАТОРНА РОБОТА №5
# Реалізація та дослідження згорткової нейронної мережі InceptionV3
# для класифікації зображень (набір даних tongpython/cat-and-dog)
# ============================================================

import os
import random
import shutil
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Model, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# ПОСТІЙНІ НАЛАШТУВАННЯ (КОНСТАНТИ)
# ==========================================

# --- 1.1 Визначення та налаштування середовища ---
CURRENT_LAB = "lab05"

def is_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if is_kaggle():
    print("Running on Kaggle")
    BASE_DIR = ""
else:
    print("Running locally")
    ABSOLUTE_PATH = os.getcwd()
    BASE_DIR = ABSOLUTE_PATH + "/" + CURRENT_LAB + "/"
    os.makedirs(BASE_DIR, exist_ok=True)

NUM_EPOCHS = 10

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train") if BASE_DIR else "dataset/train"
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation") if BASE_DIR else "dataset/validation"

# ==========================================
# 1. ФУНКЦІЇ ДЛЯ ПІДГОТОВКИ ТА ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================

def dataset_is_ready(path):
    """Повертає True, якщо папки training_set та test_set існують за вказаним шляхом."""
    return (
        os.path.exists(os.path.join(path, 'training_set')) and
        os.path.exists(os.path.join(path, 'test_set'))
    )

def setup_dataset(sample_train_size=300, sample_val_size=100):
    """
    Інтелектуально завантажує датасет (або використовує змонтований у Kaggle),
    а потім створює зменшену збалансовану вибірку для навчання.
    """
    KAGGLE_INPUT_PATH = "/kaggle/input/cat-and-dog"
    LOCAL_DATASET_DIR = os.path.join(BASE_DIR, "cat_dog_source") if BASE_DIR else "cat_dog_source"
    dataset_path = ""

    # Крок 1: Пошук або завантаження повного датасету
    if is_kaggle() and dataset_is_ready(KAGGLE_INPUT_PATH):
        dataset_path = KAGGLE_INPUT_PATH
        print(f"Kaggle: датасет знайдено у {dataset_path}. Завантаження не потрібне.")
    elif dataset_is_ready(LOCAL_DATASET_DIR):
        dataset_path = LOCAL_DATASET_DIR
        print(f"Датасет вже присутній у {dataset_path}. Завантаження пропущено.")
    else:
        try:
            import kaggle
            os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
            print("Завантаження датасету з Kaggle...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'tongpython/cat-and-dog',
                path=LOCAL_DATASET_DIR,
                unzip=True
            )
            print("Завантаження та розпакування завершено.")
            dataset_path = LOCAL_DATASET_DIR
        except Exception as e:
            print(f"Помилка під час роботи з Kaggle API: {e}")
            print("Переконайтеся, що файл kaggle.json знаходиться у директорії ~/.kaggle/")
            return False

    # Крок 2: Створення збалансованої вибірки
    train_cats_dest = os.path.join(TRAIN_DIR, "cats")
    train_dogs_dest = os.path.join(TRAIN_DIR, "dogs")
    val_cats_dest = os.path.join(VAL_DIR, "cats")
    val_dogs_dest = os.path.join(VAL_DIR, "dogs")

    # Перевірка, чи сабсет вже створено
    if os.path.exists(train_cats_dest) and len(os.listdir(train_cats_dest)) > 0:
        print("Збалансована вибірка вже існує. Пропуск етапу копіювання.")
        return True

    print(f"Створення збалансованої вибірки (по {sample_train_size} для тренування, по {sample_val_size} для валідації)...")
    for folder in [train_cats_dest, train_dogs_dest, val_cats_dest, val_dogs_dest]:
        os.makedirs(folder, exist_ok=True)

    # Обробка вкладеності папок датасету tongpython/cat-and-dog
    train_src = os.path.join(dataset_path, "training_set", "training_set")
    val_src = os.path.join(dataset_path, "test_set", "test_set")
    if not os.path.exists(train_src):
        train_src = os.path.join(dataset_path, "training_set")
        val_src = os.path.join(dataset_path, "test_set")

    train_cat_images = random.sample(list(Path(train_src, "cats").glob("*.jpg")), sample_train_size)
    train_dog_images = random.sample(list(Path(train_src, "dogs").glob("*.jpg")), sample_train_size)
    val_cat_images = random.sample(list(Path(val_src, "cats").glob("*.jpg")), sample_val_size)
    val_dog_images = random.sample(list(Path(val_src, "dogs").glob("*.jpg")), sample_val_size)

    for img in train_cat_images: shutil.copy(img, train_cats_dest)
    for img in train_dog_images: shutil.copy(img, train_dogs_dest)
    for img in val_cat_images: shutil.copy(img, val_cats_dest)
    for img in val_dog_images: shutil.copy(img, val_dogs_dest)

    print("Датасет-вибірку успішно підготовлено та структуровано!")
    return True

def create_data_generators(train_dir, val_dir, target_size=(299, 299), batch_size=32):
    """Створює генератори даних із розмноженням (аугментацією) для тренувальних даних."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, rotation_range=30, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=target_size, batch_size=batch_size, class_mode="binary"
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=target_size, batch_size=batch_size, class_mode="binary", shuffle=False
    )
    return train_generator, val_generator

# ==========================================
# 2. АРХІТЕКТУРНІ БЛОКИ (INCEPTION-V3 & MINI)
# ==========================================

def conv2d_bn(x, filters, num_row, num_col, padding="same", strides=(1, 1)):
    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_custom_inception_v3(input_shape=(299, 299, 3), num_classes=1):
    img_input = layers.Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding="valid")
    x = conv2d_bn(x, 32, 3, 3, padding="valid")
    x = conv2d_bn(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding="valid")
    x = conv2d_bn(x, 192, 3, 3, padding="valid")
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Блок Inception A
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 3, 3)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3)

    # Блок Inception B
    branch1x1 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch_pool], axis=3)

    # Блок Inception C
    branch1x1 = conv2d_bn(x, 320, 1, 1)
    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3)
    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch1x1, branch3x3, branch_pool], axis=3)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    return Model(img_input, outputs, name="custom_inception_v3")


def build_mini_inception(input_shape=(150, 150, 3), num_classes=1):
    img_input = layers.Input(shape=input_shape)
    x = conv2d_bn(img_input, 16, 3, 3, strides=(2, 2))
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    branch1x1 = conv2d_bn(x, 32, 1, 1)
    branch3x3 = conv2d_bn(x, 16, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 32, 3, 3)
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1)

    x = layers.concatenate([branch1x1, branch3x3, branch_pool], axis=3)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    return Model(img_input, outputs, name="mini_inception")

# ==========================================
# 3. ФУНКЦІЇ НАВЧАННЯ ТА ОЦІНКИ МОДЕЛІ
# ==========================================

def get_or_train_model(model_name, build_fn, train_gen, val_gen, epochs=NUM_EPOCHS):
    """
    Шукає модель локально, потім на GitHub, якщо не знаходить - тренує з нуля.
    """
    model_path = os.path.join(BASE_DIR, model_name) if BASE_DIR else model_name
    raw_url = f"https://github.com/YuHryshchenko/zpi-zp41_design_neural_networks_HryshchenkoYuliia_KPI_2026/raw/refs/heads/main/lab05/{model_name}"
    
    loaded = False
    model = None

    if os.path.exists(model_path):
        print(f"\n[ІНФО] Знайдено локальну модель {model_name}. Завантаження...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded = True
        except Exception as e:
            print(f"[ПОМИЛКА] Помилка завантаження файлу: {e}")

    if not loaded:
        print(f"\n[ІНФО] Спроба завантаження {model_name} з GitHub...")
        try:
            urllib.request.urlretrieve(raw_url, model_path)
            print(f"[ІНФО] Модель успішно завантажено з GitHub!")
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded = True
        except Exception as e:
            print(f"[ІНФО] Не вдалося завантажити модель з GitHub ({e}).")

    if not loaded:
        print(f"\n[ІНФО] Починаємо навчання {model_name} з нуля...")
        model = build_fn()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy", metrics=["accuracy"]
        )
        
        steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
        validation_steps = max(1, val_gen.samples // val_gen.batch_size)
        
        model.fit(
            train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
            validation_data=val_gen, validation_steps=validation_steps
        )
        
        print(f"[ІНФО] Збереження навченої моделі у {model_path}...")
        model.save(model_path)
    else:
        # Компіляція завантаженої моделі для уникнення попереджень під час evaluation
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy", metrics=["accuracy"]
        )
        
    return model

def evaluate_model(model, val_generator, model_name):
    print(f"\nГенерація прогнозів для валідаційної вибірки ({model_name})...")
    Y_pred = model.predict(val_generator)
    y_pred_classes = (Y_pred > 0.5).astype(int).flatten()
    y_true = val_generator.classes

    class_labels = list(val_generator.class_indices.keys())
    print(f"\n--- Метрики якості {model_name} ---")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Матриця помилок ({model_name})")
    plt.ylabel("Справжній клас")
    plt.xlabel("Передбачений клас")
    
    clean_name = model_name.replace('.keras', '').replace('.h5', '')
    cm_path = os.path.join(BASE_DIR, f'confusion_matrix_{clean_name}.png') if BASE_DIR else f'confusion_matrix_{clean_name}.png'
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"[ІНФО] Матрицю помилок збережено у: {cm_path}")

def demonstrate_models_comparatively(inc_model, mini_model, val_dir, class_indices, num_images=4):
    """Випадково вибирає зображення з валідаційної вибірки та прогнозує їх обома моделями."""
    print("\n=== ДЕМОНСТРАЦІЯ РОБОТИ МОДЕЛЕЙ НА ВИПАДКОВИХ ЗОБРАЖЕННЯХ ===")
    all_images = []
    for class_name in ['cats', 'dogs']:
        class_path = os.path.join(val_dir, class_name)
        if os.path.exists(class_path):
            imgs = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.jpg')]
            all_images.extend([(img, class_name) for img in imgs])

    if not all_images:
        print("[ПОМИЛКА] Не знайдено зображень для демонстрації.")
        return

    selected = random.sample(all_images, min(num_images, len(all_images)))
    labels_inv = {v: k for k, v in class_indices.items()}

    fig, axes = plt.subplots(1, len(selected), figsize=(16, 5))
    if len(selected) == 1: axes = [axes]

    for i, (img_path, true_label) in enumerate(selected):
        # Передбачення Inception
        img_inc = Image.open(img_path).resize((299, 299))
        img_arr_inc = np.expand_dims(np.array(img_inc).astype("float32") / 255.0, axis=0)
        pred_inc = inc_model.predict(img_arr_inc, verbose=0)[0][0]
        inc_class = labels_inv[1] if pred_inc > 0.5 else labels_inv[0]
        inc_conf = pred_inc if pred_inc > 0.5 else 1 - pred_inc

        # Передбачення Mini-Inception
        img_mini = Image.open(img_path).resize((150, 150))
        img_arr_mini = np.expand_dims(np.array(img_mini).astype("float32") / 255.0, axis=0)
        pred_mini = mini_model.predict(img_arr_mini, verbose=0)[0][0]
        mini_class = labels_inv[1] if pred_mini > 0.5 else labels_inv[0]
        mini_conf = pred_mini if pred_mini > 0.5 else 1 - pred_mini

        axes[i].imshow(img_inc)
        axes[i].axis('off')
        axes[i].set_title(
            f"Справжній: {true_label}\n\nInception: {inc_class} ({inc_conf*100:.1f}%)\nMini: {mini_class} ({mini_conf*100:.1f}%)",
            fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    demo_path = os.path.join(BASE_DIR, 'models_demonstration_lab05.png') if BASE_DIR else 'models_demonstration_lab05.png'
    plt.savefig(demo_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"[ІНФО] Результати порівняльної демонстрації збережено у: {demo_path}")


# ==========================================
# ГОЛОВНИЙ ПОТІК ВИКОНАННЯ (MAIN EXECUTION)
# ==========================================
if __name__ == "__main__":
    
    # Крок 1: Завантаження та структурування
    success = setup_dataset(sample_train_size=300, sample_val_size=100)
    
    if not success or not (os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR)):
        print("Помилка: Не вдалося налаштувати директорії даних. Перевірте підключення до Kaggle.")
    else:
        # Крок 2: Генератори для Inception (299x299) та Mini (150x150)
        print("\nІніціалізація генераторів для Inception-v3 (299x299)...")
        train_gen_inc, val_gen_inc = create_data_generators(TRAIN_DIR, VAL_DIR, target_size=(299, 299))
        
        print("Ініціалізація генераторів для Mini-Inception (150x150)...")
        train_gen_mini, val_gen_mini = create_data_generators(TRAIN_DIR, VAL_DIR, target_size=(150, 150))

        # Крок 3: Завантаження або навчання InceptionV3
        inception_model = get_or_train_model(
            model_name="my_custom_inception_model.keras",
            build_fn=build_custom_inception_v3,
            train_gen=train_gen_inc,
            val_gen=val_gen_inc,
            epochs=NUM_EPOCHS
        )

        # Крок 4: Завантаження або навчання Mini-Inception
        mini_inception_model = get_or_train_model(
            model_name="mini_inception_model.keras",
            build_fn=build_mini_inception,
            train_gen=train_gen_mini,
            val_gen=val_gen_mini,
            epochs=NUM_EPOCHS
        )

        # Крок 5: Оцінка моделей (Матриці помилок)
        evaluate_model(inception_model, val_gen_inc, "InceptionV3")
        evaluate_model(mini_inception_model, val_gen_mini, "Mini-Inception")

        # Крок 6: Демонстрація моделей на випадкових зображеннях
        demonstrate_models_comparatively(
            inception_model, 
            mini_inception_model, 
            VAL_DIR, 
            train_gen_inc.class_indices, 
            num_images=4
        )

        # Крок 7: Порівняння архітектур
        print("\n=== ПОРІВНЯННЯ ОБЧИСЛЮВАЛЬНОЇ СКЛАДНОСТІ ARCHITECTURES ===")
        print("\n1. Специфікація повної кастомної Inception-v3:")
        inception_model.summary()

        print("\n2. Специфікація оптимізованої Mini-Inception:")
        mini_inception_model.summary()
        
        print("\nВисновок: Mini-Inception оперує значно меншою кількістю параметрів, зменшуючи")
        print("використання оперативної пам'яті (VRAM) та прискорюючи крок навчання, проте")
        print("може демонструвати нижчу узагальнюючу здатність на складних вибірках.")