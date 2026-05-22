import os
import random
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Model, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kaggle

# ==========================================
# ПОСТІЙНІ НАЛАШТУВАННЯ (КОНСТАНТИ)
# ==========================================
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/validation"
MODEL_FILE_NAME = "my_custom_inception_model.h5"

# ==========================================
# 1. ФУНКЦІЇ ДЛЯ ПІДГОТОВКИ ТА ЗАВАНТАЖЕННЯ ДАНИХ
# ==========================================
def download_and_prepare_dataset(dataset_base_dir="dataset", sample_train_size=300, sample_val_size=100):
    """Завантажує датасет з Kaggle та створює збалансовану вибірку."""
    if kaggle is None:
        print("Помилка: Бібліотека 'kaggle' не встановлена. Пропуск кроку завантаження.")
        return

    print("Аутентифікація в Kaggle та завантаження датасету...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset="tongpython/cat-and-dog", path=".", unzip=True)

    TRAIN_SOURCE_DIR = "training_set/training_set"
    VALIDATION_SOURCE_DIR = "test_set/test_set"

    train_cats = os.path.join(dataset_base_dir, "train/cats")
    train_dogs = os.path.join(dataset_base_dir, "train/dogs")
    val_cats = os.path.join(dataset_base_dir, "validation/cats")
    val_dogs = os.path.join(dataset_base_dir, "validation/dogs")

    # Створення структури папок
    for folder in [train_cats, train_dogs, val_cats, val_dogs]:
        os.makedirs(folder, exist_ok=True)

    # Визначення джерел файлів
    train_cat_source = Path(TRAIN_SOURCE_DIR) / "cats"
    train_dog_source = Path(TRAIN_SOURCE_DIR) / "dogs"
    validation_cat_source = Path(VALIDATION_SOURCE_DIR) / "cats"
    validation_dog_source = Path(VALIDATION_SOURCE_DIR) / "dogs"

    print(f"Вибірка випадкових зображень (по {sample_train_size} для тренування, по {sample_val_size} для валідації)...")
    train_cat_images = random.sample(list(train_cat_source.glob("*.jpg")), sample_train_size)
    train_dog_images = random.sample(list(train_dog_source.glob("*.jpg")), sample_train_size)
    validation_cat_images = random.sample(list(validation_cat_source.glob("*.jpg")), sample_val_size)
    validation_dog_images = random.sample(list(validation_dog_source.glob("*.jpg")), sample_val_size)

    # Копіювання обраних файлів
    for img in train_cat_images:
        shutil.copy(img, train_cats)
    for img in train_dog_images:
        shutil.copy(img, train_dogs)
    for img in validation_cat_images:
        shutil.copy(img, val_cats)
    for img in validation_dog_images:
        shutil.copy(img, val_dogs)

    print("Датасет успішно підготовлено та структуровано!")


def create_data_generators(train_dir, val_dir, target_size=(299, 299), batch_size=32):
    """Створює генератори даних із розмноженням (аугментацією) для тренувальних даних."""
    print("Ініціалізація генераторів даних та конвеєра аугментації...")
    
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, val_generator


# ==========================================
# 2. АРХІТЕКТУРНІ БЛОКИ (INCEPTION-V3 & MINI)
# ==========================================
def conv2d_bn(x, filters, num_row, num_col, padding="same", strides=(1, 1)):
    """Допоміжна функція: Згортка + Batch Normalization + Activation (ReLU)."""
    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_custom_inception_v3(input_shape=(299, 299, 3), num_classes=1):
    """Пошарова кастомна реалізація архітектури Inception-v3."""
    img_input = layers.Input(shape=input_shape)

    # Stem (Вхідний блок)
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

    # Вихідний блок класифікації
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = Model(img_input, outputs, name="custom_inception_v3")
    return model


def build_mini_inception(input_shape=(150, 150, 3), num_classes=1):
    """Спрощена архітектура Mini-Inception для швидких обчислень."""
    img_input = layers.Input(shape=input_shape)
    x = conv2d_bn(img_input, 16, 3, 3, strides=(2, 2))
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Один базовий оптимізований Inception блок
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
def train_or_load_model(model, train_generator, val_generator, model_path, epochs=10):
    """Навчає модель, якщо файл збереженої моделі відсутній, або завантажує наявний."""
    if not os.path.exists(model_path):
        print(f"Файл моделі '{model_path}' не знайдено. Початок навчання...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
        validation_steps = max(1, val_generator.samples // val_generator.batch_size)
        
        model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps
        )
        model.save(model_path)
        print(f"Модель успішно збережено у файл '{model_path}'.")
    else:
        print(f"Знайдено збережену модель. Завантаження ваг з '{model_path}'...")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
    return model


def evaluate_model(model, val_generator):
    """Розраховує метрики класифікації та візуалізує матрицю помилок (Confusion Matrix)."""
    print("\nГенерація прогнозів для валідаційної вибірки...")
    Y_pred = model.predict(val_generator)
    y_pred_classes = (Y_pred > 0.5).astype(int).flatten()
    y_true = val_generator.classes

    print("\n--- Метрики якості (Accuracy, Precision, Recall, F1-Score) ---")
    class_labels = list(val_generator.class_indices.keys())
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))

    # Візуалізація матриці помилок
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Матриця помилок (Confusion Matrix)")
    plt.ylabel("Справжній клас")
    plt.xlabel("Передбачений клас")
    plt.show()


def predict_image(image_path, model, class_indices, target_size=(299, 299)):
    """Аналізує та класифікує окреме нове зображення за допомогою навченої мережі."""
    try:
        img = Image.open(image_path).resize(target_size)
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Додаємо вимірність батчу

        prediction = model.predict(img_array)[0][0]

        # Інвертуємо словник міток класів ({0: 'cats', 1: 'dogs'})
        labels = {v: k for k, v in class_indices.items()}
        predicted_class = labels[1] if prediction > 0.5 else labels[0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        plt.imshow(img)
        plt.title(f"Прогноз: {predicted_class} ({confidence * 100:.2f}%)")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Помилка при обробці зображення '{image_path}': {e}")

# ==========================================
# ГОЛОВНИЙ ПОТІК ВИКОНАННЯ (MAIN EXECUTION)
# ==========================================
if __name__ == "__main__":
    # Крок 1: Підготовка папок та завантаження даних (розкоментуйте за потреби)
    # download_and_prepare_dataset(dataset_base_dir="dataset")

    # Перевірка наявності даних перед запуском генераторів
    if not (os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR)):
        print(f"Помилка: Директорії даних не знайдені. Створіть папки або запустіть 'download_and_prepare_dataset()'.")
    else:
        # Крок 2: Ініціалізація генераторів
        train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)

        # Крок 3: Створення кастомної моделі Inception-v3
        inception_v2_v3_model = build_custom_inception_v3()
        print("\nАрхітектура Inception-v3 успішно побудована пошарово.")

        # Крок 4: Навчання або завантаження моделі
        # Виправлено виклик функції: видалено зайвий аргумент validation_data
        inception_v2_v3_model = train_or_load_model(
            model=inception_v2_v3_model,
            train_generator=train_gen,
            val_generator=val_gen,
            model_path=MODEL_FILE_NAME,
            epochs=10
        )

        # Крок 5: Оцінка якості (Звіти та Матриця помилок)
        evaluate_model(inception_v2_v3_model, val_gen)

        # Крок 6: Тестування на конкретному файлі (вкажіть шлях до тестового фото за наявності)
        # predict_image('test_image.jpg', inception_v2_v3_model, train_gen.class_indices)

        # Крок 7: Додаткове завдання — Оптимізація (Порівняння з Mini-Inception)
        mini_inception_model = build_mini_inception()

        print("\n=== ПОРІВНЯННЯ ОБЧИСЛЮВАЛЬНОЇ СКЛАДНОСТІ ARCHITECTURES ===")
        print("\n1. Специфікація повної кастомної Inception-v3:")
        inception_v2_v3_model.summary()

        print("\n2. Специфікація оптимізованої Mini-Inception:")
        mini_inception_model.summary()
        
        print("\nВисновок: Mini-Inception оперує значно меншою кількістю параметрів, зменшуючи")
        print("використання оперативної пам'яті (VRAM) та прискорюючи крок навчання, проте")
        print("може демонструвати нижчу узагальнюючу здатність на складних вибірках.")