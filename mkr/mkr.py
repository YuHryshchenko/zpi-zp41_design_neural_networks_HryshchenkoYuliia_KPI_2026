# Uncomment on Kaggle
# !pip install ultralytics

# ============================================================
# МКР: ДЕТЕКЦІЯ ОБ'ЄКТУ НА ВІДЕО З ВИКОРИСТАННЯМ YOLOv11
# Предмет: Проектування та реалізація програмних систем з НМ
#
# Завдання:
#   1. Створити датасет у форматі YOLO (зображення + анотації bbox)
#      Зібрати зображення та розмітити (labelImg / roboflow / синтетична
#      генерація для автономної роботи скрипту)
#   2. Розмножити датасет бібліотекою albumentations — мінімум 7 методів
#   3. Навчити нейронну мережу YOLOv11 (бібліотека ultralytics)
#   4. Опрацювати відео навченою НМ, знайдений об'єкт взяти в рамку,
#      зберегти відео
#
# Вимоги до середовища:
#   pip install ultralytics albumentations opencv-python pillow
#          matplotlib seaborn numpy tqdm scipy pandas
#
# Рекомендована структура папок після запуску:
#   mkr/
#     dataset_raw/        ← оригінальні зображення + YOLO-мітки
#     dataset_augmented/  ← аугментований датасет
#     runs/               ← результати навчання YOLOv11
#     video_output/       ← відео з рамками детекції
# ============================================================

import os
import cv2
import sys
import time
import random
import shutil
import datetime
import urllib.request
import zipfile
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.ndimage import median_filter
import yaml

# --- Бібліотека аугментації (з підтримкою обмежувальних рамок) ---
import albumentations as A
from albumentations.core.composition import Compose as ACompose

# --- Бібліотека YOLOv11 (ultralytics) ---
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("[ПОПЕРЕДЖЕННЯ] Бібліотека ultralytics не встановлена!")
    print("  Встановіть: pip install ultralytics")
    ULTRALYTICS_AVAILABLE = False

# ============================================================
# РОЗДІЛ 0: ГЛОБАЛЬНІ КОНСТАНТИ ТА НАЛАШТУВАННЯ
# ============================================================

# --- Назва об'єкту для детекції ---
# Оберіть один варіант:
#   'logo'  — логотип (тематика Лабораторної №6)
#   'pet'   — улюбленець (тематика Лабораторної №5)
DETECTION_TARGET = 'logo'          # Назва класу що детектується
CLASS_NAMES = [DETECTION_TARGET]   # Список класів YOLOv11

# --- Кількість синтетичних зображень для генерації ---
NUM_TRAIN_IMAGES  = 350   # тренувальна вибірка
NUM_VAL_IMAGES    = 100   # валідаційна вибірка
NUM_TEST_IMAGES   = 50   # тестова вибірка

# --- Розміри зображень ---
IMAGE_SIZE = (640, 640)   # стандартний вхід YOLOv11

# --- Параметри навчання ---
EPOCHS             = 15           # кількість епох навчання
PATIENCE_EPOCHS    = 10           # кількість епох для зупинки якщо нема покращення
EPOCHS_SAVE_PERIOD = 5            # зберігати checkpoint кожні EPOCHS_SAVE_PERIOD епох
BATCH_SIZE         = 16           # розмір батчу
LEARNING_RATE      = 0.01         # початковий lr (автоматично регулюється)
YOLO_MODEL_NAME    = 'yolo11n.pt' # nano-версія (найшвидша); для кращої точності: yolo11s.pt

# --- Поріг впевненості при детекції ---
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD        = 0.5

# --- Аугментація: кількість копій кожного зображення ---
AUGMENTATION_COPIES = 2   # кожне зображення → 2 аугментовані варіанти

# ============================================================
# --- 1.1 Визначення та налаштування середовища (Kaggle / Local) ---
# ============================================================

CURRENT_LAB = "mkr"

def is_kaggle() -> bool:
    """Повертає True якщо скрипт запущено в середовищі Kaggle Kernel."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if is_kaggle():
    print("Running on Kaggle")
    # На Kaggle всі файли зберігаються прямо у поточній директорії (/kaggle/working/)
    BASE_DIR = Path(".")
else:
    print("Running locally")
    # Локально: всі артефакти МКР зберігаються у підпапці mkr/
    ABSOLUTE_PATH = os.getcwd()
    BASE_DIR = Path(ABSOLUTE_PATH) / CURRENT_LAB

# --- Посилання на архів з логотипами Toyota (GitHub) ---
# Архів містить 180 найкращих зображень логотипу у папці cleaned-img/toyota/
GITHUB_BASE_URL     = (
    "https://github.com/YuHryshchenko/"
    "zpi-zp41_design_neural_networks_HryshchenkoYuliia_KPI_2026"
    "/raw/main/mkr/"
)
CLEANED_IMG_ZIP_URL     = GITHUB_BASE_URL + "cleaned-img.zip"
SOURCE_VIDEO_URL        = GITHUB_BASE_URL + "input_video.mp4"
KAGGLE_WORKING_DIR      = "/kaggle/working/"
VIDEO_INPUT_FILENAME    = "input_video.mp4"

# --- Шляхи до архіву та розпакованих логотипів ---
# Архів знаходиться поряд зі скриптом mkr.py (поточна директорія)
WORKING_DIR        = Path(".")                          # завжди поточна директорія
ZIP_LOCAL_PATH     = WORKING_DIR / "cleaned-img.zip"   # cleaned-img.zip поруч зі скриптом
EXTRACTED_DIR      = WORKING_DIR / "cleaned-img"        # папка після розпакування
TOYOTA_LOGO_DIR    = EXTRACTED_DIR / "toyota"           # 180 логотипів Toyota

# --- Базова директорія проєкту та похідні шляхи ---
AUGMENTED_DATASET_DIR = BASE_DIR / "dataset_augmented"
ASSETS_DIR            = BASE_DIR / "assets"
VIDEO_OUTPUT_DIR      = BASE_DIR / "video_output"
RUNS_DIR              = BASE_DIR / "runs"
DATA_YAML_PATH        = BASE_DIR / "data.yaml"
VIDEO_INPUT_PATH      = BASE_DIR / VIDEO_INPUT_FILENAME
VIDEO_OUTPUT_PATH     = VIDEO_OUTPUT_DIR / "output_detected.mp4"
DATASET_SAMPLES_PREVIEW = BASE_DIR / "dataset_samples_preview.png"

# --- Структура датасету (split → subdir) ---
SPLITS = ['train', 'val', 'test']

import os
import urllib.request
import zipfile

# ===========================================================
# ЗАВАНТАЖЕННЯ ДАНИХ (АВТОМАТИЧНЕ)
# ===========================================================

# Шляхи для збереження завантажених файлів
ZIP_PATH = "cleaned-img.zip"
VIDEO_INPUT_FILENAME = "input_video.mp4"
EXTRACT_DIR = "cleaned-img" # Після розпакування тут буде папка toyota

# Завантаження та розпакування архіву з зображеннями (логотипи Toyota)
if not os.path.exists(EXTRACT_DIR):
    if not os.path.exists(ZIP_PATH):
        print(f"[*] Завантаження архіву з зображеннями: {ZIP_PATH}...")
        urllib.request.urlretrieve(CLEANED_IMG_ZIP_URL, ZIP_PATH)
        print("[+] Архів успішно завантажено.")
    
    print(f"[*] Розпакування {ZIP_PATH} у поточну директорію...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("[+] Розпакування завершено.")
else:
    print(f"[*] Директорія '{EXTRACT_DIR}' вже існує. Пропускаємо завантаження архіву.")

# Завантаження вхідного відео
if not os.path.exists(VIDEO_INPUT_FILENAME):
    print(f"[*] Завантаження вхідного відео: {VIDEO_INPUT_FILENAME}...")
    urllib.request.urlretrieve(SOURCE_VIDEO_URL, VIDEO_INPUT_FILENAME)
    print("[+] Відео успішно завантажено.")
else:
    print(f"[*] Відео '{VIDEO_INPUT_FILENAME}' вже існує. Пропускаємо завантаження.")

# Оновлюємо шлях до директорії з сирими зображеннями для подальшого використання у коді
RAW_DATASET_DIR = Path(EXTRACT_DIR) / "toyota"

# ============================================================
# РОЗДІЛ 1: ПІДГОТОВКА ДИРЕКТОРІЙ ТА ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================

def create_directory_structure() -> None:
    """
    Створює всі необхідні директорії проєкту.
    Структура відповідає вимогам формату YOLOv8/v11:
      dataset/
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/
    """
    print("\n[1/5] Створення структури директорій...")
    dirs_to_create = [
        ASSETS_DIR,
        VIDEO_OUTPUT_DIR,
        RUNS_DIR,
    ]

    # ──── Якщо хочемо видалити і перетренувати моделі на Kaggle ────
    if os.path.exists(AUGMENTED_DATASET_DIR):
      # shutil.rmtree(AUGMENTED_DATASET_DIR)
      remove_folder_contents(AUGMENTED_DATASET_DIR)
      os.rmdir(AUGMENTED_DATASET_DIR)

    if os.path.exists(DATASET_SAMPLES_PREVIEW):
      os.unlink(DATASET_SAMPLES_PREVIEW)

    # Для кожного split створюємо папки images/ та labels/
    for split in SPLITS:
        for dataset_dir in [RAW_DATASET_DIR, AUGMENTED_DATASET_DIR]:
            dirs_to_create.append(dataset_dir / split / "images")
            dirs_to_create.append(dataset_dir / split / "labels")

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    print(f"  ✓ Структуру директорій створено у '{BASE_DIR}'")


def yolo_bbox_from_paste(
    bg_width: int, bg_height: int,
    paste_x: int, paste_y: int,
    logo_width: int, logo_height: int
) -> tuple[float, float, float, float]:
    """
    Обчислює нормалізовані координати обмежувальної рамки у форматі YOLO.

    YOLO-формат: (x_center, y_center, width, height) — всі значення в [0, 1].
    Це центр об'єкта та його розміри, нормалізовані відносно розмірів зображення.

    Args:
        bg_width, bg_height: розміри фонового зображення (пікселі)
        paste_x, paste_y: координати лівого верхнього кута вставки (пікселі)
        logo_width, logo_height: розміри логотипу після вставки (пікселі)

    Returns:
        (x_center_norm, y_center_norm, width_norm, height_norm) у діапазоні [0, 1]
    """
    # Центр рамки в абсолютних координатах
    x_center_abs = paste_x + logo_width  / 2
    y_center_abs = paste_y + logo_height / 2

    # Нормалізація відносно розмірів зображення
    x_center_norm = x_center_abs / bg_width
    y_center_norm = y_center_abs / bg_height
    width_norm    = logo_width   / bg_width
    height_norm   = logo_height  / bg_height

    # Обрізаємо до [0, 1] для уникнення значень поза межами
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm    = max(0.001, min(1.0, width_norm))
    height_norm   = max(0.001, min(1.0, height_norm))

    return x_center_norm, y_center_norm, width_norm, height_norm


def save_yolo_label(label_path: Path, class_id: int,
                    x_center: float, y_center: float,
                    width: float, height: float) -> None:
    """
    Зберігає анотацію в форматі YOLO у текстовий файл.
    Кожний рядок файлу: <class_id> <x_center> <y_center> <width> <height>
    """
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}\n")


# ============================================================
# РОЗДІЛ 2: СИНТЕТИЧНА ГЕНЕРАЦІЯ ДАТАСЕТУ У ФОРМАТІ YOLO
# ============================================================

def generate_synthetic_logo(size: int = 80) -> Image.Image:
    """
    Генерує синтетичний логотип-заглушку (кольоровий геометричний значок).
    У реальному проєкті замість цієї функції слід використовувати
    реальні зображення логотипу, зібрані вручну.

    Args:
        size: розмір логотипу в пікселях (width = height = size)

    Returns:
        Зображення логотипу у режимі RGBA (з прозорим фоном)
    """
    # Створюємо прозоре зображення
    logo = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)

    # Генеруємо випадковий колір логотипу (яскравий, насичений)
    hue = random.randint(0, 360)
    r = int(200 + 55 * math.cos(math.radians(hue)))
    g = int(200 + 55 * math.cos(math.radians(hue + 120)))
    b = int(200 + 55 * math.cos(math.radians(hue + 240)))
    color = (r % 256, g % 256, b % 256, 255)

    # Малюємо характерну форму: еліпс + прямокутник всередині (як значок)
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin],
                 fill=color, outline=(30, 30, 30, 255), width=3)

    # Малюємо літеру у центрі для розрізнення
    center = size // 2
    letter_size = max(size // 3, 12)
    try:
        # Намагаємось завантажити системний шрифт
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  letter_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    draw.text(
        (center - letter_size // 3, center - letter_size // 2),
        letter,
        fill=(255, 255, 255, 255),
        font=font
    )
    return logo


def generate_synthetic_background(width: int, height: int) -> Image.Image:
    """
    Генерує синтетичний фон (градієнт + шум + геометрія).
    Імітує реальні умови зйомки: вулиця, офіс, природа тощо.

    Args:
        width, height: розміри фонового зображення

    Returns:
        Фонове зображення у режимі RGB
    """
    # Базовий градієнтний фон
    bg_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Генеруємо кольоровий градієнт
    top_color    = np.array([random.randint(50, 200) for _ in range(3)])
    bottom_color = np.array([random.randint(50, 200) for _ in range(3)])

    for y in range(height):
        alpha = y / height
        row_color = ((1 - alpha) * top_color + alpha * bottom_color).astype(np.uint8)
        bg_array[y, :] = row_color

    # Додаємо текстурний шум для реалістичності
    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
    bg_array = np.clip(bg_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Малюємо кілька геометричних фігур-«перешкод» для ускладнення задачі
    bg_img = Image.fromarray(bg_array, 'RGB')
    draw = ImageDraw.Draw(bg_img)

    num_shapes = random.randint(3, 8)
    for _ in range(num_shapes):
        shape_color = tuple(random.randint(0, 255) for _ in range(3))
        x0 = random.randint(0, width  - 50)
        y0 = random.randint(0, height - 50)
        x1 = x0 + random.randint(20, 100)
        y1 = y0 + random.randint(20, 100)
        shape_type = random.choice(['rectangle', 'ellipse', 'line'])
        if shape_type == 'rectangle':
            draw.rectangle([x0, y0, x1, y1], fill=shape_color, outline=(0, 0, 0))
        elif shape_type == 'ellipse':
            draw.ellipse([x0, y0, x1, y1], fill=shape_color, outline=(0, 0, 0))
        else:
            draw.line([x0, y0, x1, y1], fill=shape_color, width=2)

    return bg_img


def generate_single_sample(
    split: str,
    idx: int,
    dataset_dir: Path,
    logo_files: list | None = None,
    background_files: list | None = None
) -> None:
    """
    Генерує одне зображення датасету з YOLO-анотацією.

    Алгоритм:
    1. Завантажуємо або генеруємо фоновий знімок
    2. Завантажуємо або синтезуємо зображення логотипу
    3. Застосовуємо аугментацію до самого логотипу (поворот, масштаб, прозорість)
    4. Вставляємо логотип на фон у випадкову позицію
    5. Обчислюємо координати bbox у форматі YOLO
    6. Зберігаємо зображення та файл з анотацією

    Args:
        split: 'train', 'val' або 'test'
        idx: унікальний індекс зображення в рамках split
        dataset_dir: базова директорія датасету
        logo_files: список шляхів до реальних логотипів (або None → синтез)
        background_files: список шляхів до фонів (або None → синтез)
    """
    img_dir   = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"

    # --- Крок 2a: Завантаження або генерація фону ---
    if background_files:
        bg = Image.open(random.choice(background_files)).convert('RGB')
        bg = bg.resize(IMAGE_SIZE)
    else:
        bg = generate_synthetic_background(*IMAGE_SIZE)
        bg = bg.resize(IMAGE_SIZE)

    bg_w, bg_h = bg.size

    # --- Крок 2b: Завантаження або генерація логотипу ---
    logo_size = random.randint(60, int(min(bg_w, bg_h) * 0.35))
    if logo_files:
        try:
            logo = Image.open(random.choice(logo_files)).convert('RGBA')
            logo = logo.resize((logo_size, logo_size))
        except Exception:
            logo = generate_synthetic_logo(logo_size)
    else:
        logo = generate_synthetic_logo(logo_size)

    # --- Крок 2c: Аугментація самого логотипу перед вставкою ---
    # Поворот
    angle = random.uniform(-25, 25)
    logo = logo.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Масштабування (може збільшитись через поворот, обрізаємо до допустимого)
    max_allowed = int(min(bg_w, bg_h) * 0.4)
    if logo.width > max_allowed or logo.height > max_allowed:
        logo.thumbnail((max_allowed, max_allowed), Image.LANCZOS)

    logo_w, logo_h = logo.size

    # Регулювання прозорості
    alpha_factor = random.uniform(0.65, 1.0)
    if logo.mode == 'RGBA':
        r, g, b, a = logo.split()
        a = a.point(lambda p: int(p * alpha_factor))
        logo = Image.merge('RGBA', (r, g, b, a))

    # --- Крок 2d: Вставка логотипу в випадкову позицію на фоні ---
    max_paste_x = max(0, bg_w - logo_w)
    max_paste_y = max(0, bg_h - logo_h)
    paste_x = random.randint(0, max_paste_x)
    paste_y = random.randint(0, max_paste_y)

    # Зберігаємо фон у режимі RGBA для вставки з альфа-каналом
    bg_rgba = bg.convert('RGBA')
    if logo.mode == 'RGBA':
        bg_rgba.paste(logo, (paste_x, paste_y), logo)
    else:
        bg_rgba.paste(logo, (paste_x, paste_y))

    final_img = bg_rgba.convert('RGB')

    # --- Крок 2e: Обчислення bbox у форматі YOLO ---
    x_c, y_c, w_n, h_n = yolo_bbox_from_paste(
        bg_w, bg_h, paste_x, paste_y, logo_w, logo_h
    )

    # --- Крок 2f: Збереження зображення та анотації ---
    img_filename   = img_dir   / f"{split}_{idx:05d}.jpg"
    label_filename = label_dir / f"{split}_{idx:05d}.txt"

    final_img.save(str(img_filename), "JPEG", quality=92)
    save_yolo_label(label_filename, class_id=0,
                    x_center=x_c, y_center=y_c,
                    width=w_n, height=h_n)


def generate_dataset(
    logo_files: list | None = None,
    background_files: list | None = None
) -> None:
    """
    Повна генерація датасету у форматі YOLO для всіх split-розбиттів.

    Пропорції розбиття:
        Тренування   : NUM_TRAIN_IMAGES зображень
        Валідація    : NUM_VAL_IMAGES зображень
        Тестування   : NUM_TEST_IMAGES зображень

    Args:
        logo_files: список шляхів до зображень логотипу (PNG/JPEG з прозорістю)
        background_files: список шляхів до фонових зображень
    """
    print("\n[2/5] Генерація датасету у форматі YOLO...")

    split_counts = {
        'train': NUM_TRAIN_IMAGES,
        'val':   NUM_VAL_IMAGES,
        'test':  NUM_TEST_IMAGES,
    }

    for split, count in split_counts.items():
        # Перевіряємо, чи вже згенеровано достатню кількість зображень
        existing = list((RAW_DATASET_DIR / split / "images").glob("*.jpg"))
        if len(existing) >= count:
            print(f"  ✓ {split}: вже існує {len(existing)} зображень — пропускаємо")
            continue

        print(f"  Генерую {count} зображень для '{split}'...")
        for idx in tqdm(range(count), desc=f"  {split:5s}", ncols=80):
            generate_single_sample(
                split=split, idx=idx,
                dataset_dir=RAW_DATASET_DIR,
                logo_files=logo_files,
                background_files=background_files
            )

    print(f"  ✓ Датасет успішно згенеровано у '{RAW_DATASET_DIR}'")


def load_real_images(
    logo_dir: str | None = None,
    background_dir: str | None = None
) -> tuple[list | None, list | None]:
    """
    Завантажує реальні зображення логотипів та фонів зі вказаних директорій.
    Якщо директорії не вказані — повертає None (буде синтетична генерація).

    Для використання реальних даних:
      1. Зберіть 50-200 зображень логотипу (PNG з прозорим фоном або JPEG)
      2. Зберіть 50-200 фонових зображень без логотипу
      3. Вкажіть шляхи до директорій

    Returns:
        Кортеж (logo_files, background_files) або (None, None)
    """
    logo_files       = None
    background_files = None

    VALID_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    if logo_dir and os.path.isdir(logo_dir):
        logo_files = [
            str(p) for p in Path(logo_dir).iterdir()
            if p.suffix.lower() in VALID_EXTS
        ]
        print(f"  ✓ Завантажено {len(logo_files)} реальних зображень логотипу")

    if background_dir and os.path.isdir(background_dir):
        background_files = [
            str(p) for p in Path(background_dir).iterdir()
            if p.suffix.lower() in VALID_EXTS
        ]
        print(f"  ✓ Завантажено {len(background_files)} фонових зображень")

    return logo_files, background_files


def visualize_dataset_samples(n: int = 6) -> None:
    """
    Візуалізує n випадкових зображень тренувального датасету
    разом з накладеними bounding boxes для контролю якості анотацій.
    """
    print("  Візуалізація зразків датасету...")
    train_img_dir   = RAW_DATASET_DIR / "train" / "images"
    train_label_dir = RAW_DATASET_DIR / "train" / "labels"

    image_files = sorted(train_img_dir.glob("*.jpg"))
    if not image_files:
        print("  [ПОПЕРЕДЖЕННЯ] Зображення не знайдено для візуалізації")
        return

    samples = random.sample(image_files, min(n, len(image_files)))

    fig, axes = plt.subplots(2, n // 2, figsize=(15, 7))
    axes = axes.flatten()

    for i, img_path in enumerate(samples):
        label_path = train_label_dir / (img_path.stem + ".txt")
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, xc, yc, bw, bh = map(float, parts)
                        # Перетворення нормалізованих координат у пікселі
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        # Малюємо bbox
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, DETECTION_TARGET, (x1, max(0, y1 - 5)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        axes[i].imshow(img)
        axes[i].set_title(f"Зразок {i + 1}", fontsize=9)
        axes[i].axis('off')

    plt.suptitle("Зразки датасету з YOLO-анотаціями (зелена рамка)", fontsize=12)
    plt.tight_layout()
    save_path = BASE_DIR / DATASET_SAMPLES_PREVIEW
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Збережено у '{save_path}'")


# ============================================================
# РОЗДІЛ 3: АУГМЕНТАЦІЯ ДАТАСЕТУ (ALBUMENTATIONS, 7+ МЕТОДІВ)
# ============================================================

def build_augmentation_pipeline() -> ACompose:
    """
    Будує конвеєр аугментації за допомогою бібліотеки albumentations.

    Використовується мінімум 7 методів аугментації з підтримкою
    трансформації обмежувальних рамок (bbox_params).

    Застосовані методи:
      1.  HorizontalFlip            — горизонтальне відображення
      2.  VerticalFlip              — вертикальне відображення
      3.  RandomBrightnessContrast  — випадкова яскравість та контраст
      4.  HueSaturationValue        — зміна відтінку, насиченості, значення
      5.  GaussNoise                — гаусівський шум
      6.  MotionBlur                — розмиття руху (імітація руху камери)
      7.  ShiftScaleRotate          — зсув, масштабування та поворот зображення
      8.  CoarseDropout             — випадкове видалення прямокутних областей
      9.  CLAHE                     — адаптивне вирівнювання гістограми
      10. RandomGamma               — випадкова гама-корекція
      11. Perspective               — випадкова перспективна деформація

    Returns:
        Об'єкт albumentations.Compose із налаштованими трансформаціями
        та BboxParams для коректного перетворення bbox
    """
    pipeline = A.Compose(
        [
            # 1. Горизонтальне відображення: 50% ймовірність
            A.HorizontalFlip(p=0.5),

            # 2. Вертикальне відображення: 20% ймовірність
            A.VerticalFlip(p=0.2),

            # 3. Яскравість та контраст: імітація різних умов освітлення
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),

            # 4. Відтінок, насиченість, значення: різниця кольорів камер
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),

            # 5. Гаусівський шум: імітація сенсорного шуму камери
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0,
                p=0.4
            ),

            # 6. Розмиття руху: імітація тремтіння камери або руху об'єкта
            A.MotionBlur(
                blur_limit=(3, 7),
                p=0.3
            ),

            # 7. Зсув + масштабування + поворот: головний геометричний аугментатор
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6
            ),

            # 8. Випадкове видалення прямокутних областей (Cutout/CoarseDropout):
            #    навчає мережу розпізнавати частково закриті об'єкти
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(20, 60),
                hole_width_range=(20, 60),
                fill_value=0,
                p=0.3
            ),

            # 9. Адаптивне вирівнювання гістограми: покращення локального контрасту
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),

            # 10. Гама-корекція: симуляція різних налаштувань монітора/камери
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),

            # 11. Перспективна деформація: логотип на кутових поверхнях
            A.Perspective(
                scale=(0.02, 0.06),
                keep_size=True,
                p=0.3
            ),
        ],
        # Параметри для автоматичного трансформування bbox разом із зображенням
        bbox_params=A.BboxParams(
            format='yolo',            # формат: (x_center, y_center, width, height) нормалізовані
            min_area=100,             # мінімальна площа bbox після аугментації (пікселі²)
            min_visibility=0.3,       # мінімальна частка bbox що залишилась видимою
            label_fields=['class_labels']  # назва поля з мітками класів
        )
    )
    return pipeline


def augment_dataset(source_dir: Path, target_dir: Path) -> None:
    """
    Виконує аугментацію всього датасету:
      - Читає зображення та YOLO-анотації з source_dir
      - Застосовує конвеєр із 7+ методів albumentations
      - Зберігає AUGMENTATION_COPIES копій кожного зображення у target_dir
      - Оригінальне зображення також копіюється до аугментованого датасету

    Args:
        source_dir: шлях до оригінального датасету (RAW_DATASET_DIR)
        target_dir: шлях для збереження аугментованого датасету
    """
    print("\n[3/5] Аугментація датасету (albumentations, 7+ методів)...")

    augmentation_pipeline = build_augmentation_pipeline()
    total_generated = 0

    for split in SPLITS:
        src_img_dir   = source_dir / split / "images"
        src_label_dir = source_dir / split / "labels"
        dst_img_dir   = target_dir / split / "images"
        dst_label_dir = target_dir / split / "labels"

        image_files = sorted(src_img_dir.glob("*.jpg"))
        if not image_files:
            print(f"  [ПОПЕРЕДЖЕННЯ] Немає зображень у '{src_img_dir}'")
            continue

        print(f"  Аугментація '{split}': {len(image_files)} оригінальних зображень"
              f" × {AUGMENTATION_COPIES + 1} = "
              f"{len(image_files) * (AUGMENTATION_COPIES + 1)} вихідних")

        for img_path in tqdm(image_files, desc=f"  {split:5s}", ncols=80):
            label_path = src_label_dir / (img_path.stem + ".txt")

            # Завантаження зображення
            img = np.array(Image.open(img_path).convert('RGB'))
            h, w = img.shape[:2]

            # Зчитування YOLO-анотацій
            bboxes        = []
            class_labels  = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])
                            bboxes.append([xc, yc, bw, bh])
                            class_labels.append(cls_id)

            # --- Копіюємо оригінальне зображення у цільову директорію ---
            orig_img_dst   = dst_img_dir   / img_path.name
            orig_label_dst = dst_label_dir / label_path.name
            if not orig_img_dst.exists():
                shutil.copy2(img_path, orig_img_dst)
            if label_path.exists() and not orig_label_dst.exists():
                shutil.copy2(label_path, orig_label_dst)

            # --- Генеруємо AUGMENTATION_COPIES аугментованих копій ---
            for copy_idx in range(AUGMENTATION_COPIES):
                try:
                    result = augmentation_pipeline(
                        image=img,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    aug_img    = result['image']
                    aug_bboxes = result['bboxes']
                    aug_labels = result['class_labels']
                except Exception as e:
                    # Якщо аугментація провалилась — просто пропускаємо
                    continue

                # Назва аугментованого файлу
                aug_stem  = f"{img_path.stem}_aug{copy_idx:02d}"
                aug_img_path   = dst_img_dir   / f"{aug_stem}.jpg"
                aug_label_path = dst_label_dir / f"{aug_stem}.txt"

                # Збереження аугментованого зображення
                Image.fromarray(aug_img).save(
                    str(aug_img_path), "JPEG", quality=90
                )

                # Збереження аугментованих анотацій
                with open(aug_label_path, 'w') as f:
                    for (xc, yc, bw, bh), cls_id in zip(aug_bboxes, aug_labels):
                        f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                total_generated += 1

        # Підрахунок підсумку для split
        final_count = len(list(dst_img_dir.glob("*.jpg")))
        print(f"  ✓ {split}: {final_count} зображень у цільовому датасеті")

    print(f"  ✓ Аугментацію завершено: {total_generated} нових зображень згенеровано")


def visualize_augmentation_comparison(n_pairs: int = 4) -> None:
    """
    Порівнює оригінальні зображення з їх аугментованими варіантами.
    Допомагає впевнитись, що аугментація відбулась коректно.
    """
    print("  Візуалізація результатів аугментації...")
    orig_dir = RAW_DATASET_DIR / "train" / "images"
    aug_dir  = AUGMENTED_DATASET_DIR / "train" / "images"

    orig_files = sorted(orig_dir.glob("*.jpg"))[:n_pairs]
    if not orig_files:
        return

    fig, axes = plt.subplots(n_pairs, 2, figsize=(10, n_pairs * 2.5))
    if n_pairs == 1:
        axes = [axes]

    for i, orig_path in enumerate(orig_files):
        orig_img = np.array(Image.open(orig_path))

        # Шукаємо першу аугментовану версію цього зображення
        aug_pattern = f"{orig_path.stem}_aug00.jpg"
        aug_path = aug_dir / aug_pattern
        if aug_path.exists():
            aug_img = np.array(Image.open(aug_path))
        else:
            aug_img = orig_img.copy()

        axes[i][0].imshow(orig_img)
        axes[i][0].set_title(f"Оригінал {i + 1}", fontsize=9)
        axes[i][0].axis('off')

        axes[i][1].imshow(aug_img)
        axes[i][1].set_title(f"Аугментація {i + 1}", fontsize=9)
        axes[i][1].axis('off')

    plt.suptitle("Порівняння: оригінал ↔ аугментація (albumentations)", fontsize=11)
    plt.tight_layout()
    save_path = BASE_DIR / "augmentation_comparison.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Збережено у '{save_path}'")

# ============================================================
def remove_folder_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                remove_folder_contents(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(e)

# ============================================================
# РОЗДІЛ 4: КОНФІГУРАЦІЯ YOLO ТА ПІДГОТОВКА data.yaml
# ============================================================

def create_data_yaml(dataset_dir: Path) -> Path:
    """
    Створює конфігураційний файл data.yaml для YOLOv11.
    Цей файл описує структуру датасету: шляхи, кількість класів, назви класів.

    Формат файлу:
        path: /абсолютний/шлях/до/датасету
        train: train/images
        val:   val/images
        test:  test/images
        nc: 1
        names: ['logo']

    Args:
        dataset_dir: шлях до датасету (може бути raw або augmented)

    Returns:
        Шлях до створеного data.yaml
    """
    yaml_content = {
        'path' : str(dataset_dir.resolve()),
        'train': 'train/images',
        'val'  : 'val/images',
        'test' : 'test/images',
        'nc'   : len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    with open(DATA_YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"  ✓ data.yaml збережено: '{DATA_YAML_PATH}'")
    print(f"     Датасет: {dataset_dir.resolve()}")
    print(f"     Класи  : {CLASS_NAMES}")
    return DATA_YAML_PATH


def count_dataset_statistics(dataset_dir: Path) -> dict:
    """
    Рахує та виводить статистику датасету:
      - Кількість зображень по split
      - Кількість анотацій (bbox) по split
      - Середній розмір bbox

    Args:
        dataset_dir: базова директорія датасету

    Returns:
        Словник зі статистикою
    """
    stats = {}
    for split in SPLITS:
        img_dir   = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"

        n_images   = len(list(img_dir.glob("*.jpg")))
        n_labels   = len(list(label_dir.glob("*.txt")))
        n_bboxes   = 0
        bbox_sizes = []

        for label_file in label_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        n_bboxes += 1
                        bbox_sizes.append((float(parts[3]), float(parts[4])))

        avg_w = np.mean([s[0] for s in bbox_sizes]) if bbox_sizes else 0
        avg_h = np.mean([s[1] for s in bbox_sizes]) if bbox_sizes else 0

        stats[split] = {
            'images'  : n_images,
            'labels'  : n_labels,
            'bboxes'  : n_bboxes,
            'avg_bbox': (avg_w, avg_h)
        }

    return stats


def print_dataset_statistics(dataset_dir: Path, label: str = "Датасет") -> None:
    """Виводить таблицю статистики датасету."""
    stats = count_dataset_statistics(dataset_dir)
    print(f"\n  {'─' * 55}")
    print(f"  {label}")
    print(f"  {'─' * 55}")
    print(f"  {'Split':<10} {'Зображень':>10} {'Файлів міток':>13} {'Bbox':>8}  Середній bbox (w×h)")
    print(f"  {'─' * 55}")
    for split, s in stats.items():
        aw, ah = s['avg_bbox']
        print(f"  {split:<10} {s['images']:>10} {s['labels']:>13} "
              f"{s['bboxes']:>8}  {aw:.3f} × {ah:.3f}")
    print(f"  {'─' * 55}")


# ============================================================
# РОЗДІЛ 5: НАВЧАННЯ YOLOv11 (ULTRALYTICS)
# ============================================================

def train_yolov11(data_yaml: Path) -> tuple:
    """
    Навчає нейронну мережу YOLOv11 на підготовленому датасеті.

    YOLOv11 — найновіша версія архітектури YOLO від Ultralytics.
    Відмінності від попередніх версій:
      - Покращена архітектура C3k2 та SPPF блоки
      - Більш ефективна обробка ознак за рахунок C2PSA
      - Зменшена кількість параметрів при збереженні точності

    Доступні розміри моделей:
      yolo11n.pt — nano   (2.6M параметрів, найшвидша)
      yolo11s.pt — small  (9.4M параметрів)
      yolo11m.pt — medium (20.1M параметрів)
      yolo11l.pt — large  (25.3M параметрів)
      yolo11x.pt — extra  (56.9M параметрів, найточніша)

    Args:
        data_yaml: шлях до файлу data.yaml з описом датасету

    Returns:
        (model, results) — навчена модель та результати навчання
    """
    if not ULTRALYTICS_AVAILABLE:
        print("  [ПОМИЛКА] ultralytics не встановлено. Навчання неможливе.")
        print("  Встановіть: pip install ultralytics")
        return None, None

    print(f"\n[4/5] Навчання YOLOv11 ({YOLO_MODEL_NAME})...")
    print(f"  Конфігурація:")
    print(f"    Модель      : {YOLO_MODEL_NAME}")
    print(f"    Датасет     : {data_yaml}")
    print(f"    Епохи       : {EPOCHS}")
    print(f"    Батч-розмір : {BATCH_SIZE}")
    print(f"    Розмір входу: {IMAGE_SIZE[0]}px")

    # Завантажуємо попередньо навчену модель YOLOv11
    # (з вагами ImageNet для кращого перенесеного навчання)
    model = YOLO(YOLO_MODEL_NAME)

    # --- Запуск навчання ---
    # Параметр pretrained=True дозволяє використовувати ваги ImageNet

    results = model.train(
        data=str(data_yaml),          # шлях до data.yaml
        epochs=EPOCHS,                 # кількість епох
        batch=BATCH_SIZE,              # розмір батчу
        imgsz=IMAGE_SIZE[0],           # розмір вхідного зображення
        lr0=LEARNING_RATE,             # початковий learning rate
        project=str(RUNS_DIR),         # директорія для збереження результатів
        name="yolo11_logo_detection",  # назва експерименту
        save=True,                     # зберігати найкращу та останню модель
        save_period=EPOCHS_SAVE_PERIOD,# зберігати checkpoint кожні EPOCHS_SAVE_PERIOD епох
        patience=PATIENCE_EPOCHS,      # early stopping: зупинка якщо PATIENCE_EPOCHS епох без покращення
        device='0' if _gpu_available() else 'cpu',  # GPU якщо є, інакше CPU
        workers=4,                     # кількість паралельних процесів завантаження даних
        plots=True,                    # генерувати графіки метрик
        val=True,                      # проводити валідацію під час навчання
        augment=True,                  # вбудована аугментація YOLOv11 (Mosaic, MixUp тощо)
        verbose=True                   # детальний вивід
    )

    # =======================================================
    # ЗБЕРЕЖЕННЯ КОПІЇ НАВЧЕНОЇ МОДЕЛІ
    # =======================================================
    # YOLOv11 автоматично зберігає результати глибоко у директорії проєкту.
    # Для виконання вимог МКР та зручності, копіюємо найкращу модель у корінь RUNS_DIR.
    auto_saved_path = RUNS_DIR / "yolo11_logo_detection" / "weights" / "best.pt"
    manual_save_path = RUNS_DIR / "best_trained_model.pt"

    if auto_saved_path.exists():
        print(f"\n[*] Копіювання найкращої моделі у головну директорію...")
        shutil.copy2(auto_saved_path, manual_save_path)
        print(f"  ✓ Модель явно збережено за шляхом: '{manual_save_path}'")
    else:
        print(f"\n  [ПОПЕРЕДЖЕННЯ] Файл {auto_saved_path} не знайдено.")
    # =======================================================

    print(f"\n  ✓ Навчання завершено!")
    print(f"  ✓ Результати збережено у '{RUNS_DIR}/yolo11_logo_detection'")

    return model, results


def _gpu_available() -> bool:
    """Перевіряє наявність GPU через TensorFlow або torch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        pass
    return False


def evaluate_model(model, data_yaml: Path) -> dict:
    """
    Оцінює точність навченої моделі на тестовому датасеті.
    Виводить та візуалізує основні метрики:
      - mAP50   : mean Average Precision при IoU=0.5
      - mAP50-95: mean Average Precision при IoU=0.5...0.95
      - Precision: точність (TP / (TP + FP))
      - Recall:   повнота  (TP / (TP + FN))
      - F1-Score: гармонічне середнє Precision та Recall

    Args:
        model: навчена модель YOLO
        data_yaml: шлях до data.yaml

    Returns:
        Словник з метриками
    """
    if model is None:
        print("  [ПОМИЛКА] Модель не навчена або не завантажена")
        return {}

    print("\n  Оцінка моделі на тестовій вибірці...")
    metrics = model.val(
        data=str(data_yaml),
        split='test',
        imgsz=IMAGE_SIZE[0],
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=True
    )

    # Витягуємо ключові метрики
    results_dict = {
        'mAP50'    : float(metrics.box.map50)   if hasattr(metrics, 'box') else 0.0,
        'mAP50-95' : float(metrics.box.map)     if hasattr(metrics, 'box') else 0.0,
        'precision': float(metrics.box.mp)      if hasattr(metrics, 'box') else 0.0,
        'recall'   : float(metrics.box.mr)      if hasattr(metrics, 'box') else 0.0,
    }
    # F1-Score = 2 * P * R / (P + R)
    p = results_dict['precision']
    r = results_dict['recall']
    results_dict['f1'] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    print(f"\n  {'─' * 40}")
    print(f"  Метрики якості моделі (тестова вибірка):")
    print(f"  {'─' * 40}")
    print(f"  mAP@50      : {results_dict['mAP50']:.4f}")
    print(f"  mAP@50-95   : {results_dict['mAP50-95']:.4f}")
    print(f"  Precision   : {results_dict['precision']:.4f}")
    print(f"  Recall      : {results_dict['recall']:.4f}")
    print(f"  F1-Score    : {results_dict['f1']:.4f}")
    print(f"  {'─' * 40}")

    # --- Візуалізація метрик ---
    visualize_metrics(results_dict)

    return results_dict


def visualize_metrics(metrics: dict) -> None:
    """
    Будує стовпчасту діаграму основних метрик якості моделі.
    """
    labels  = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall', 'F1-Score']
    values  = [
        metrics.get('mAP50', 0),
        metrics.get('mAP50-95', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1', 0),
    ]
    colors = ['#2196F3', '#1976D2', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)

    # Підписи значень над стовпцями
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Значення метрики', fontsize=11)
    ax.set_title(f'Метрики якості YOLOv11 — детекція «{DETECTION_TARGET}»', fontsize=12)
    ax.set_xlabel('Метрика', fontsize=11)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Поріг 0.5')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    save_path = BASE_DIR / "metrics_bar_chart.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Діаграму метрик збережено у '{save_path}'")


def load_best_model() -> object | None:
    """
    Завантажує найкращу збережену модель з директорії runs/.
    Шукає файл best.pt у підпапках RUNS_DIR.

    Returns:
        Завантажена YOLO-модель або None якщо файл не знайдено
    """
    if not ULTRALYTICS_AVAILABLE:
        return None

    # Шукаємо найновіший best.pt файл
    best_models = sorted(
        RUNS_DIR.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if best_models:
        best_path = best_models[0]
        print(f"  ✓ Завантаження найкращої моделі: '{best_path}'")
        return YOLO(str(best_path))

    print("  [ПОПЕРЕДЖЕННЯ] Файл best.pt не знайдено у runs/")
    return None


# ============================================================
# РОЗДІЛ 6: ОБРОБКА ВІДЕО — ДЕТЕКЦІЯ ОБ'ЄКТА З РАМКАМИ
# ============================================================

def create_test_video(output_path: Path, num_frames: int = 300) -> None:
    """
    Створює синтетичне тестове відео для демонстрації детекції.
    Відео містить анімований «логотип», що рухається по фону.

    У реальному проєкті цю функцію замінює реальне відео (mp4/avi).

    Args:
        output_path: шлях для збереження відеофайлу
        num_frames:  кількість кадрів у відео
    """
    print(f"  Генерація тестового відео ({num_frames} кадрів)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w, h   = 640, 480
    fps    = 25.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Розмір та початкова позиція логотипу
    logo_size = 80
    logo_x, logo_y = w // 4, h // 4
    dx, dy = 3, 2  # швидкість руху

    for frame_idx in range(num_frames):
        # Генеруємо фон
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        # Градієнтний фон, що плавно змінюється
        t = frame_idx / num_frames
        bg[:, :, 0] = int(50 + 50 * math.sin(t * math.pi))
        bg[:, :, 1] = int(50 + 50 * math.cos(t * math.pi))
        bg[:, :, 2] = int(100 + 50 * math.sin(t * 2 * math.pi))

        # Рухаємо логотип: відбиваємось від меж кадру
        logo_x += dx
        logo_y += dy
        if logo_x < 0 or logo_x + logo_size > w:
            dx = -dx
            logo_x = max(0, min(w - logo_size, logo_x))
        if logo_y < 0 or logo_y + logo_size > h:
            dy = -dy
            logo_y = max(0, min(h - logo_size, logo_y))

        # Малюємо логотип (кольоровий еліпс з літерою)
        cx, cy = logo_x + logo_size // 2, logo_y + logo_size // 2
        color = (
            int(200 + 55 * math.sin(frame_idx * 0.1)),
            int(100 + 100 * math.cos(frame_idx * 0.07)),
            int(150 + 105 * math.sin(frame_idx * 0.05))
        )
        cv2.ellipse(bg, (cx, cy), (logo_size // 2, logo_size // 2),
                    0, 0, 360, color, -1)
        cv2.ellipse(bg, (cx, cy), (logo_size // 2, logo_size // 2),
                    0, 0, 360, (255, 255, 255), 2)
        cv2.putText(bg, 'L', (cx - 12, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Додаємо шум для реалістичності
        noise = np.random.randint(-15, 15, bg.shape, dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        writer.write(bg)

    writer.release()
    print(f"  ✓ Тестове відео збережено: '{output_path}'")


def process_video_with_yolo(
    model,
    input_video_path: Path,
    output_video_path: Path,
    conf_threshold: float = CONFIDENCE_THRESHOLD
) -> dict:
    """
    Обробляє вхідне відео нейронною мережею YOLOv11:
      1. Зчитує відео покадрово
      2. Для кожного кадру виконує детекцію об'єктів
      3. Малює bounding boxes навколо знайдених об'єктів
      4. Записує відео з рамками у вихідний файл
      5. Визначає часові інтервали появи логотипу в кадрі
      6. Будує графік впевненості по кадрах
      7. Виводить топ-5 кадрів з найвищою впевненістю

    Args:
        model: навчена YOLO-модель
        input_video_path: шлях до вхідного відео
        output_video_path: шлях для збереження відео з детекцією
        conf_threshold: поріг впевненості (0-1)

    Returns:
        Словник з результатами обробки:
          intervals    — часові інтервали (кадр_поч, кадр_кін, t_поч, t_кін)
          frame_confs  — впевненість по кадрах
          fps          — частота кадрів відео
          total_frames — загальна кількість кадрів
    """
    if model is None:
        print("  [ПОМИЛКА] Модель не завантажена. Відео не оброблено.")
        return {}

    if not input_video_path.exists():
        print(f"  [ПОМИЛКА] Відео не знайдено: '{input_video_path}'")
        return {}

    print(f"\n[5/5] Обробка відео нейронною мережею YOLOv11...")
    print(f"  Вхід  : {input_video_path}")
    print(f"  Вихід : {output_video_path}")
    print(f"  Поріг : {conf_threshold}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Відкриваємо вхідне відео ---
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"  [ПОМИЛКА] Не вдається відкрити відео: '{input_video_path}'")
        return {}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Відео: {frame_width}×{frame_height}px, {fps:.1f} fps, {total_frames} кадрів")
    print(f"  Тривалість: {total_frames / fps:.1f} секунд")

    # --- Налаштовуємо запис вихідного відео ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc, fps,
        (frame_width, frame_height)
    )

    # --- Зберігаємо результати обробки кожного кадру ---
    frame_confs   = []     # максимальна впевненість для кожного кадру
    frame_numbers = []     # номери кадрів
    top_frames    = []     # (номер_кадру, впевненість, RGB-зображення)

    frame_idx = 0
    start_time = time.time()

    print(f"  Детекція об'єктів у {total_frames} кадрах...")

    with tqdm(total=total_frames, desc="  Обробка", ncols=80) as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # --- YOLOv11 детекція ---
            results = model.predict(
                source=frame_bgr,
                conf=conf_threshold,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE[0],
                verbose=False
            )

            # Отримуємо максимальну впевненість серед усіх детекцій у кадрі
            max_conf = 0.0
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    confs = r.boxes.conf.cpu().numpy()
                    max_conf = float(max(confs))

            frame_confs.append(max_conf)
            frame_numbers.append(frame_idx)

            # --- Малювання bounding boxes на кадрі ---
            annotated_frame = draw_detections_on_frame(
                frame_bgr.copy(), results, frame_idx, fps
            )

            # --- Збереження топ-кадрів за впевненістю ---
            if max_conf >= conf_threshold:
                top_frames.append((frame_idx, max_conf, frame_bgr.copy()))

            writer.write(annotated_frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    processing_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"\n  ✓ Відео оброблено: {frame_idx} кадрів за {elapsed:.1f}с "
          f"({processing_fps:.1f} fps)")
    print(f"  ✓ Збережено у '{output_video_path}'")

    # --- Визначаємо часові інтервали появи логотипу ---
    intervals = extract_detection_intervals(frame_confs, frame_numbers, fps, conf_threshold)

    # --- Звіт про інтервали ---
    print_detection_intervals(intervals, fps)

    # --- Графіки ---
    plot_confidence_timeline(frame_numbers, frame_confs, fps, conf_threshold)

    # --- Візуалізація топ-5 кадрів ---
    visualize_top_frames(top_frames, n=5)

    return {
        'intervals'    : intervals,
        'frame_confs'  : frame_confs,
        'frame_numbers': frame_numbers,
        'fps'          : fps,
        'total_frames' : frame_idx,
        'processing_fps': processing_fps,
    }


def draw_detections_on_frame(
    frame_bgr: np.ndarray,
    results: list,
    frame_idx: int,
    fps: float
) -> np.ndarray:
    """
    Малює bounding boxes та мітки класів на кадрі відео.

    Дизайн:
      - Товста кольорова рамка навколо об'єкта
      - Підпис із назвою класу та відсотком впевненості
      - Напівпрозорий фон підпису для читабельності
      - Лічильник часу у верхньому правому куті

    Args:
        frame_bgr: кадр у форматі BGR (OpenCV)
        results: результати YOLOv11 detections
        frame_idx: номер поточного кадру
        fps: частота кадрів

    Returns:
        Анотований кадр у форматі BGR
    """
    h, w = frame_bgr.shape[:2]

    # Визначаємо колір рамки та шрифту
    BOX_COLOR    = (0, 230, 0)     # яскраво-зелений (BGR)
    TEXT_COLOR   = (255, 255, 255) # білий
    BG_COLOR     = (0, 150, 0)     # темно-зелений фон підпису
    BOX_THICKNESS = 3
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            boxes = r.boxes
            for i in range(len(boxes)):
                # Координати рамки (xyxy формат — абсолютні пікселі)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf  = float(boxes.conf[i].cpu().numpy())
                cls   = int(boxes.cls[i].cpu().numpy())
                label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

                # Обрізаємо до меж кадру
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(x1 + 1, min(x2, w))
                y2 = max(y1 + 1, min(y2, h))

                # Малюємо основну рамку
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2),
                              BOX_COLOR, BOX_THICKNESS)

                # Малюємо кутові акценти рамки (стильний ефект)
                corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
                for cx, cy, sx, sy in [
                    (x1, y1,  1,  1),
                    (x2, y1, -1,  1),
                    (x1, y2,  1, -1),
                    (x2, y2, -1, -1),
                ]:
                    cv2.line(frame_bgr, (cx, cy),
                             (cx + sx * corner_len, cy), BOX_COLOR, BOX_THICKNESS + 1)
                    cv2.line(frame_bgr, (cx, cy),
                             (cx, cy + sy * corner_len), BOX_COLOR, BOX_THICKNESS + 1)

                # Підпис класу та впевненості
                text   = f"{label} {conf * 100:.1f}%"
                font_scale = max(0.4, min(0.8, (x2 - x1) / 200))
                (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, 1)

                # Фон підпису
                label_y = max(y1 - 5, th + baseline + 5)
                cv2.rectangle(frame_bgr,
                              (x1, label_y - th - baseline - 4),
                              (x1 + tw + 6, label_y + 2),
                              BG_COLOR, -1)
                cv2.putText(frame_bgr, text,
                            (x1 + 3, label_y - baseline),
                            FONT, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)

    # --- Час у верхньому правому куті ---
    timestamp = f"T: {frame_idx / fps:.2f}s  F: {frame_idx}"
    ts_font_scale = 0.5
    (tw, th), _ = cv2.getTextSize(timestamp, FONT, ts_font_scale, 1)
    cv2.rectangle(frame_bgr, (w - tw - 10, 5), (w - 2, th + 10),
                  (20, 20, 20), -1)
    cv2.putText(frame_bgr, timestamp, (w - tw - 7, th + 7),
                FONT, ts_font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    return frame_bgr


def extract_detection_intervals(
    frame_confs: list,
    frame_numbers: list,
    fps: float,
    threshold: float,
    min_gap_frames: int = 10,
    smooth_window: int = 5
) -> list:
    """
    Визначає часові інтервали, протягом яких логотип / об'єкт присутній у кадрі.

    Алгоритм:
      1. Застосовуємо медіанне згладжування для усунення одиничних помилок
      2. Переводимо у бінарну маску: 1 = детектовано, 0 = не детектовано
      3. Заповнюємо короткі прогалини (< min_gap_frames кадрів) між сегментами
      4. Витягуємо інтервали як послідовні блоки одиниць

    Args:
        frame_confs:    список впевненостей для кожного кадру
        frame_numbers:  список номерів кадрів
        fps:            частота кадрів
        threshold:      поріг впевненості для позначення «об'єкт є»
        min_gap_frames: коротші прогалини між сегментами заповнюються
        smooth_window:  розмір вікна медіанного згладжування

    Returns:
        Список кортежів (frame_start, frame_end, time_start_s, time_end_s)
    """
    if not frame_confs:
        return []

    confs_arr = np.array(frame_confs)

    # Медіанне згладжування
    if smooth_window > 1 and len(confs_arr) >= smooth_window:
        confs_smooth = median_filter(confs_arr, size=smooth_window)
    else:
        confs_smooth = confs_arr

    # Бінарна маска
    binary = (confs_smooth >= threshold).astype(int)

    # Заповнення коротких прогалин
    def fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        filled = mask.copy()
        i = 0
        while i < len(filled):
            if filled[i] == 0:
                j = i
                while j < len(filled) and filled[j] == 0:
                    j += 1
                if 0 < i and j < len(filled) and (j - i) <= max_gap:
                    filled[i:j] = 1
                i = j
            else:
                i += 1
        return filled

    binary_filled = fill_gaps(binary, min_gap_frames)

    # Витягуємо інтервали
    intervals = []
    in_seg, seg_start = False, 0

    for idx, val in enumerate(binary_filled):
        fn = frame_numbers[idx] if idx < len(frame_numbers) else idx
        if val and not in_seg:
            in_seg = True
            seg_start = fn
        elif not val and in_seg:
            in_seg = False
            seg_end = frame_numbers[idx - 1] if idx > 0 and (idx - 1) < len(frame_numbers) else fn
            intervals.append((seg_start, seg_end,
                              seg_start / fps, seg_end / fps))

    if in_seg:
        last_fn = frame_numbers[-1] if frame_numbers else 0
        intervals.append((seg_start, last_fn,
                          seg_start / fps, last_fn / fps))

    return intervals


def print_detection_intervals(intervals: list, fps: float) -> None:
    """
    Виводить таблицю часових інтервалів виявлення об'єкту.
    """
    print(f"\n  ┌{'─' * 52}┐")
    print(f"  │  Часові інтервали появи '{DETECTION_TARGET}' у відео{' ' * 4}│")
    print(f"  ├{'─' * 52}┤")

    if not intervals:
        print(f"  │  Об'єкт не виявлено (впевненість < {CONFIDENCE_THRESHOLD}){' ' * 5}│")
    else:
        print(f"  │  № │ Кадри {' ' * 15}│ Час, секунди       │")
        print(f"  ├{'─' * 52}┤")
        total_duration = 0.0
        for i, (f_start, f_end, t_start, t_end) in enumerate(intervals):
            duration = t_end - t_start
            total_duration += duration
            print(f"  │ {i + 1:2d} │ {f_start:5d} – {f_end:5d}  ({f_end - f_start + 1:4d} кадри) "
                  f"│ {t_start:5.2f}s – {t_end:5.2f}s ({duration:.2f}s) │")
        print(f"  ├{'─' * 52}┤")
        print(f"  │  Загальна тривалість появи: {total_duration:.2f}с, "
              f"сегментів: {len(intervals)}{' ' * max(0, 8 - len(str(len(intervals))))}│")

    print(f"  └{'─' * 52}┘")


def plot_confidence_timeline(
    frame_numbers: list,
    frame_confs: list,
    fps: float,
    threshold: float
) -> None:
    """
    Будує графік зміни впевненості моделі впродовж відео.
    Затінює зони, де об'єкт детектовано (вище порогу).
    """
    if not frame_confs:
        return

    print("  Побудова графіку впевненості по кадрах...")
    confs  = np.array(frame_confs)
    frames = np.array(frame_numbers)
    times  = frames / fps  # переводимо кадри в секунди

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Верхній графік: впевненість у часі ---
    ax1.plot(times, confs, color='steelblue', linewidth=0.9, label='Впевненість моделі')
    ax1.axhline(threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'Поріг = {threshold}')
    ax1.fill_between(times, confs, threshold,
                     where=(confs >= threshold),
                     alpha=0.3, color='green',
                     label=f'Об\'єкт виявлено (≥{threshold})')
    ax1.set_ylabel('Впевненість детекції', fontsize=11)
    ax1.set_title(f'YOLOv11 — Детекція «{DETECTION_TARGET}» у відео', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(alpha=0.3)
    ax1.spines[['top', 'right']].set_visible(False)

    # --- Нижній графік: бінарна маска наявності об'єкта ---
    binary = (confs >= threshold).astype(float)
    ax2.fill_between(times, binary, step='mid',
                     alpha=0.7, color='limegreen',
                     label='Детектовано (1=так, 0=ні)')
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Ні', 'Так'])
    ax2.set_xlabel('Час, секунди', fontsize=11)
    ax2.set_ylabel('Наявність', fontsize=11)
    ax2.set_title('Бінарна маска виявлення', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    save_path = BASE_DIR / "video_confidence_timeline.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Графік збережено у '{save_path}'")


def visualize_top_frames(top_frames: list, n: int = 5) -> None:
    """
    Відображає топ-n кадрів з найвищою впевненістю детекції.

    Args:
        top_frames: список кортежів (frame_idx, conf, frame_bgr)
        n: кількість кадрів для відображення
    """
    if not top_frames:
        print("  [ПОПЕРЕДЖЕННЯ] Топ-кадри відсутні (логотип не виявлено)")
        return

    print(f"  Топ-{min(n, len(top_frames))} кадрів з найвищою впевненістю:")

    # Сортуємо за спаданням впевненості
    sorted_frames = sorted(top_frames, key=lambda x: x[1], reverse=True)[:n]

    ncols = min(n, len(sorted_frames))
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    # Директорія для збереження топ-кадрів
    top_dir = BASE_DIR / "top_detections"
    top_dir.mkdir(parents=True, exist_ok=True)

    for i, (f_num, conf, frame_bgr) in enumerate(sorted_frames):
        # Конвертуємо BGR → RGB для matplotlib
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        axes[i].imshow(frame_rgb)
        axes[i].set_title(f"Кадр {f_num}\n{conf * 100:.1f}%", fontsize=10)
        axes[i].axis('off')

        # Зберігаємо кадр на диск
        save_path = top_dir / f"top_{i + 1:02d}_frame{f_num:04d}_conf{conf:.3f}.jpg"
        cv2.imwrite(str(save_path), frame_bgr)
        print(f"    ✓ Кадр {f_num}: {conf * 100:.1f}% → '{save_path}'")

    plt.suptitle(f"Топ-{ncols} кадрів з найвищою впевненістю детекції «{DETECTION_TARGET}»",
                 fontsize=11)
    plt.tight_layout()
    summary_path = BASE_DIR / "top_detections_summary.png"
    plt.savefig(summary_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Зведений рисунок збережено у '{summary_path}'")


# ============================================================
# РОЗДІЛ 7: ПОРІВНЯЛЬНИЙ АНАЛІЗ (ПОФРЕЙМОВА vs БАТЧЕВА ОБРОБКА)
# ============================================================

def benchmark_inference_speed(model, input_video_path: Path) -> dict:
    """
    Порівнює швидкість обробки відео:
      A) Покадрово (одне зображення за виклик predict)
      B) Батчами (BATCH кадрів за виклик predict)

    Також демонструє вплив пост-обробки (медіанне згладжування):
      C) Без пост-обробки
      D) З медіанним фільтром + заповненням прогалин

    Args:
        model: навчена YOLO-модель
        input_video_path: шлях до відеофайлу

    Returns:
        Словник з результатами бенчмарку
    """
    if model is None or not input_video_path.exists():
        return {}

    print("\n=== ДОДАТКОВЕ ЗАВДАННЯ: Порівняльний аналіз ефективності обробки відео ===")

    # --- Завантажуємо всі кадри у пам'ять ---
    print("  Завантаження кадрів у пам'ять...")
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    all_frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        all_frames.append(fr)
    cap.release()
    total = len(all_frames)
    print(f"  Завантажено {total} кадрів")

    BENCH_FRAMES = min(100, total)  # обмежуємо для швидкого бенчмарку
    bench_frames = all_frames[:BENCH_FRAMES]

    # ---- A) ПОКАДРОВА ОБРОБКА ----
    print(f"\n  A) Покадрова обробка ({BENCH_FRAMES} кадрів)...")
    t0 = time.perf_counter()
    preds_single = []
    for fr in bench_frames:
        r = model.predict(source=fr, conf=CONFIDENCE_THRESHOLD,
                          imgsz=IMAGE_SIZE[0], verbose=False)
        conf = 0.0
        if r and len(r) > 0 and r[0].boxes is not None and len(r[0].boxes) > 0:
            conf = float(r[0].boxes.conf.cpu().numpy().max())
        preds_single.append(conf)
    t_single = time.perf_counter() - t0

    # ---- B) БАТЧЕВА ОБРОБКА ----
    BATCH = 8
    print(f"  B) Батчева обробка (batch={BATCH})...")
    t0 = time.perf_counter()
    preds_batch = []
    for i in range(0, BENCH_FRAMES, BATCH):
        chunk = bench_frames[i:i + BATCH]
        r_list = model.predict(source=chunk, conf=CONFIDENCE_THRESHOLD,
                               imgsz=IMAGE_SIZE[0], verbose=False)
        for r in r_list:
            conf = 0.0
            if r.boxes is not None and len(r.boxes) > 0:
                conf = float(r.boxes.conf.cpu().numpy().max())
            preds_batch.append(conf)
    t_batch = time.perf_counter() - t0

    fps_single = BENCH_FRAMES / t_single if t_single > 0 else 0.0
    fps_batch  = BENCH_FRAMES / t_batch  if t_batch  > 0 else 0.0
    speedup    = fps_batch / fps_single  if fps_single > 0 else 0.0

    print(f"\n  ┌{'─' * 50}┐")
    print(f"  │  A) Покадрово : {t_single:.2f}с → {fps_single:.1f} fps{' ' * 15}│")
    print(f"  │  B) Батч={BATCH} : {t_batch:.2f}с → {fps_batch:.1f} fps{' ' * 15}│")
    print(f"  │  Прискорення : {speedup:.2f}× за рахунок батчування{' ' * 5}│")
    print(f"  └{'─' * 50}┘")

    # ---- C/D) Пост-обробка: медіана + заповнення прогалин ----
    preds_arr    = np.array(preds_batch)
    preds_smooth = median_filter(preds_arr, size=9)

    binary_raw    = (preds_arr >= CONFIDENCE_THRESHOLD).astype(int)
    binary_smooth = (preds_smooth >= CONFIDENCE_THRESHOLD).astype(int)

    def fill_gaps(mask, max_gap):
        """Заповнює відрізки нулів довжиною ≤ max_gap одиницями."""
        filled = mask.copy()
        i = 0
        while i < len(filled):
            if filled[i] == 0:
                j = i
                while j < len(filled) and filled[j] == 0:
                    j += 1
                if 0 < i < len(filled) and j < len(filled) and (j - i) <= max_gap:
                    filled[i:j] = 1
                i = j
            else:
                i += 1
        return filled

    binary_filled = fill_gaps(binary_smooth, max_gap=15)

    frames_axis = list(range(len(preds_arr)))
    segs_raw     = _count_segments(binary_raw)
    segs_filled  = _count_segments(binary_filled)

    print(f"\n  C) Без пост-обробки        : {segs_raw} сегментів детекції")
    print(f"  D) Медіана + заповнення    : {segs_filled} сегментів (менше шуму)")

    # ---- Порівняльний графік ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(frames_axis, preds_arr, color='steelblue', linewidth=0.9, label='Без обробки')
    axes[0].axhline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                    label=f'Поріг = {CONFIDENCE_THRESHOLD}')
    axes[0].set_ylabel('Впевненість')
    axes[0].set_title('A) Покадрові передбачення — без пост-обробки')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(frames_axis, preds_smooth, color='darkorange', linewidth=0.9,
                 label='Медіана (вікно=9)')
    axes[1].axhline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('Впевненість')
    axes[1].set_title('B) Після медіанного згладжування')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(frames_axis, binary_filled, step='mid',
                         alpha=0.7, color='green',
                         label=f'Виявлено (медіана + заповнення прогалин)')
    axes[2].set_ylim(-0.1, 1.3)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['Ні', 'Так'])
    axes[2].set_xlabel('Номер кадру')
    axes[2].set_ylabel('Наявність')
    axes[2].set_title('C) Фінальна маска після пост-обробки')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.suptitle(f'Порівняльний аналіз методів обробки відео — «{DETECTION_TARGET}»',
                 fontsize=13)
    plt.tight_layout()
    save_path = BASE_DIR / "video_postprocessing_comparison.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Збережено: '{save_path}'")

    # ---- Зведена таблиця ----
    df = pd.DataFrame({
        'Метод'       : [f'Покадрово', f'Батч={BATCH} (покращений)'],
        'Час (с)'     : [f'{t_single:.2f}', f'{t_batch:.2f}'],
        'FPS'         : [f'{fps_single:.1f}', f'{fps_batch:.1f}'],
        'Прискорення' : ['1.0×', f'{speedup:.2f}×'],
        'Сегментів'   : [segs_raw, segs_filled],
    })
    print("\n  Зведена таблиця порівняння:")
    print(df.to_string(index=False))

    return {
        't_single': t_single, 't_batch': t_batch,
        'fps_single': fps_single, 'fps_batch': fps_batch,
        'speedup': speedup,
        'segs_raw': segs_raw, 'segs_smooth': segs_filled,
    }


def _count_segments(binary_mask: np.ndarray) -> int:
    """Рахує кількість неперервних сегментів одиниць у бінарній масці."""
    count, in_seg = 0, False
    for v in binary_mask:
        if v and not in_seg:
            in_seg = True
            count += 1
        elif not v and in_seg:
            in_seg = False
    return count


# ============================================================
# РОЗДІЛ 8: ПІДСУМКОВИЙ ЗВІТ
# ============================================================

def print_final_report(metrics: dict, video_results: dict) -> None:
    """
    Виводить підсумковий звіт з усіма результатами МКР.
    """
    sep = '=' * 60
    print(f"\n{sep}")
    print(f"  ПІДСУМКОВИЙ ЗВІТ МКР")
    print(f"  Предмет: ПіР ПС з НМ")
    print(f"  Завдання: Детекція «{DETECTION_TARGET}» на відео за YOLOv11")
    print(sep)

    print(f"\n  1. ДАТАСЕТ:")
    print_dataset_statistics(AUGMENTED_DATASET_DIR, "Аугментований датасет")

    print(f"\n  2. АУГМЕНТАЦІЯ (albumentations):")
    print(f"     Методи (7+):")
    aug_methods = [
        "HorizontalFlip", "VerticalFlip", "RandomBrightnessContrast",
        "HueSaturationValue", "GaussNoise", "MotionBlur",
        "ShiftScaleRotate", "CoarseDropout", "CLAHE",
        "RandomGamma", "Perspective"
    ]
    for i, m in enumerate(aug_methods, 1):
        print(f"       {i:2d}. {m}")

    print(f"\n  3. МОДЕЛЬ YOLOv11:")
    print(f"     Версія: {YOLO_MODEL_NAME}")
    print(f"     Епохи : {EPOCHS}")
    print(f"     Батч  : {BATCH_SIZE}")

    if metrics:
        print(f"\n  4. МЕТРИКИ ЯКОСТІ:")
        print(f"     mAP@50    : {metrics.get('mAP50', 0):.4f}")
        print(f"     mAP@50-95 : {metrics.get('mAP50-95', 0):.4f}")
        print(f"     Precision : {metrics.get('precision', 0):.4f}")
        print(f"     Recall    : {metrics.get('recall', 0):.4f}")
        print(f"     F1-Score  : {metrics.get('f1', 0):.4f}")

    if video_results:
        fps          = video_results.get('fps', 25.0)
        total        = video_results.get('total_frames', 0)
        proc_fps     = video_results.get('processing_fps', 0)
        intervals    = video_results.get('intervals', [])
        total_dur    = sum(t[3] - t[2] for t in intervals) if intervals else 0

        print(f"\n  5. ОБРОБКА ВІДЕО:")
        print(f"     Кадрів оброблено   : {total}")
        print(f"     Тривалість відео   : {total / fps:.1f}с")
        print(f"     Швидкість обробки  : {proc_fps:.1f} fps")
        print(f"     Сегментів детекції : {len(intervals)}")
        print(f"     Час присутності    : {total_dur:.1f}с")
        print(f"     Вихідне відео      : {VIDEO_OUTPUT_PATH}")

    print(f"\n  6. ЗБЕРЕЖЕНІ ФАЙЛИ:")
    output_files = [
        (DATA_YAML_PATH,                      "Конфігурація YOLOv11"),
        (VIDEO_OUTPUT_PATH,                   "Відео з детекцією"),
        (BASE_DIR / "dataset_samples_preview.png", "Зразки датасету"),
        (BASE_DIR / "augmentation_comparison.png", "Порівняння аугментацій"),
        (BASE_DIR / "metrics_bar_chart.png",  "Метрики якості"),
        (BASE_DIR / "video_confidence_timeline.png", "Графік впевненості"),
        (BASE_DIR / "top_detections_summary.png", "Топ кадри"),
    ]
    for path, descr in output_files:
        status = "✓" if Path(path).exists() else "·"
        print(f"     {status} {descr:<35} → {path}")

    print(f"\n{sep}\n")


# ============================================================
# ГОЛОВНА ФУНКЦІЯ (MAIN PIPELINE)
# ============================================================

def main():
    """
    Головна функція: повний пайплайн МКР.

    Послідовність виконання:
      1. Створення структури директорій
      2. (Опціонально) Завантаження реальних зображень
      3. Генерація синтетичного датасету у форматі YOLO
      4. Аугментація датасету (albumentations, 7+ методів)
      5. Навчання YOLOv11 на аугментованому датасеті
      6. Оцінка якості моделі на тестовій вибірці
      7. Обробка відео з детекцією та збереженням рамок
      8. Підсумковий звіт

    НАЛАШТУВАННЯ ДЛЯ РЕАЛЬНИХ ДАНИХ:
      - Встановіть LOGO_DIR / PET_DIR на шлях до ваших зображень
      - Встановіть BACKGROUND_DIR на шлях до фонових зображень
      - Замініть VIDEO_INPUT_PATH на шлях до реального відео
      - Для розмітки: використайте labelImg або Roboflow, тоді замість
        generate_dataset() завантажте готовий датасет з Roboflow
    """
    print("=" * 60)
    print("  МКР: Детекція об'єкту на відео (YOLOv11)")
    print(f"  Об'єкт: «{DETECTION_TARGET}»")
    print(f"  Час запуску: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # -----------------------------------------------------------
    # КРОК 0: Перевірка залежностей
    # -----------------------------------------------------------
    print("\n[0/6] Перевірка залежностей...")
    missing = []
    for lib in ['cv2', 'albumentations', 'PIL', 'tqdm', 'yaml']:
        try:
            __import__(lib)
            print(f"  ✓ {lib}")
        except ImportError:
            print(f"  ✗ {lib} — НЕ ВСТАНОВЛЕНО")
            missing.append(lib)
    if ULTRALYTICS_AVAILABLE:
        print(f"  ✓ ultralytics (YOLOv11)")
    else:
        print(f"  ✗ ultralytics — НЕ ВСТАНОВЛЕНО → pip install ultralytics")
        missing.append('ultralytics')

    if missing:
        print(f"\n  [ПОМИЛКА] Встановіть відсутні бібліотеки:")
        for lib in missing:
            print(f"    pip install {lib}")
        print("  Після встановлення запустіть скрипт знову.")
        sys.exit(1)

    # -----------------------------------------------------------
    # КРОК 1: Створення директорій
    # -----------------------------------------------------------
    create_directory_structure()

    # -----------------------------------------------------------
    # КРОК 2: Завантаження реальних або використання синтетичних зображень
    # -----------------------------------------------------------
    print("\n[2/6] Підготовка вихідних зображень...")

    # === НАЛАШТУЙТЕ ЦІ ШЛЯХИ ДЛЯ РЕАЛЬНИХ ДАНИХ ===
    # Для логотипу (лаб 6): папка з PNG/JPEG зображеннями логотипу
    # Приклад: "C:/MyProject/toyota_logos"
    LOGO_DIR_REAL = RAW_DATASET_DIR  
    # Для улюбленця (лаб 5): папка з PNG/JPEG зображеннями тварини
    # Приклад: "C:/MyProject/golden_retriever"
    # Залишаємо None, оскільки ми працюємо з автомобільною тематикою
    PET_DIR_REAL = None
    # Фонові зображення (без об'єкту)
    BACKGROUND_DIR_REAL = None  # Приклад: "C:/MyProject/backgrounds"

    # Вибираємо папку відповідно до типу об'єкту
    obj_dir = LOGO_DIR_REAL if DETECTION_TARGET == 'logo' else PET_DIR_REAL
    logo_files, background_files = load_real_images(obj_dir, BACKGROUND_DIR_REAL)

    if not logo_files:
        print("  ℹ Використовується синтетична генерація (реальні зображення не вказані)")
        print("    Щоб використати реальні зображення, задайте LOGO_DIR_REAL / PET_DIR_REAL")

    # -----------------------------------------------------------
    # КРОК 3: Генерація датасету у форматі YOLO
    # -----------------------------------------------------------
    generate_dataset(
        logo_files=logo_files,
        background_files=background_files
    )

    # Перегляд зразків датасету
    visualize_dataset_samples(n=6)
    print_dataset_statistics(RAW_DATASET_DIR, "Оригінальний датасет (сирий)")

    # -----------------------------------------------------------
    # КРОК 4: Аугментація (albumentations, 7+ методів)
    # -----------------------------------------------------------
    augment_dataset(
        source_dir=RAW_DATASET_DIR,
        target_dir=AUGMENTED_DATASET_DIR
    )
    visualize_augmentation_comparison(n_pairs=4)
    print_dataset_statistics(AUGMENTED_DATASET_DIR, "Аугментований датасет")

    # Створюємо data.yaml для аугментованого датасету
    data_yaml = create_data_yaml(AUGMENTED_DATASET_DIR)

    # -----------------------------------------------------------
    # КРОК 5: Навчання YOLOv11
    # -----------------------------------------------------------
    model = None
    metrics = {}

    # Перевіряємо, чи вже навчена модель існує
    existing_models = list(RUNS_DIR.rglob("best.pt"))
    if existing_models:
        print(f"\n[4/5] Знайдено навчену модель. Завантаження...")
        model = load_best_model()
    else:
        # Запуск навчання
        model, train_results = train_yolov11(data_yaml)
        if model is None:
            print("\n  [УВАГА] Навчання не виконано. Спроба завантажити базову модель...")
            if ULTRALYTICS_AVAILABLE:
                model = YOLO(YOLO_MODEL_NAME)  # нена навчена базова модель

    # Оцінка якості моделі
    if model is not None:
        metrics = evaluate_model(model, data_yaml)

    # -----------------------------------------------------------
    # КРОК 6: Підготовка відео та обробка
    # -----------------------------------------------------------
    # Перевіряємо наявність реального відео; якщо немає — генеруємо тестове
    if not VIDEO_INPUT_PATH.exists():
        print(f"\n  Реальне відео не знайдено: '{VIDEO_INPUT_PATH}'")
        print("  Генерація тестового синтетичного відео...")
        create_test_video(VIDEO_INPUT_PATH, num_frames=300)
    else:
        print(f"\n  Використовується реальне відео: '{VIDEO_INPUT_PATH}'")

    # Обробка відео
    video_results = {}
    if model is not None:
        video_results = process_video_with_yolo(
            model=model,
            input_video_path=VIDEO_INPUT_PATH,
            output_video_path=VIDEO_OUTPUT_PATH,
            conf_threshold=CONFIDENCE_THRESHOLD
        )

        # Додаткове завдання: бенчмарк швидкості обробки
        bench_results = benchmark_inference_speed(model, VIDEO_INPUT_PATH)

    # -----------------------------------------------------------
    # КРОК 7: Підсумковий звіт
    # -----------------------------------------------------------
    print_final_report(metrics, video_results)


# ============================================================
# ТОЧКА ВХОДУ
# ============================================================

if __name__ == "__main__":
    # Встановлюємо seed для відтворюваності результатів
    random.seed(42)
    np.random.seed(42)

    main()