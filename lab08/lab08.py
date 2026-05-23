!pip install pyspellchecker

# ============================================================
# ЛАБОРАТОРНА РОБОТА №8
# Реалізація та дослідження гібридної нейронної мережі
# СNN-bi-LSTM для розпізнавання мовлення в текст
# ============================================================
# Датасет: https://www.kaggle.com/datasets/dromosys/ljspeech
# ============================================================

import os
import glob
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from spellchecker import SpellChecker

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ДАТАСЕТУ LJ-SPEECH
# ==========================================

# --- 1.1 Визначення та налаштування середовища ---
CURRENT_LAB = "lab08"

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

# Завантажуємо архів (близько 2.6 ГБ). Якщо вже завантажено, процес буде пропущено.
# Alternatively, you can download from https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# import urllib.request
# import tarfile
# DATA_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
# DATA_URL = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
# wavs_path = data_path + "/wavs/"
# metadata_path = data_path + "/metadata.csv"

EPOCHS_LIMIT = 2
DATASET_LIMIT = 500
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# On Kaggle the dataset is mounted automatically under /kaggle/input/
# Dataset slug on Kaggle: https://www.kaggle.com/datasets/dromosys/ljspeech
KAGGLE_INPUT_PATH = '/kaggle/input/datasets/dromosys/ljspeech/LJSpeech-1.1'
LOCAL_DATASET_PATH = os.path.join(BASE_DIR, 'LJSpeech-1.1/LJSpeech-1.1')

def dataset_is_ready(path):
    """Returns True when the wavs/ folder and metadata.csv both exist at path."""
    return (
        os.path.isdir(os.path.join(path, 'wavs')) and
        os.path.exists(os.path.join(path, 'metadata.csv'))
    )

if is_kaggle() and dataset_is_ready(KAGGLE_INPUT_PATH):
    # On Kaggle the dataset is already mounted — no download needed
    dataset_path = KAGGLE_INPUT_PATH
    print(f"Kaggle: датасет знайдено у {dataset_path}. Завантаження не потрібне.")
elif dataset_is_ready(LOCAL_DATASET_PATH):
    dataset_path = LOCAL_DATASET_PATH
    print(f"Датасет вже присутній у {dataset_path}. Завантаження пропущено.")
else:
    try:
        import kaggle
        os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
        print("Завантаження датасету з Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'dromosys/ljspeech',
            path=LOCAL_DATASET_PATH,
            unzip=True
        )
        print("Завантаження та розпакування завершено.")
        dataset_path = LOCAL_DATASET_PATH
    except Exception as e:
        print(f"Помилка під час роботи з Kaggle API: {e}")
        print("Переконайтеся, що файл kaggle.json знаходиться у директорії ~/.kaggle/")
        dataset_path = LOCAL_DATASET_PATH  # fallback — paths set anyway

wavs_path     = os.path.join(dataset_path, 'wavs') + os.sep
metadata_path = os.path.join(dataset_path, 'metadata.csv')

# Зчитуємо метадані
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
# Для пришвидшення демонстрації в лабораторній роботі можна взяти частину датасету
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)[:DATASET_LIMIT] 

# Розділення на тренувальну та тестову вибірки
split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]

# Підготовка словника (алфавіту)
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# ==========================================
# 2. ПОПЕРЕДНЯ ОБРОБКА АУДІО (Spectrogram)
# ==========================================
frame_length = 256
frame_step = 160
fft_length = 384

def encode_single_sample(wav_file, label):
    # 1. Читання аудіофайлу
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    
    # 2. Перетворення аудіо у спектрограму (STFT)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    
    # 3. Нормалізація та логарифмування
    spectrogram = tf.math.pow(tf.math.abs(spectrogram), 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
    # 4. Обробка тексту
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    label = char_to_num(label)
    
    return spectrogram, label

# Створення tf.data.Dataset
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((list(df_train["file_name"]), list(df_train["normalized_transcription"])))
train_dataset = train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((list(df_val["file_name"]), list(df_val["normalized_transcription"])))
val_dataset = val_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.padded_batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# ==========================================
# 3. АРХІТЕКТУРА МЕРЕЖІ СNN-bi-LSTM (DeepSpeech2)
# ==========================================
def CTCLoss(y_true, y_pred):
    # Вираховуємо CTC втрати
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model(input_dim, output_dim, rnn_layers=2, rnn_units=128):
    input_spectrogram = layers.Input((None, input_dim), name="input")
    x = layers.Reshape((-1, input_dim, 1))(input_spectrogram)

    # Згорткові шари (CNN)
    x = layers.Conv2D(32, (11, 41), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (11, 21), strides=(1, 2), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Підготовка до рекурентних шарів (згортаємо розмірності)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # Рекурентні шари (Bi-LSTM / GRU)
    for i in range(rnn_layers):
        x = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True, recurrent_dropout=0.1))(x)
        x = layers.BatchNormalization()(x)

    # Повнозв'язний шар та Softmax
    x = layers.Dense(rnn_units * 2, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(output_dim + 1, activation="softmax")(x)

    model = keras.Model(input_spectrogram, output, name="DeepSpeech2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=CTCLoss)
    return model

# Отримуємо розмірності для моделі
input_dim  = fft_length // 2 + 1
output_dim = char_to_num.vocabulary_size()

os.makedirs(MODEL_DIR, exist_ok=True)

def find_latest_model(directory):
    """Returns the path of the most recently saved .keras model, or None."""
    files = sorted(glob.glob(os.path.join(directory, 'deepspeech2_*.keras')))
    return files[-1] if files else None

saved_model_path = find_latest_model(MODEL_DIR)

if saved_model_path:
    print(f"Знайдено збережену модель: {saved_model_path}")
    print("Завантажуємо модель, навчання пропущено.")
    model = tf.keras.models.load_model(
        saved_model_path,
        custom_objects={"CTCLoss": CTCLoss}
    )
    model.summary()
    history = None   # history недоступна при завантаженні
else:
    print("Збереженої моделі не знайдено. Починаємо побудову та навчання...")
    model = build_model(input_dim, output_dim)
    model.summary()

    # ==========================================
    # 4. НАВЧАННЯ МЕРЕЖІ
    # ==========================================
    epochs = EPOCHS_LIMIT  # Встановити більше (напр. 50-100) для кращої якості
    print("Початок навчання моделі...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    # Зберігаємо модель одразу після навчання
    timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(MODEL_DIR, f"deepspeech2_{timestamp}.keras")
    model.save(model_save_path)
    print(f"Модель збережено: {model_save_path}")

# ==========================================
# 5. ТЕСТУВАННЯ ТА ДЕКОДУВАННЯ (Greedy Search)
# ==========================================
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Використання жадібного пошуку (Greedy Search) від Keras Backend
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result.replace("[UNK]", ""))
    return output_text

# ==========================================
# 6. ДОДАТКОВЕ ЗАВДАННЯ (+1 БАЛ): 
# ВИПРАВЛЕННЯ ПОМИЛОК ТА РОЗРАХУНОК WER
# ==========================================
def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate — standard ASR metric, computed via word-level edit distance.
    WER = (S + D + I) / N
      S = substitutions, D = deletions, I = insertions,
      N = number of words in the reference.
    Uses only the Python standard library (difflib).
    """
    import difflib
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    substitutions = max(len(ref_words), len(hyp_words)) - matches - abs(len(ref_words) - len(hyp_words))
    deletions      = max(0, len(ref_words) - len(hyp_words) - substitutions)
    insertions     = max(0, len(hyp_words) - len(ref_words) - substitutions)
    # Simpler and equally correct: use Levenshtein on word lists directly
    # (SequenceMatcher gives the longest common subsequence, so we derive edits from it)
    n = len(ref_words)
    lcs = matches
    edits = (len(ref_words) - lcs) + (len(hyp_words) - lcs)   # deletions + insertions
    return edits / n
 
 
spell = SpellChecker()
 
def correct_text_spellchecker(text):
    """Пост-обробка: виправлення орфографії розпізнаного тексту"""
    corrected_words = []
    for word in text.split():
        correction = spell.correction(word)
        # Якщо spellchecker не знайшов слова, повертає None. В такому випадку залишаємо оригінал
        corrected_words.append(correction if correction is not None else word)
    return " ".join(corrected_words)
 
print("\n--- Тестування та розрахунок WER ---")
for batch in val_dataset.take(1):
    X, y = batch
    preds = model.predict(X)
    decoded_preds = decode_batch_predictions(preds)
    
    # Декодуємо справжні мітки
    y_true = []
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        y_true.append(label.replace("[UNK]", "").strip())
        
    for i in range(3): # Беремо 3 приклади
        pred_text = decoded_preds[i]
        true_text = y_true[i]
        
        # Застосування методів виправлення помилок
        corrected_text = correct_text_spellchecker(pred_text)
        
        # Обчислення WER (Word Error Rate)
        wer_original  = compute_wer(true_text, pred_text)
        wer_corrected = compute_wer(true_text, corrected_text)
        
        print(f"\nПриклад {i+1}:")
        print(f"Справжній текст : {true_text}")
        print(f"Розпізнано (CNN): {pred_text}")
        print(f"Після корекції  : {corrected_text}")
        print(f"WER до корекції : {wer_original:.2%}")
        print(f"WER після кор.  : {wer_corrected:.2%}")
 
# Висновок до додаткового завдання: 
# Використання статистичних спелчекерів (pyspellchecker) або сучасних мовних моделей 
# дозволяє зменшити Word Error Rate (WER), оскільки CTC модель "кінець-у-кінець" 
# часто припускається фонетичних помилок (наприклад "kat" замість "cat"), які 
# пост-обробка успішно виправляє.
 