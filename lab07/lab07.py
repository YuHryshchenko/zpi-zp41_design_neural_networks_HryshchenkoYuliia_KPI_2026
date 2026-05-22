# ============================================================
# ЛАБОРАТОРНА РОБОТА №7
# Реалізація та дослідження рекурентної нейронної мережі LSTM
# для аналізу настроїв тексту (Yelp Review Polarity Dataset)
# ============================================================
# Датасет: https://www.kaggle.com/datasets/irustandi/yelp-review-polarity
# ============================================================

import os
import re
import spacy
import nltk
import glob
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from nltk.corpus import stopwords
 
# ============================================================
# КРОК 0. Встановлення залежностей
# ============================================================
# !pip install swifter kaggle spacy tokenizers nltk tensorflow pandas scikit-learn seaborn matplotlib --quiet
# !python -m spacy download en_core_web_sm --quiet

# --- 0.1 Визначення та налаштування середовища ---
CURRENT_LAB = "lab07"

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

# ============================================================
# КРОК 1. Завантаження датасету з Kaggle
# ============================================================

# --- Альтернативний варіант (Google Colab + Google Drive): ---
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir -p ~/.kaggle
# !cp /content/drive/MyDrive/kaggle/kaggle.json ~/.kaggle/kaggle.json
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download irustandi/yelp-review-polarity -p ./
# !unzip -o yelp-review-polarity.zip -d ./yelp_dataset/
# -------------------------------------------------------------

# Пряме завантаження у поточну директорію через Kaggle API
dataset_path = BASE_DIR + './yelp_dataset/yelp_review_polarity_csv'
 
# Kaggle Notebooks mount the dataset automatically under /kaggle/input/
KAGGLE_INPUT_PATH = '/kaggle/input/datasets/irustandi/yelp-review-polarity/yelp_review_polarity_csv'

MODEL_DIR = os.path.join(BASE_DIR, 'model')

def dataset_is_ready(path):
    """Returns True when both train.csv and test.csv exist at the given path."""
    return (
        os.path.exists(os.path.join(path, 'train.csv')) and
        os.path.exists(os.path.join(path, 'test.csv'))
    )
 
if is_kaggle() and dataset_is_ready(KAGGLE_INPUT_PATH):
    # On Kaggle the dataset is already mounted — no download needed
    dataset_path = KAGGLE_INPUT_PATH
    print(f"Kaggle: датасет знайдено у {dataset_path}. Завантаження не потрібне.")
elif dataset_is_ready(dataset_path):
    print(f"Датасет вже присутній у {dataset_path}. Завантаження пропущено.")
else:
    try:
        import kaggle
        os.makedirs(dataset_path, exist_ok=True)
        print("Завантаження датасету з Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'irustandi/yelp-review-polarity',
            path=dataset_path,
            unzip=True
        )
        print("Завантаження та розпакування завершено.")
    except Exception as e:
        print(f"Помилка під час роботи з Kaggle API: {e}")
        print("Переконайтеся, що файл kaggle.json знаходиться у директорії ~/.kaggle/")

# ============================================================
# КРОК 2. Імпорти та завантаження даних
# ============================================================

nltk.download('stopwords', quiet=True)

# ------------------------------------------------------------------
# Завантажуємо дані.
# Датасет irustandi/yelp-review-polarity містить CSV-файли
# (train.csv / test.csv), де перша колонка — оцінка (1 або 2),
# друга — текст відгуку. Також підтримується JSON-формат.
# ------------------------------------------------------------------

csv_train_path = os.path.join(dataset_path, 'train.csv')

print("Loading Yelp review data...")
df_raw = pd.read_csv(csv_train_path, header=None, names=['stars', 'text'], nrows=50001)
df = df_raw.copy()
print(f"Successfully loaded {len(df)} entries.")

# ============================================================
# КРОК 3. Попередня обробка текстів
# ============================================================

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Очищення тексту відгуку:
      1. переведення в нижній регістр
      2. видалення символів, що не є словами
      3. стиснення пробілів
      4. видалення стоп-слів
    """
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

print("Cleaning review content...")
df['clean_text'] = df['text'].apply(preprocess_text)
print("Text preprocessing complete.")

# Мітки: у CSV-варіанті stars = 1 або 2
# CSV: 1 → 0 (negative), 2 → 1 (positive)
if df['stars'].max() <= 2:
    df['label'] = df['stars'].apply(lambda x: 1 if x == 2 else 0)
else:
    df['label'] = df['stars'].apply(lambda x: 1 if x > 3 else 0)

print(f"Label distribution:\n{df['label'].value_counts()}")

# ============================================================
# КРОК 4. Розділення на тренувальні та тестові дані
# ============================================================

print("Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'],
    test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============================================================
# КРОК 5. Токенізація та padding (Keras Tokenizer)
# ============================================================

print("Converting text to sequences...")

VOCAB_SIZE      = 10000
SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(train_sequences, maxlen=SEQUENCE_LENGTH, padding='post')
X_test_pad  = pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH, padding='post')

y_train = np.array(y_train)
y_test  = np.array(y_test)

print("Tokenization and padding completed.")
print(f"  Vocabulary size : {min(VOCAB_SIZE, len(tokenizer.word_index))}")
print(f"  X_train_pad shape: {X_train_pad.shape}")
print(f"  X_test_pad shape : {X_test_pad.shape}")

# ============================================================
# КРОК 6. Побудова архітектури LSTM
# ============================================================
 
os.makedirs(MODEL_DIR, exist_ok=True)
 
def find_latest_model(directory):
    """Returns the path of the most recently saved .keras model, or None."""
    files = sorted(glob.glob(os.path.join(directory, 'lstm_*.keras')))
    return files[-1] if files else None
 
saved_model_path = find_latest_model(MODEL_DIR)
 
if saved_model_path:
    print(f"Знайдено збережену модель: {saved_model_path}")
    print("Завантажуємо модель, навчання пропущено.")
    model = tf.keras.models.load_model(saved_model_path)
    model.summary()
    history = None   # history недоступна при завантаженні
else:
    print("Збереженої моделі не знайдено. Починаємо побудову та навчання...")
 
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=SEQUENCE_LENGTH),
        LSTM(128, return_sequences=True),   # перший LSTM — повертає послідовність
        Dropout(0.5),
        LSTM(64),                           # другий LSTM — повертає лише останній стан
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1,  activation='sigmoid')     # бінарна класифікація
    ])
 
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
 
    model.summary()
 
    # ============================================================
    # КРОК 7. Навчання моделі
    # ============================================================
 
    print("Training the model...")
 
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
 
    history = model.fit(
        X_train_pad, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test_pad, y_test),
        callbacks=[early_stop]
    )
 
    # Зберігаємо модель одразу після навчання
    timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(MODEL_DIR, f'lstm_{timestamp}.keras')
    model.save(model_save_path)
    print(f"Модель збережено: {model_save_path}")

# ============================================================
# КРОК 8. Оцінка моделі на тестових даних
# ============================================================

print("Evaluating on test data...")
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Set Accuracy: {accuracy:.4f}")

# ============================================================
# КРОК 9. Графіки точності та втрат
# ============================================================

if history is not None:
    plt.figure(figsize=(12, 5))
 
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'],     label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
 
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'],     label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
 
    plt.tight_layout()
    plt.savefig('lstm_training_history.png')
    plt.show()
else:
    print("Графіки навчання недоступні (модель завантажена з файлу).")

# ============================================================
# КРОК 10. Матриця помилок + accuracy, precision, recall, F-Score
# ============================================================

y_pred_probs = model.predict(X_test_pad).flatten()
y_pred       = (y_pred_probs > 0.5).astype(int)

# --- Матриця помилок ---
cm = confusion_matrix(y_test, y_pred)
class_labels = ['Negative (0)', 'Positive (1)']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.xlabel('Передбачений клас')
plt.ylabel('Справжній клас')
plt.title('Матриця помилок (Confusion Matrix)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# --- Числові метрики ---
acc_val  = accuracy_score(y_test, y_pred)
prec_val = precision_score(y_test, y_pred, zero_division=0)
rec_val  = recall_score(y_test, y_pred, zero_division=0)
f1_val   = f1_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy  : {acc_val:.4f}")
print(f"Precision : {prec_val:.4f}")
print(f"Recall    : {rec_val:.4f}")
print(f"F-Score   : {f1_val:.4f}")
print("\nДетальний звіт:")
print(classification_report(y_test, y_pred,
                             target_names=class_labels,
                             zero_division=0))

# ============================================================
# КРОК 11. Функція аналізу настроїв + тестування
# ============================================================

def analyze_sentiment(input_text: str) -> str:
    """
    Визначає настрій тексту за допомогою навченої LSTM-моделі.
    Параметри:
        input_text — довільний рядок тексту (відгук).
    Повертає:
        'Positive' або 'Negative' з оцінкою впевненості.
    """
    cleaned = preprocess_text(input_text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=SEQUENCE_LENGTH, padding='post')
    prob    = float(model.predict(padded, verbose=0)[0][0])
    label   = "Positive" if prob > 0.5 else "Negative"
    print(f"  Текст     : {input_text[:80]}...")
    print(f"  Настрій   : {label}  (впевненість: {prob:.4f})")
    return label

print("\nSample prediction tests:")

# --- Тексти поза датасетом ---
sample_1 = "Absolutely loved the atmosphere and the food was fantastic!"
print(f"\nReview: {sample_1}")
print(f"Sentiment: {analyze_sentiment(sample_1)}")

sample_2 = "I had to wait forever and the service was terrible."
print(f"\nReview: {sample_2}")
print(f"Sentiment: {analyze_sentiment(sample_2)}")

sample_3 = "The coffee was okay, nothing special but not bad either."
print(f"\nReview: {sample_3}")
print(f"Sentiment: {analyze_sentiment(sample_3)}")

sample_4 = "Best restaurant I have ever visited! Will definitely come back!"
print(f"\nReview: {sample_4}")
print(f"Sentiment: {analyze_sentiment(sample_4)}")

# --- Тексти З датасету (для перевірки узгодженості) ---
print("\n--- Тести на прикладах з датасету ---")
sample_indices = [0, 1, 5, 10, 20]
for idx in sample_indices:
    raw_text   = df['text'].iloc[idx]
    true_label = df['label'].iloc[idx]
    predicted  = analyze_sentiment(raw_text)
    true_str   = 'Positive' if true_label == 1 else 'Negative'
    match      = '✓' if predicted == true_str else '✗'
    print(f"  [{match}] Очікувано: {true_str}, Отримано: {predicted}")

# ============================================================
# ДОДАТКОВЕ ЗАВДАННЯ (+1 бал)
# Порівняння різних методів токенізації
# ============================================================
# Методи:
#   1. Keras Tokenizer           (вже використовувався вище)
#   2. NLTK word_tokenize
#   3. SpaCy tokenizer
#   4. BPE через HuggingFace tokenizers (WordPiece / byte-level BPE)
# ============================================================

# ! pip install spacy --quiet
# ! python -m spacy download en_core_web_sm --quiet
# ! pip install tokenizers --quiet

def train_and_evaluate(X_tr_pad, X_te_pad, y_tr, y_te,
                       vocab_size, seq_len, method_name,
                       epochs=5, batch_size=64):
    """Будує, навчає та оцінює LSTM для конкретного методу токенізації."""
    print(f"\n{'='*55}")
    print(f"  Метод токенізації: {method_name}")
    print(f"{'='*55}")

    m = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=seq_len),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dense(1,  activation='sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    cb = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=0)
    hist = m.fit(X_tr_pad, y_tr, epochs=epochs, batch_size=batch_size,
                 validation_data=(X_te_pad, y_te), callbacks=[cb], verbose=0)

    y_p = (m.predict(X_te_pad, verbose=0).flatten() > 0.5).astype(int)

    acc  = accuracy_score(y_te, y_p)
    prec = precision_score(y_te, y_p, zero_division=0)
    rec  = recall_score(y_te, y_p, zero_division=0)
    f1   = f1_score(y_te, y_p, zero_division=0)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return {'method': method_name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'history': hist}

results_list = []

# ─────────────────────────────────────────────
# 1. Keras Tokenizer (baseline)
# ─────────────────────────────────────────────
res_keras = {
    'method'   : 'Keras Tokenizer',
    'accuracy' : acc_val,
    'precision': prec_val,
    'recall'   : rec_val,
    'f1'       : f1_val,
}
results_list.append(res_keras)
print(f"\n[1] Keras Tokenizer → Accuracy: {acc_val:.4f}")

# ─────────────────────────────────────────────
# 2. NLTK word_tokenize
# ─────────────────────────────────────────────
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize

print("\n[2] Токенізація за допомогою NLTK word_tokenize...")

def nltk_tokenize(texts, vocab_size=10000, seq_len=100):
    tokenized = [word_tokenize(t) for t in texts]
    from collections import Counter
    counter = Counter(w for toks in tokenized for w in toks)
    vocab   = {w: i + 2 for i, (w, _) in enumerate(counter.most_common(vocab_size - 2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    seqs = [[vocab.get(w, 1) for w in toks] for toks in tokenized]
    return pad_sequences(seqs, maxlen=seq_len, padding='post'), vocab

X_all_text = pd.concat([X_train, X_test], ignore_index=True)
all_pad_nltk, vocab_nltk = nltk_tokenize(list(X_train) + list(X_test), VOCAB_SIZE, SEQUENCE_LENGTH)
split = len(X_train)
X_tr_nltk, X_te_nltk = all_pad_nltk[:split], all_pad_nltk[split:]

res_nltk = train_and_evaluate(
    X_tr_nltk, X_te_nltk, y_train, y_test,
    VOCAB_SIZE, SEQUENCE_LENGTH, 'NLTK word_tokenize'
)
results_list.append(res_nltk)

# ─────────────────────────────────────────────
# 3. SpaCy tokenizer
# ─────────────────────────────────────────────
print("\n[3] Токенізація за допомогою SpaCy...")

nlp_spacy = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

def spacy_tokenize_texts(texts, vocab_size=10000, seq_len=100):
    from collections import Counter
    tokenized = [[tok.text.lower() for tok in nlp_spacy(str(t))] for t in texts]
    counter   = Counter(w for toks in tokenized for w in toks)
    vocab_sp  = {w: i + 2 for i, (w, _) in enumerate(counter.most_common(vocab_size - 2))}
    vocab_sp['<PAD>'] = 0
    vocab_sp['<UNK>'] = 1
    seqs = [[vocab_sp.get(w, 1) for w in toks] for toks in tokenized]
    return pad_sequences(seqs, maxlen=seq_len, padding='post'), vocab_sp

all_texts_combined = list(X_train) + list(X_test)
all_pad_spacy, vocab_spacy = spacy_tokenize_texts(all_texts_combined, VOCAB_SIZE, SEQUENCE_LENGTH)
X_tr_spacy = all_pad_spacy[:split]
X_te_spacy = all_pad_spacy[split:]

res_spacy = train_and_evaluate(
    X_tr_spacy, X_te_spacy, y_train, y_test,
    VOCAB_SIZE, SEQUENCE_LENGTH, 'SpaCy tokenizer'
)
results_list.append(res_spacy)

# ─────────────────────────────────────────────
# 4. BPE (Byte Pair Encoding) — HuggingFace tokenizers
# ─────────────────────────────────────────────
print("\n[4] BPE токенізація (HuggingFace tokenizers)...")

# Зберігаємо тренувальні тексти у локальний тимчасовий файл
train_texts_file = './train_texts.txt'
with open(train_texts_file, 'w', encoding='utf-8') as f:
    for t in X_train:
        f.write(str(t) + '\n')

BPE_VOCAB   = 10000
bpe_tok     = HFTokenizer(BPE(unk_token='[UNK]'))
bpe_tok.pre_tokenizer = Whitespace()
trainer     = BpeTrainer(vocab_size=BPE_VOCAB, special_tokens=['[PAD]', '[UNK]'])
bpe_tok.train([train_texts_file], trainer)

def bpe_encode(texts, tokenizer_hf, seq_len=100):
    seqs = [tokenizer_hf.encode(str(t)).ids for t in texts]
    return pad_sequences(seqs, maxlen=seq_len, padding='post', truncating='post')

X_tr_bpe = bpe_encode(X_train, bpe_tok, SEQUENCE_LENGTH)
X_te_bpe = bpe_encode(X_test,  bpe_tok, SEQUENCE_LENGTH)

res_bpe = train_and_evaluate(
    X_tr_bpe, X_te_bpe, y_train, y_test,
    BPE_VOCAB, SEQUENCE_LENGTH, 'BPE (HuggingFace)'
)
results_list.append(res_bpe)

# Видаляємо тимчасовий файл
if os.path.exists(train_texts_file):
    os.remove(train_texts_file)

# ─────────────────────────────────────────────
# Зведена таблиця порівняння
# ─────────────────────────────────────────────
df_results = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'history'}
    for r in results_list
])

print("\n" + "=" * 60)
print("  ПОРІВНЯННЯ МЕТОДІВ ТОКЕНІЗАЦІЇ")
print("=" * 60)
print(df_results.to_string(index=False))

# Стовпчаста діаграма порівняння метрик
metrics   = ['accuracy', 'precision', 'recall', 'f1']
x_labels  = df_results['method'].tolist()
x_pos     = np.arange(len(x_labels))
bar_width  = 0.18

fig, ax = plt.subplots(figsize=(13, 6))
for i, metric in enumerate(metrics):
    vals = df_results[metric].tolist()
    ax.bar(x_pos + i * bar_width, vals, bar_width, label=metric.capitalize())

ax.set_xticks(x_pos + bar_width * (len(metrics) - 1) / 2)
ax.set_xticklabels(x_labels, rotation=10, ha='right')
ax.set_ylim(0.0, 1.05)
ax.set_ylabel('Значення метрики')
ax.set_title('Порівняння методів токенізації (LSTM, Yelp)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('tokenization_comparison.png')
plt.show()

# ─────────────────────────────────────────────
# Графіки навчання для кожного методу (крім baseline)
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(len(results_list) - 1, 2, figsize=(14, 4 * (len(results_list) - 1)))

for row_idx, res in enumerate(results_list[1:]):   # пропускаємо Keras baseline
    h = res['history'].history
    ax_acc, ax_loss = axes[row_idx]

    ax_acc.plot(h['accuracy'],     label='Train')
    ax_acc.plot(h['val_accuracy'], label='Val')
    ax_acc.set_title(f"{res['method']} — Accuracy")
    ax_acc.set_xlabel('Epoch')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(h['loss'],     label='Train')
    ax_loss.plot(h['val_loss'], label='Val')
    ax_loss.set_title(f"{res['method']} — Loss")
    ax_loss.set_xlabel('Epoch')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

plt.suptitle('Криві навчання для різних методів токенізації', fontsize=13)
plt.tight_layout()
plt.savefig('tokenization_training_curves.png')
plt.show()
