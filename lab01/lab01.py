# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ЛАБОРАТОРНА РОБОТА №1                                                       ║
# ║  Реалізація та дослідження нейронної мережі Перцептрон для логічних функцій  ║
# ║  Предмет: Проектування та реалізація програмних систем з нейронними мережами ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Завдання:
#   Написати програму використовуючи бібліотеку TensorFlow, що реалізує
#   Перцептрон для виконання логічної функції XOR для 4-х змінних.
#
# Залежності:
#   pip install tensorflow numpy matplotlib

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 1 – Вступний приклад
# Демонструє базову роботу з Python / matplotlib у Kaggle Notebook
# ══════════════════════════════════════════════════════════════════════════════

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# matplotlib.use("Agg")           # для запуску без дисплея (сервер/CI)

# --- 1.1 Визначення та налаштування середовища ---
CURRENT_LAB = "lab01"

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

# Привітання у Kaggle
print("Привіт, Kaggle!")

# Робота з масивом чисел
numbers = [1, 2, 3, 4, 5]
print("Масив чисел:", numbers)

# Обчислення суми чисел у масиві
total = sum(numbers)
print("Сума чисел:", total)

# Дані для графіка
x_plot = [1, 2, 3, 4, 5]
y_plot = [2, 4, 6, 8, 10]

# Побудова графіку з бібліотекою matplotlib
plt.plot(x_plot, y_plot)
plt.title("Простий графік")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("simple_plot.png", bbox_inches="tight")
plt.show()
print()

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 2 –  Перцептрон для XOR трьох змінних
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ПРИКЛАД: XOR для 3-х змінних (рисунок з методички)")
print("=" * 60)

# Таблиця істинності XOR(a, b, c) – усі 8 комбінацій
x = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [1, 0, 0],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
y = np.array([0, 1, 1, 1, 0, 0, 0, 1])

# Побудова моделі Перцептрону (Sequential MLP)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_dim=3, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# Компіляція та навчання
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(x, y, epochs=100, verbose=0)

# Оцінка якості
loss, accuracy = model.evaluate(x, y, verbose=0)
print("loss", loss)
print("accuracy", accuracy)

# Прогноз для кожного рядка таблиці
prediction = model.predict(x, verbose=0)
for inp, pred in zip(x, prediction):
    print(inp, round(pred[0]))

print()

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 3 – Основне завдання: Перцептрон для XOR ЧОТИРЬОХ змінних
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ЗАВДАННЯ: XOR для 4-х змінних")
print("=" * 60)

# Фіксуємо зерно випадковості для відтворюваності результату
tf.random.set_seed(0)
np.random.seed(0)

# Таблиця істинності XOR(a, b, c, d) – усі 16 комбінацій
# XOR повертає 1, якщо кількість одиниць є непарною
x4 = np.array(
    [[int(b) for b in format(i, '04b')] for i in range(16)],
    dtype=np.float32
)
y4 = np.array(
    [int(row[0]) ^ int(row[1]) ^ int(row[2]) ^ int(row[3]) for row in x4],
    dtype=np.float32
)

print("Таблиця істинності XOR(a, b, c, d):")
print(f"  {'a':>2} {'b':>2} {'c':>2} {'d':>2} | XOR")
print("  " + "-" * 18)
for row, label in zip(x4, y4):
    a, b, c, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    print(f"  {a:>2} {b:>2} {c:>2} {d:>2} |  {int(label)}")
print()

# ── Архітектура моделі ────────────────────────────────────────────────────────
# Використовується 2 прихованих шари з активацією tanh (як у прикладі з
# методички), але збільшено кількість нейронів і глибину мережі, оскільки
# 4-змінний XOR є складнішою нелінійно-несепарабельною задачею.
model4 = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_dim=4, activation="tanh"),
    tf.keras.layers.Dense(8, activation="tanh"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# Компіляція (аналогічно прикладу з методички)
model4.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Зворотній виклик: зупиняємо навчання, щойно точність досягла 100 %
class StopAt100(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy', 0) >= 1.0:
            print(f"\nДосягнуто 100% точності на епосі {epoch + 1}. Зупинка.")
            self.model.stop_training = True

print("Навчання моделі (макс. 500 епох)...")
history4 = model4.fit(
    x4, y4,
    epochs=500,
    verbose=0,
    callbacks=[StopAt100()]
)

# ── Оцінка якості ─────────────────────────────────────────────────────────────
loss4, accuracy4 = model4.evaluate(x4, y4, verbose=0)
print(f"loss     {loss4:.4f}")
print(f"accuracy {accuracy4:.4f}")
print()

# ── Прогноз для кожного рядка таблиці ────────────────────────────────────────
print("Прогнози моделі:")
print(f"  {'a':>2} {'b':>2} {'c':>2} {'d':>2} | Очік. | Прогн.")
print("  " + "-" * 28)
prediction4 = model4.predict(x4, verbose=0)
correct = 0
for inp, expected, pred in zip(x4, y4, prediction4):
    a, b, c, d = int(inp[0]), int(inp[1]), int(inp[2]), int(inp[3])
    exp_i = int(expected)
    pred_i = round(float(pred[0]))
    mark = "✓" if pred_i == exp_i else "✗"
    print(f"  {a:>2} {b:>2} {c:>2} {d:>2} |   {exp_i}   |   {pred_i}   {mark}")
    if pred_i == exp_i:
        correct += 1

print()
print(f"Правильно класифіковано: {correct}/{len(y4)}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИНА 4 – Графіки навчання (loss та accuracy по епохах)
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Графік втрат
ax1.plot(history4.history['loss'], color='crimson', linewidth=2)
ax1.set_title("Втрати (Loss) під час навчання\nXOR для 4 змінних")
ax1.set_xlabel("Епоха")
ax1.set_ylabel("Loss (binary crossentropy)")
ax1.grid(True, alpha=0.3)

# Графік точності
ax2.plot(history4.history['accuracy'], color='steelblue', linewidth=2)
ax2.set_title("Точність (Accuracy) під час навчання\nXOR для 4 змінних")
ax2.set_xlabel("Епоха")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.05)
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='100 %')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history_xor4.png", bbox_inches="tight", dpi=120)
plt.show()
print("Графік збережено: training_history_xor4.png")
