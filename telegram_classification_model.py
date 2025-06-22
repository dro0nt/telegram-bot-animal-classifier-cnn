import io
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Параметры
MODEL_PATH = 'cub_birds_classifier_model.keras'
CLASS_NAMES_PATH = 'class_names.txt'
IMG_SIZE = (224, 224)

# Загрузка модели и классов
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return tf.expand_dims(img_array, axis=0)

def plot_top_predictions(image: Image.Image, preds, top=5):
    top_indices = np.argsort(preds[0])[::-1][:top]
    top_probs = preds[0][top_indices]
    top_labels = [class_names[i] for i in top_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title("Присланное фото")

    ax2.barh(range(top), top_probs[::-1], color='skyblue')
    ax2.set_yticks(range(top))
    ax2.set_yticklabels(top_labels[::-1])
    ax2.invert_yaxis()
    ax2.set_xlabel("Вероятность")
    ax2.set_title("Топ-5 предсказаний")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я — бот, обученный распознавать 200 видов птиц по фото 🐦\n\n"
        "📸 Просто отправь мне фотографию птицы — я постараюсь определить её вид и покажу наиболее вероятные варианты.\n\n"
        "❓ Для получения списка команд — введи /help.\n\n"
        "🔍 Обрати внимание: бот обучен на ограниченном наборе данных, поэтому может не распознать редкие или экзотические виды."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Команды бота:\n"
        "/start - начать общение\n"
        "/help - показать это сообщение\n"
        "/classes - показать список известных видов птиц\n"
        "Просто пришли фото птицы для распознавания."
    )
    await update.message.reply_text(help_text)

async def classes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chunk_size = 50
    total = len(class_names)
    chunks = [class_names[i:i + chunk_size] for i in range(0, total, chunk_size)]

    for i, chunk in enumerate(chunks):
        text = f"Список птиц ({i*chunk_size+1}-{i*chunk_size+len(chunk)} из {total}):\n"
        text += '\n'.join(chunk)
        await update.message.reply_text(text)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        byte_array = await photo_file.download_as_bytearray()
        image = Image.open(io.BytesIO(byte_array)).convert('RGB')

        input_tensor = preprocess_image(image)
        preds = model.predict(input_tensor)

        top_index = preds[0].argmax()
        top_class = class_names[top_index]
        top_prob = preds[0][top_index]

        result_buf = plot_top_predictions(image, preds)

        caption_text = f"Я думаю, что это — {top_class}.\nВероятность: {top_prob:.2%}"

        await update.message.reply_photo(photo=result_buf, caption=caption_text)
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения. Попробуйте ещё раз.")

async def handle_non_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пожалуйста, пришли фотографию птицы для распознавания.")

def main():
    TOKEN = 'TOKEN'  # <-- Замени на свой токен
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("classes", classes_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(~filters.PHOTO, handle_non_photo))  # Всё, что не фото

    logger.info("Бот запущен")
    app.run_polling()

if __name__ == '__main__':
    main()