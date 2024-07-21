import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import google.generativeai as genai
from pydub import AudioSegment  # to convert from .oga to .wav
from io import BytesIO
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import whisper
from google.cloud import storage, secretmanager
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from telegram.ext import Application
import asyncio 
app = FastAPI()

def access_secret_version(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

telegram_bot_token=access_secret_version('174627504804','secret_access_to_storage','4')
service_account_json = access_secret_version('174627504804', 'secret_access_to_storage', '1')  # for storage
storage_client = storage.Client(service_account_json)
bucket_name = 'attempticus1'
speaking_model = whisper.load_model("base")
genai.configure(api_key=access_secret_version('174627504804', 'secret_access_to_storage', '2'))  # api token of gimini model
model = genai.GenerativeModel('gemini-1.5-flash')  # my gimini model
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="можете начать разговор через голосовой или текстовой чат, по возможности, оскорбления автоматически игнорируются")

async def voice_install(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Обрабатываю голосовое сообщение...")

    voice_file = await update.message.voice.get_file()
    voice_data = await voice_file.download_as_bytearray()  # our voice data
    audio = AudioSegment.from_ogg(BytesIO(voice_data))
    
    audio_data = BytesIO()
    audio.export(audio_data, format='wav')

    audio_data.seek(0)
    original_sample_rate, numpy = wavfile.read(audio_data)
    new_sample_rate = 16000
    number_of_samples = round(len(numpy) * float(new_sample_rate) / original_sample_rate)
    resampled_data = resample(numpy, number_of_samples)

    if numpy.dtype == np.int16:
        new_bit_depth_data = resampled_data.astype(np.int16)
    else:
        # Scale and convert to 16-bit
        new_bit_depth_data = (resampled_data / np.max(np.abs(resampled_data)) * 32767).astype(np.int16)

    buferus = BytesIO()
    wavfile.write(buferus, new_sample_rate, new_bit_depth_data)
    
    buferus = buferus.getbuffer()
    buferus = buferus.tobytes()

    result = speaking_model.transcribe(buferus)
    message = result["text"]

    await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Я услышал: {message}')

    response = model.generate_content(f"{message}", safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    })

    response = response.text

    await context.bot.send_message(chat_id=update.effective_chat.id, text='Ответ бота: ' + response)

async def botvetchick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    response = model.generate_content(f"{message}", safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    })
    message = response.text
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

async def start_telegram_bot():
    application = ApplicationBuilder().token(f'{telegram_bot_token}').post_init(post_init).build()

    start_handler = CommandHandler('start', start)
    voice_installer = MessageHandler(filters.VOICE & (~filters.FORWARDED), voice_install)
    botvetchick_handler = MessageHandler(filters.TEXT & (~filters.FORWARDED), botvetchick)

    application.add_handler(voice_installer)
    application.add_handler(start_handler)
    application.add_handler(botvetchick_handler)

    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    return application

def post_init(application: Application):
    application.job_queue.scheduler.configure(event_loop=asyncio.get_event_loop())

@asynccontextmanager
async def lifespan(app: FastAPI):
    application = await start_telegram_bot()
    yield
    await application.stop()
    await application.shutdown()

app.router.lifespan_context = lifespan

@app.get("/")
def read_root():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
