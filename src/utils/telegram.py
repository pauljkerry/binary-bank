from pathlib import Path
from dotenv import load_dotenv
import os
import requests

env_path = Path.cwd().parent / ".env"
load_dotenv(dotenv_path=env_path)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_message(message):
    """
    telegramでメッセージを送信する関数

    Parameters
    ----------
    message : str
        送信するメッセージ
    """
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    response = requests.post(url, data=data)
    if response.ok:
        print("✅ Message sent successfully.")
    else:
        print("❌ Failed to send message:", response.text)