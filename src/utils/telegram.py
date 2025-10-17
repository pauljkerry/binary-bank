from dotenv import load_dotenv
import os
import requests

load_dotenv(dotenv_path="../../.env")

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_message(message: str) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
        r.raise_for_status()                      # HTTP 4xx/5xx を例外化
        data = r.json()
        if not data.get("ok", True):              # Bot APIレベルの失敗を検知
            raise RuntimeError(data.get("description", "Telegram API error"))
        print("✅ Message sent.")
    except Exception as e:
        print(f"⚠️ Failed to send message: {e}")


def send_document(path: str, caption: str = "") -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
    with open(path, "rb") as f:
        files = {"document": (os.path.basename(path), f)}
        data = {"chat_id": CHAT_ID, "caption": caption}
        try:
            r = requests.post(url, data=data, files=files, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok", True):
                raise RuntimeError(data.get("description", "Telegram API error"))
            print("✅ Document sent.")
        except Exception as e:
            print(f"⚠️ Failed to send document: {e}")