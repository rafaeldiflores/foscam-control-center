"""
Telegram Security Alert Notifier for Foscam IP Camera with AI Vision.
Sends instant push notifications with annotated snapshots and interactive action buttons.
"""

import io
import json
import time
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("telegram_alerts")
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("C:/app camara/telegram_config.json")

class TelegramNotifier:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.enabled = False
        self.bot_token = ""
        self.chat_id = ""
        self.cooldown_seconds = 45
        self.last_alert_time = 0
        self.silenced_until = 0
        self.alert_person = True
        self.alert_vehicle = True
        self.alert_pet = False
        self.min_confidence = 0.65
        self.include_photo = True
        self.app_url = "https://intl-specialty-concrete-feature.trycloudflare.com"
        self.web_url = self.app_url
        self.load_config()

    def load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.enabled = data.get("enabled", False)
                    self.bot_token = data.get("bot_token", "")
                    self.chat_id = data.get("chat_id", "")
                    self.cooldown_seconds = data.get("cooldown_seconds", 45)
                    self.include_photo = data.get("include_photo", True)
                    self.alert_person = data.get("alert_person", True)
                    self.alert_vehicle = data.get("alert_vehicle", True)
                    self.alert_pet = data.get("alert_pet", False)
                    self.min_confidence = data.get("min_confidence", 0.65)
                    self.app_url = data.get("app_url") or data.get("web_url", "https://intl-specialty-concrete-feature.trycloudflare.com")
                    self.web_url = self.app_url
            except Exception as e:
                logger.error(f"Error cargando telegram_config.json: {e}")

    def save_config(self):
        data = {
            "enabled": self.enabled,
            "bot_token": self.bot_token,
            "chat_id": self.chat_id,
            "cooldown_seconds": self.cooldown_seconds,
            "include_photo": self.include_photo,
            "alert_person": self.alert_person,
            "alert_vehicle": self.alert_vehicle,
            "alert_pet": self.alert_pet,
            "min_confidence": self.min_confidence,
            "app_url": self.app_url,
            "web_url": self.app_url
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando telegram_config.json: {e}")

    def is_silenced(self) -> bool:
        return time.time() < self.silenced_until

    def silence(self, minutes: int = 30):
        self.silenced_until = time.time() + (minutes * 60)
        logger.info(f"Alertas de Telegram silenciadas por {minutes} minutos.")

    def resume(self):
        self.silenced_until = 0

    def get_api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}/{method}"

    def send_message(self, text: str, photo_bytes: Optional[bytes] = None, buttons: Optional[List[List[Dict[str, str]]]] = None) -> Dict[str, Any]:
        if not self.bot_token or not self.chat_id:
            logger.warning("Bot token o Chat ID no configurados en Telegram.")
            return {"ok": False, "description": "Token o Chat ID no configurados"}

        if photo_bytes and self.include_photo:
            success = self.send_photo(photo_bytes, text, buttons=buttons)
            return {"ok": success, "description": "Enviado con foto" if success else "Error enviando foto"}

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        if buttons:
            payload["reply_markup"] = json.dumps({"inline_keyboard": buttons})

        try:
            data = urllib.parse.urlencode(payload).encode("utf-8")
            req = urllib.request.Request(self.get_api_url("sendMessage"), data=data)
            with urllib.request.urlopen(req, timeout=8) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res
        except Exception as e:
            logger.error(f"Error enviando mensaje a Telegram: {e}")
            return {"ok": False, "description": str(e)}

    def send_photo(self, image_bytes: bytes, caption: str, filename: str = "alert.jpg", buttons: Optional[List[List[Dict[str, str]]]] = None) -> bool:
        if not self.bot_token or not self.chat_id:
            return False

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = io.BytesIO()

        # Chat ID
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
        body.write(f"{self.chat_id}\r\n".encode())

        # Caption
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
        body.write(caption.encode("utf-8"))
        body.write(b"\r\n")

        # Parse Mode
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="parse_mode"\r\n\r\n')
        body.write(b"HTML\r\n")

        # Buttons (reply_markup)
        if buttons:
            body.write(f"--{boundary}\r\n".encode())
            body.write(b'Content-Disposition: form-data; name="reply_markup"\r\n\r\n')
            body.write(json.dumps({"inline_keyboard": buttons}).encode("utf-8"))
            body.write(b"\r\n")

        # Photo File
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'.encode())
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(image_bytes)
        body.write(b"\r\n")

        # End
        body.write(f"--{boundary}--\r\n".encode())
        payload = body.getvalue()

        try:
            req = urllib.request.Request(
                self.get_api_url("sendPhoto"),
                data=payload,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("ok", False)
        except Exception as e:
            logger.error(f"Error enviando foto a Telegram: {e}")
            return False

    def trigger_ai_detection_alert(self, image_bytes: bytes, detections: List[Dict[str, Any]]) -> bool:
        """Envía alerta de detección con foto y teclado de acción rápida si el cooldown lo permite."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False

        if self.is_silenced():
            logger.debug("Alerta ignorada: Telegram en modo silencioso.")
            return False

        now = time.time()
        if (now - self.last_alert_time) < self.cooldown_seconds:
            return False

        self.last_alert_time = now

        # Resumen de detecciones
        det_lines = []
        for d in detections:
            emoji = "👤" if d["class_name"] == "person" else ("🚗" if d["class_name"] in ["car", "motorcycle", "bus", "truck"] else "🐾")
            det_lines.append(f"{emoji} <b>{d['class_name'].capitalize()}</b> ({int(d['confidence'] * 100)}%)")

        caption = (
            f"🚨 <b>ALERTA DE SEGURIDAD - Foscam</b>\n\n"
            f"<b>Detección IA:</b>\n" + "\n".join(det_lines) + "\n\n"
            f"📍 <b>Cámara:</b> Foscam FI9821W\n"
            f"🕒 <b>Hora:</b> {time.strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"🎯 <b>Auto-Tracking:</b> Motores activos siguiendo al objetivo"
        )

        buttons = [
            [
                {"text": "📹 Abrir App en Celular", "url": self.web_url}
            ]
        ]

        logger.info(f"Enviando alerta a Telegram ({len(detections)} detecciones)...")
        return self.send_photo(image_bytes, caption, buttons=buttons)

    def test_connection(self) -> Dict[str, Any]:
        """Prueba las credenciales enviando un mensaje directo."""
        if not self.bot_token:
            return {"ok": False, "message": "Falta ingresar el Token del Bot"}
        if not self.chat_id:
            return {"ok": False, "message": "Falta ingresar el Chat ID"}

        test_msg = (
            f"🔔 <b>Foscam Control Center</b>\n\n"
            f"✅ ¡Conexión con Telegram configurada exitosamente!\n"
            f"Recibirás aquí fotos y alertas automáticas cada vez que la IA detecte personas o vehículos."
        )
        buttons = [
            [{"text": "📹 Abrir Centro de Control", "url": self.web_url}]
        ]
        ok = self.send_message(test_msg, buttons=buttons)
        return {
            "ok": ok,
            "message": "Mensaje de prueba enviado a Telegram" if ok else "Fallo al enviar mensaje. Revisa el Token y Chat ID"
        }

telegram_notifier = TelegramNotifier()
