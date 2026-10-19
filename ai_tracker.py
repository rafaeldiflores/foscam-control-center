"""
AI Vision & Autonomous PTZ Motor Auto-Tracking Controller for Foscam FI9821W V2.
Coordinates real-time detection, proportional motor jogging, and Telegram alert dispatch.
"""

import io
import json
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from ai_detector import ai_detector, CLASS_GROUPS
from telegram_alerts import telegram_notifier

logger = logging.getLogger("ai_tracker")
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("C:/app camara/ai_tracker_config.json")

class AITrackerController:
    def __init__(self, camera=None, stream_worker=None):
        self.camera = camera
        self.stream_worker = stream_worker
        self.lock = threading.Lock()
        
        # Opciones configurables
        self.ai_enabled = True
        self.tracking_enabled = False
        self.target_classes = ["person", "vehicle"]
        self.min_confidence = 0.65
        self.return_home_enabled = True
        self.home_preset = "center"
        self.deadzone_x = 0.16   # Margen horizontal antes de mover motor
        self.deadzone_y = 0.22   # Margen vertical antes de mover motor

        # Estado en tiempo real
        self.is_running = False
        self.last_target_time = 0
        self.last_motor_command_time = 0
        self.returned_home = True
        self.current_detections: List[Dict[str, Any]] = []
        self.last_annotated_jpeg: Optional[bytes] = None
        self.fps = 0.0
        self.inference_ms = 0.0

        self.load_config()

        # Iniciar hilo en segundo plano
        self.worker_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.worker_thread.start()

    def set_camera_and_worker(self, camera, stream_worker):
        self.camera = camera
        self.stream_worker = stream_worker

    def load_config(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ai_enabled = data.get("ai_enabled", True)
                    self.tracking_enabled = data.get("tracking_enabled", False)
                    self.target_classes = data.get("target_classes", ["person", "vehicle"])
                    self.min_confidence = data.get("min_confidence", 0.65)
                    self.return_home_enabled = data.get("return_home_enabled", True)
                    self.home_preset = data.get("home_preset", "center")
                    self.deadzone_x = data.get("deadzone_x", 0.16)
                    self.deadzone_y = data.get("deadzone_y", 0.22)
            except Exception as e:
                logger.error(f"Error cargando ai_tracker_config.json: {e}")

    def save_config(self):
        with self.lock:
            data = {
                "ai_enabled": self.ai_enabled,
                "tracking_enabled": self.tracking_enabled,
                "target_classes": self.target_classes,
                "min_confidence": self.min_confidence,
                "return_home_enabled": self.return_home_enabled,
                "home_preset": self.home_preset,
                "deadzone_x": self.deadzone_x,
                "deadzone_y": self.deadzone_y
            }
            try:
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Error guardando ai_tracker_config.json: {e}")

    def get_allowed_classes(self) -> List[str]:
        classes = []
        for grp in self.target_classes:
            classes.extend(CLASS_GROUPS.get(grp, [grp]))
        return classes

    def _tracking_loop(self):
        logger.info("Iniciando bucle de IA y Auto-Tracking PTZ...")
        self.is_running = True
        frame_count = 0
        fps_start = time.time()

        while True:
            try:
                if not self.ai_enabled:
                    time.sleep(0.5)
                    continue

                # 1. Obtener el fotograma más reciente del RTSP Stream Worker
                jpeg_bytes = None
                if self.stream_worker:
                    jpeg_bytes = self.stream_worker.get_latest_frame()

                if not jpeg_bytes and self.camera:
                    jpeg_bytes = self.camera.get_immediate_frame()

                if not jpeg_bytes:
                    time.sleep(0.1)
                    continue

                # 2. Cargar imagen con Pillow
                try:
                    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                except Exception:
                    time.sleep(0.05)
                    continue

                # 3. Inferencia de detección con clases permitidas
                allowed = self.get_allowed_classes()
                detections = ai_detector.detect(img, conf_threshold=self.min_confidence, allowed_classes=allowed)
                self.current_detections = detections

                now = time.time()
                frame_count += 1
                if (now - fps_start) >= 2.0:
                    self.fps = round(frame_count / (now - fps_start), 1)
                    frame_count = 0
                    fps_start = now

                if detections:
                    self.inference_ms = detections[0].get("inference_ms", 0.0)
                    self.last_target_time = now
                    self.returned_home = False

                    # Generar imagen anotada
                    annotated_jpeg = ai_detector.annotate_and_get_jpeg(img, detections)
                    self.last_annotated_jpeg = annotated_jpeg

                    # 4. Disparar alerta en Telegram si corresponde
                    telegram_notifier.trigger_ai_detection_alert(annotated_jpeg, detections)

                    # 5. Lógica de Auto-Tracking con Motores PTZ
                    if self.tracking_enabled and self.camera:
                        self._process_auto_tracking(detections)

                else:
                    # Sin detecciones: comprobar si debe regresar a la posición de guardia
                    if self.tracking_enabled and self.return_home_enabled and not self.returned_home:
                        if (now - self.last_target_time) > 12.0: # 12 segundos sin presencia
                            logger.info("Objetivo ausente por >12s. Regresando a Posición de Guardia...")
                            self._return_home()
                            self.returned_home = True

                # Mantener un sampling moderado de 4 a 6 FPS (150ms de pausa)
                time.sleep(0.15)

            except Exception as e:
                logger.error(f"Error en bucle de IA Tracker: {e}")
                time.sleep(0.5)

    def _process_auto_tracking(self, detections: List[Dict[str, Any]]):
        """Calcula el error respecto al centro y acciona los motores paso a paso."""
        now = time.time()
        # Rate-limiter: al menos 450ms entre comandos de motor para permitir desplazamiento físico
        if (now - self.last_motor_command_time) < 0.45:
            return

        # Priorizar 'person' si existe, de lo contrario el de mayor confianza
        persons = [d for d in detections if d["class_name"] == "person"]
        target = persons[0] if persons else detections[0]

        norm_cx, norm_cy = target["norm_center"]
        err_x = norm_cx - 0.5   # Negativo: Izquierda, Positivo: Derecha
        err_y = norm_cy - 0.5   # Negativo: Arriba, Positivo: Abajo

        moved = False

        # Corrección Horizontal (Pan)
        if abs(err_x) > self.deadzone_x:
            direction = "left" if err_x < 0 else "right"
            # Pulso proporcional según distancia al centro (120ms a 220ms)
            duration = int(120 + min(120, abs(err_x) * 200))
            self.camera.ptz_step(direction, duration_ms=duration)
            moved = True
            logger.info(f"Auto-Tracking PAN -> {direction.upper()} (err: {err_x:.2f}, dur: {duration}ms)")

        # Corrección Vertical (Tilt) si el error es significativo
        if not moved and abs(err_y) > self.deadzone_y:
            direction = "up" if err_y < 0 else "down"
            duration = int(100 + min(100, abs(err_y) * 150))
            self.camera.ptz_step(direction, duration_ms=duration)
            moved = True
            logger.info(f"Auto-Tracking TILT -> {direction.upper()} (err: {err_y:.2f}, dur: {duration}ms)")

        if moved:
            self.last_motor_command_time = time.time()

    def _return_home(self):
        if not self.camera:
            return
        if self.home_preset == "center":
            self.camera.ptz_reset()
        else:
            self.camera.cgi_request("ptzGotoPresetPoint", {"name": self.home_preset})

    def get_status(self) -> Dict[str, Any]:
        return {
            "ai_enabled": self.ai_enabled,
            "tracking_enabled": self.tracking_enabled,
            "target_classes": self.target_classes,
            "min_confidence": self.min_confidence,
            "return_home_enabled": self.return_home_enabled,
            "home_preset": self.home_preset,
            "fps": self.fps,
            "inference_ms": self.inference_ms,
            "detections_count": len(self.current_detections),
            "detections": self.current_detections,
            "telegram": {
                "enabled": telegram_notifier.enabled,
                "configured": bool(telegram_notifier.bot_token and telegram_notifier.chat_id),
                "is_silenced": telegram_notifier.is_silenced(),
                "cooldown_seconds": telegram_notifier.cooldown_seconds
            }
        }

ai_tracker = AITrackerController()
