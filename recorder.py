"""
Continuous Video Recording Engine with Automatic Google Drive Backup.
Uses FFmpeg stream copy for 0% CPU consumption and AAC audio transcoding for universal mobile compatibility.
"""

import os
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from gdrive_sync import gdrive_manager

logger = logging.getLogger("recorder")
logging.basicConfig(level=logging.INFO)

RECORDINGS_DIR = Path("C:/app camara/recordings")
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("C:/app camara/recorder_config.json")
DB_PATH = Path("C:/app camara/recordings_db.json")

CAMERA_IP = "192.168.1.13"
CAMERA_PORT = "88"

class ContinuousRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.is_recording = False
        self.current_file = ""
        self.start_time = 0
        self.upload_queue: List[str] = []
        self.active_uploading = False
        
        # Opciones por defecto
        self.enabled = False
        self.stream_mode = "main"       # 'main' (HD 720p) o 'sub' (VGA)
        self.segment_duration = 900     # 15 minutos en segundos
        self.delete_after_upload = False
        self.retention_days = 7
        
        self.load_config()
        self.db: Dict[str, Any] = self.load_db()

        # Iniciar hilo de monitoreo y sincronización
        self.worker_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.worker_thread.start()

    def load_config(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.enabled = cfg.get("enabled", False)
                    self.stream_mode = cfg.get("stream_mode", "main")
                    self.segment_duration = cfg.get("segment_duration", 900)
                    self.delete_after_upload = cfg.get("delete_after_upload", False)
                    self.retention_days = cfg.get("retention_days", 7)
            except Exception as e:
                logger.error(f"Error cargando recorder_config.json: {e}")

    def save_config(self):
        with self.lock:
            data = {
                "enabled": self.enabled,
                "stream_mode": self.stream_mode,
                "segment_duration": self.segment_duration,
                "delete_after_upload": self.delete_after_upload,
                "retention_days": self.retention_days
            }
            try:
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Error guardando recorder_config.json: {e}")

    def load_db(self) -> Dict[str, Any]:
        if DB_PATH.exists():
            try:
                with open(DB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_db(self):
        with self.lock:
            try:
                with open(DB_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.db, f, indent=2)
            except Exception as e:
                logger.error(f"Error guardando recordings_db.json: {e}")

    def get_rtsp_url(self) -> str:
        stream_path = "videoMain" if self.stream_mode == "main" else "videoSub"
        return f"rtsp://admin:@{CAMERA_IP}:{CAMERA_PORT}/{stream_path}"

    def start_recording(self):
        with self.lock:
            if self.is_recording and self.proc and self.proc.poll() is None:
                logger.info("La grabación continua ya se encuentra activa.")
                return True

            self.enabled = True
            self.save_config()

            rtsp_url = self.get_rtsp_url()
            segment_pattern = str(RECORDINGS_DIR / "rec_%Y%m%d_%H%M%S.mp4")

            # Comando FFmpeg optimizado:
            # -rtsp_transport tcp: evita pérdida de paquetes
            # -c:v copy: copia directa de H264 sin decodificar (0% CPU)
            # -c:a aac: transcodifica audio PCM mulaw a AAC compatible con móviles y Drive
            # -f segment: división automática cada N segundos
            cmd = [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "64k",
                "-f", "segment",
                "-segment_time", str(self.segment_duration),
                "-segment_format", "mp4",
                "-reset_timestamps", "1",
                "-strftime", "1",
                segment_pattern
            ]

            logger.info(f"Iniciando grabación continua ({self.stream_mode.upper()} - {self.segment_duration}s por bloque)...")
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                self.is_recording = True
                self.start_time = time.time()
                return True
            except Exception as e:
                logger.error(f"Error al iniciar subproceso FFmpeg de grabación: {e}")
                self.is_recording = False
                return False

    def stop_recording(self):
        with self.lock:
            self.enabled = False
            self.save_config()
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=3)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None
            self.is_recording = False
            logger.info("Grabación continua detenida.")
            return True

    def _monitor_loop(self):
        """Monitorea el proceso de grabación y procesa archivos terminados para Google Drive."""
        time.sleep(2)
        # Si estaba configurado como activo, iniciar
        if self.enabled:
            self.start_recording()

        while True:
            try:
                # 1. Comprobar salud del subproceso si debe estar grabando
                if self.enabled:
                    if not self.proc or self.proc.poll() is not None:
                        logger.warning("El proceso de grabación se detuvo. Reiniciando en 3s...")
                        time.sleep(3)
                        self.start_recording()

                # 2. Escanear directorio de grabaciones
                files = sorted(
                    RECORDINGS_DIR.glob("*.mp4"),
                    key=lambda p: p.stat().st_mtime
                )

                now = time.time()
                for f in files:
                    fname = f.name
                    # Evitar procesar el archivo que actualmente se está grabando
                    # Si fue modificado hace más de 10 segundos y tiene tamaño > 10KB, está cerrado
                    mtime = f.stat().st_mtime
                    size = f.stat().st_size

                    if fname not in self.db:
                        # Si es el archivo más reciente y tiene menos de 10s de inactividad, sigue abierto
                        if f == files[-1] and (now - mtime) < 15 and self.is_recording:
                            self.current_file = fname
                            continue

                        if size > 10240: # Mayor a 10KB
                            dt_str = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                            self.db[fname] = {
                                "filename": fname,
                                "path": str(f),
                                "size": size,
                                "created_at": dt_str,
                                "status": "pending_upload" if gdrive_manager.auto_upload else "local",
                                "drive_id": None,
                                "drive_url": None
                            }
                            self.save_db()
                            if gdrive_manager.auto_upload and fname not in self.upload_queue:
                                self.upload_queue.append(fname)

                # 3. Procesar cola de subida a Google Drive
                if self.upload_queue and not self.active_uploading and gdrive_manager.is_connected:
                    next_file = self.upload_queue.pop(0)
                    threading.Thread(target=self._upload_worker, args=(next_file,), daemon=True).start()

            except Exception as e:
                logger.error(f"Error en monitor de grabación: {e}")

            time.sleep(5)

    def _upload_worker(self, filename: str):
        self.active_uploading = True
        try:
            file_path = RECORDINGS_DIR / filename
            if not file_path.exists():
                return

            logger.info(f"Subiendo grabación a Google Drive: {filename}...")
            res = gdrive_manager.upload_file(str(file_path))
            if res and "id" in res:
                drive_id = res.get("id")
                drive_url = res.get("webViewLink") or f"https://drive.google.com/file/d/{drive_id}/view"
                
                with self.lock:
                    if filename in self.db:
                        self.db[filename]["status"] = "uploaded"
                        self.db[filename]["drive_id"] = drive_id
                        self.db[filename]["drive_url"] = drive_url
                        self.save_db()

                logger.info(f"✅ Grabación subida a Google Drive: {filename} -> {drive_url}")

                # Si está configurado borrar local tras subir
                if self.delete_after_upload:
                    try:
                        file_path.unlink()
                        with self.lock:
                            if filename in self.db:
                                self.db[filename]["status"] = "cloud_only"
                                self.save_db()
                        logger.info(f"Archivo local eliminado tras subida a Drive: {filename}")
                    except Exception as err:
                        logger.warning(f"No se pudo eliminar archivo local: {err}")

        except Exception as e:
            logger.error(f"Error subiendo {filename} a Google Drive: {e}")
            with self.lock:
                if filename in self.db:
                    self.db[filename]["status"] = "upload_error"
                    self.db[filename]["last_error"] = str(e)
                    self.save_db()
        finally:
            self.active_uploading = False

    def get_status(self) -> Dict[str, Any]:
        drive_status = gdrive_manager.test_connection() if gdrive_manager.get_service() else {
            "connected": False,
            "message": "Sin credenciales configuradas"
        }

        # Calcular espacio ocupado localmente
        total_local_size = sum(f.stat().st_size for f in RECORDINGS_DIR.glob("rec_*.mp4") if f.exists())
        total_local_gb = round(total_local_size / (1024**3), 2)

        return {
            "is_recording": self.is_recording,
            "enabled": self.enabled,
            "stream_mode": self.stream_mode,
            "segment_duration": self.segment_duration,
            "delete_after_upload": self.delete_after_upload,
            "current_file": self.current_file if self.is_recording else None,
            "local_recordings_count": len(list(RECORDINGS_DIR.glob("rec_*.mp4"))),
            "local_storage_gb": total_local_gb,
            "pending_uploads": len(self.upload_queue),
            "active_uploading": self.active_uploading,
            "gdrive": drive_status
        }

    def list_recordings(self, limit: int = 50) -> List[Dict[str, Any]]:
        items = list(self.db.values())
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items[:limit]

    def delete_recording(self, filename: str) -> bool:
        with self.lock:
            local_path = RECORDINGS_DIR / filename
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass
            if filename in self.db:
                del self.db[filename]
                self.save_db()
            return True

recorder_manager = ContinuousRecorder()
