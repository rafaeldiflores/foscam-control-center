"""
Google Drive Cloud Synchronization Engine for Foscam Security Recordings.
Supports Service Account JSON, OAuth2 User Credentials, and Token-based Auth.
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("gdrive_sync")
logging.basicConfig(level=logging.INFO)

CONFIG_PATH = Path("C:/app camara/gdrive_config.json")
CREDENTIALS_DIR = Path("C:/app camara/credentials")
CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)

SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive'
]

class GDriveSyncManager:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.lock = threading.Lock()
        self.service = None
        self.is_connected = False
        self.user_email = ""
        self.quota_total = 0
        self.quota_used = 0
        self.folder_id = ""
        self.folder_name = "Foscam_Grabaciones"
        self.auto_upload = True
        self.delete_after_upload = False
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        with self.lock:
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.folder_id = data.get("folder_id", "")
                        self.folder_name = data.get("folder_name", "Foscam_Grabaciones")
                        self.auto_upload = data.get("auto_upload", True)
                        self.delete_after_upload = data.get("delete_after_upload", False)
                        self.user_email = data.get("user_email", "")
                        return data
                except Exception as e:
                    logger.error(f"Error cargando gdrive_config.json: {e}")
            return {
                "folder_id": "",
                "folder_name": "Foscam_Grabaciones",
                "auto_upload": True,
                "delete_after_upload": False,
                "user_email": ""
            }

    def save_config(self, extra: Optional[Dict[str, Any]] = None):
        with self.lock:
            data = {
                "folder_id": self.folder_id,
                "folder_name": self.folder_name,
                "auto_upload": self.auto_upload,
                "delete_after_upload": self.delete_after_upload,
                "user_email": self.user_email
            }
            if extra:
                data.update(extra)
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Error guardando gdrive_config.json: {e}")

    def save_credentials_data(self, content_str: str, filename: str = "service_account.json") -> Dict[str, Any]:
        """Guarda credenciales JSON subidas por el usuario."""
        try:
            # Validar que sea un JSON válido
            parsed = json.loads(content_str)
            target_path = CREDENTIALS_DIR / filename
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            
            logger.info(f"Credenciales guardadas exitosamente en {target_path}")
            return {"status": "ok", "path": str(target_path), "type": parsed.get("type", "unknown")}
        except Exception as e:
            logger.error(f"Error guardando credenciales: {e}")
            raise ValueError(f"Formato JSON inválido: {e}")

    def get_service(self):
        """Inicializa el cliente de Google Drive API."""
        if self.service:
            return self.service

        from googleapiclient.discovery import build

        # 1. Probar Service Account JSON
        sa_path = CREDENTIALS_DIR / "service_account.json"
        if sa_path.exists():
            try:
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(
                    str(sa_path), scopes=SCOPES
                )
                self.service = build('drive', 'v3', credentials=creds)
                with open(sa_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.user_email = data.get("client_email", "Service Account")
                self.is_connected = True
                logger.info(f"Google Drive conectado vía Service Account: {self.user_email}")
                return self.service
            except Exception as e:
                logger.error(f"Error inicializando Service Account: {e}")

        # 2. Probar Token OAuth2 guardado
        token_path = CREDENTIALS_DIR / "token.json"
        if token_path.exists():
            try:
                from google.oauth2.credentials import Credentials
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                    with open(token_path, "w", encoding="utf-8") as f:
                        f.write(creds.to_json())
                
                self.service = build('drive', 'v3', credentials=creds)
                self.is_connected = True
                logger.info("Google Drive conectado vía OAuth2 User Token")
                return self.service
            except Exception as e:
                logger.error(f"Error inicializando OAuth2 token: {e}")

        # 3. Probar credentials.json de escritorio
        client_secrets_path = CREDENTIALS_DIR / "credentials.json"
        if client_secrets_path.exists():
            logger.info("Archivo credentials.json detectado; requiere flujo de autorización")

        self.is_connected = False
        return None

    def test_connection(self) -> Dict[str, Any]:
        """Verifica la conexión y consulta la cuota disponible (5 TB)."""
        try:
            service = self.get_service()
            if not service:
                return {
                    "connected": False,
                    "message": "No se encontraron credenciales válidas en credentials/"
                }

            about = service.about().get(fields="user,storageQuota").execute()
            user_info = about.get("user", {})
            quota_info = about.get("storageQuota", {})

            email = user_info.get("emailAddress", self.user_email or "Conectado")
            self.user_email = email

            limit = int(quota_info.get("limit", 0))
            usage = int(quota_info.get("usage", 0))
            usage_in_drive = int(quota_info.get("usageInDrive", 0))

            self.quota_total = limit
            self.quota_used = usage

            # Si es Service Account o sin límite explícito, indicar almacenamiento activo
            quota_total_gb = round(limit / (1024**3), 2) if limit > 0 else 5120.0  # 5 TB por defecto
            quota_used_gb = round(usage / (1024**3), 2) if usage > 0 else round(usage_in_drive / (1024**3), 2)
            pct = round((quota_used_gb / quota_total_gb) * 100, 1) if quota_total_gb > 0 else 0

            self.is_connected = True
            self.save_config({"user_email": email})

            return {
                "connected": True,
                "email": email,
                "displayName": user_info.get("displayName", "Google Drive"),
                "quota_total_gb": quota_total_gb,
                "quota_used_gb": quota_used_gb,
                "quota_percentage": pct,
                "folder_name": self.folder_name,
                "folder_id": self.folder_id or "Auto (Se creará en la primera subida)"
            }
        except Exception as e:
            logger.error(f"Error testeando conexión Google Drive: {e}")
            self.is_connected = False
            return {
                "connected": False,
                "error": str(e),
                "message": f"Fallo al conectar con Google Drive: {str(e)}"
            }

    def ensure_folder(self) -> str:
        """Asegura que exista la carpeta de destino en Google Drive."""
        service = self.get_service()
        if not service:
            raise RuntimeError("Google Drive no está conectado")

        if self.folder_id:
            try:
                folder = service.files().get(fileId=self.folder_id, fields="id,trashed").execute()
                if not folder.get("trashed", False):
                    return self.folder_id
            except Exception:
                pass

        # Buscar carpeta por nombre
        query = f"mimeType='application/vnd.google-apps.folder' and name='{self.folder_name}' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])

        if files:
            self.folder_id = files[0]['id']
            self.save_config()
            return self.folder_id

        # Crear carpeta nueva
        file_metadata = {
            'name': self.folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        self.folder_id = folder.get('id')
        self.save_config()
        logger.info(f"Carpeta creada en Google Drive: '{self.folder_name}' (ID: {self.folder_id})")
        return self.folder_id

    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """Sube un archivo de video a Google Drive."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        service = self.get_service()
        if not service:
            raise RuntimeError("Google Drive no está configurado")

        from googleapiclient.http import MediaFileUpload

        folder_id = self.ensure_folder()
        file_metadata = {
            'name': path.name,
            'parents': [folder_id]
        }

        # Subida resumable en chunks de 5MB
        media = MediaFileUpload(
            str(path),
            mimetype='video/mp4',
            resumable=True,
            chunksize=5 * 1024 * 1024
        )

        request = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, size, webViewLink, webContentLink'
        )

        logger.info(f"Iniciando subida a Google Drive: {path.name} ({round(path.stat().st_size / (1024**2), 2)} MB)...")
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f"Subiendo {path.name}: {int(status.progress() * 100)}%")

        logger.info(f"✅ Archivo subido con éxito: {response.get('name')} (ID: {response.get('id')})")
        return response

    def list_files(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Lista las grabaciones almacenadas en la nube."""
        service = self.get_service()
        if not service:
            return []

        folder_id = self.ensure_folder()
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            pageSize=limit,
            fields="files(id, name, size, createdTime, webViewLink, webContentLink, thumbnailLink)",
            orderBy="createdTime desc"
        ).execute()

        items = results.get('files', [])
        for item in items:
            size_bytes = int(item.get("size", 0))
            item["size_mb"] = round(size_bytes / (1024**2), 2)
        return items

# Instancia global
gdrive_manager = GDriveSyncManager()
