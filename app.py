import asyncio
import io
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response, Request, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from talk_protocol import FoscamTalkSession, generate_sound_effect, transcode_audio_to_pcm
from recorder import recorder_manager, RECORDINGS_DIR
from gdrive_sync import gdrive_manager, CREDENTIALS_DIR
from ai_tracker import ai_tracker
from ai_detector import ai_detector
from telegram_alerts import telegram_notifier

app = FastAPI(title="Foscam Control Center", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
STATIC_DIR = BASE_DIR / "static"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

def is_go2rtc_available() -> bool:
    """Checks if local go2rtc proxy is responding on RTSP port 8554."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.15)
            s.connect(("127.0.0.1", 8554))
            return True
    except Exception:
        return False

def ensure_go2rtc_daemon():
    """Starts local go2rtc binary in background if not already active."""
    if not is_go2rtc_available():
        go2rtc_exe = BASE_DIR / "go2rtc.exe"
        go2rtc_yaml = BASE_DIR / "go2rtc.yaml"
        if go2rtc_exe.exists() and go2rtc_yaml.exists():
            try:
                subprocess.Popen(
                    [str(go2rtc_exe), "-config", str(go2rtc_yaml)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
            except Exception:
                pass

ensure_go2rtc_daemon()

class FoscamCamera:
    def __init__(self, ip: str = "192.168.1.13", port: int = 88, usr: str = "admin", pwd: str = ""):
        self.ip = ip
        self.port = port
        self.usr = usr
        self.pwd = pwd
        self.last_seen = time.time()
        self.mac = "00626E4EBBF1"
        self.model = "FI9821W V2"
        self.is_online = True
        
        # Concurrency serialization lock for camera HTTP port 88
        self.cgi_lock = threading.Lock()
        self.ptz_motion_lock = threading.Lock()
        self.ptz_abort_event = threading.Event()
        
        # In-memory cache to prevent camera embedded web server starvation
        self.cached_info = {
            "productName": "FI9821W V2",
            "devName": "FI9821W",
            "mac": "00626E4EBBF1",
            "firmwareVer": "1.11.1.18",
            "hardwareVer": "1.4.1.8"
        }
        self.cached_ports = {
            "httpPort": "88",
            "httpsPort": "443",
            "rtspPort": "88",
            "mediaPort": "88"
        }
        self.cached_state = {
            "motionDetectAlarm": "0",
            "infraLedState": "0",
            "ioAlarm": "0"
        }
        self.cached_wifi = {
            "connected": True,
            "ssid": "CASA DE LA LOCURA",
            "signal": 94,
            "quality": "Excelente",
            "bars": 4,
            "bssid": "D4:01:68:BE:20:00"
        }
        
        # Estimated 3D spatial orientation in degrees (Pan: -150 to +150, Tilt: -20 to +90)
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.ptz_speed = 1

        # Start gentle background state poller
        self.poller_thread = threading.Thread(target=self._state_refresher, daemon=True)
        self.poller_thread.start()

    def get_wifi_details(self) -> Dict[str, Any]:
        """Parses active Wi-Fi configuration and real-time signal percentage from camera."""
        try:
            conf = self.cgi_request("getWifiConfig", timeout=2.5)
            raw_ssid = conf.get("connectedAP") or conf.get("ssid") or ""
            ssid = urllib.parse.unquote_plus(raw_ssid)
            is_connected = conf.get("isConnected") == "1"

            wlist = self.cgi_request("getWifiList", timeout=3.5)
            signal = 0
            bssid = ""
            for k, v in wlist.items():
                if k.startswith("ap") and v:
                    dec = urllib.parse.unquote_plus(v)
                    parts = dec.split("+")
                    if len(parts) >= 3:
                        ap_ssid = parts[0]
                        ap_bssid = parts[1]
                        try:
                            ap_sig = int(parts[2])
                        except Exception:
                            ap_sig = 0
                        if ap_ssid.strip().lower() == ssid.strip().lower():
                            signal = ap_sig
                            bssid = ap_bssid
                            break

            if signal >= 80:
                quality, bars = "Excelente", 4
            elif signal >= 60:
                quality, bars = "Bueno", 3
            elif signal >= 40:
                quality, bars = "Regular", 2
            elif signal > 0:
                quality, bars = "Débil", 1
            else:
                quality, bars = ("Sin señal" if not is_connected else "Conectado"), 0

            res = {
                "connected": is_connected,
                "ssid": ssid,
                "signal": signal,
                "quality": quality,
                "bars": bars,
                "bssid": bssid
            }
            self.cached_wifi = res
            return res
        except Exception as e:
            print(f"get_wifi_details error: {e}")
            return self.cached_wifi

    def ensure_max_volume(self):
        """Forces camera speaker volume to 100% (only writes when it differs)."""
        try:
            cur = self.cgi_request("getAudioSetting", timeout=2.5)
            if cur.get("volume") != "100":
                self.cgi_request("setAudioVolume", {"volume": 100}, timeout=2.5)
        except Exception as e:
            print(f"ensure_max_volume error: {e}")

    def ensure_optimal_video_encoding(self):
        """Forces camera hardware DSP to 30 FPS, ultra-low GOP (15/10), and CBR for zero-jitter streaming."""
        try:
            cur = self.cgi_request("getVideoStreamParam", timeout=2.5)
            if cur.get("frameRate0") != "30" or cur.get("GOP0") != "15" or cur.get("isVBR0") != "0":
                self.cgi_request("setVideoStreamParam", {
                    "streamType": 0,
                    "resolution": 0,
                    "bitRate": 2097152,
                    "frameRate": 30,
                    "GOP": 15,
                    "isVBR": 0
                }, timeout=2.5)
            if cur.get("frameRate1") != "30" or cur.get("GOP1") != "15" or cur.get("isVBR1") != "0":
                self.cgi_request("setVideoStreamParam", {
                    "streamType": 1,
                    "resolution": 1,
                    "bitRate": 2097152,
                    "frameRate": 30,
                    "GOP": 15,
                    "isVBR": 0
                }, timeout=2.5)
        except Exception as e:
            print(f"ensure_optimal_video_encoding error: {e}")

    def _state_refresher(self):
        """Refreshes camera state gently every 6 seconds in background without locking up UI."""
        consecutive_errors = 0
        time.sleep(1.0)
        self.ensure_max_volume()
        self.ensure_optimal_video_encoding()
        self.get_wifi_details()
        wifi_cycle = 0
        while True:
            try:
                time.sleep(6.0)
                wifi_cycle += 1
                if wifi_cycle >= 5:
                    wifi_cycle = 0
                    self.get_wifi_details()
                state = self.cgi_request("getDevState", timeout=2.5)
                if state.get("result") == "0" or "motionDetectAlarm" in state:
                    self.cached_state.update(state)
                    self.is_online = True
                    self.last_seen = time.time()
                    if consecutive_errors > 0:
                        # camera came back (e.g. after reboot): restore 100% volume
                        self.ensure_max_volume()
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1

                # If needed, refresh device info and ports once on startup
                if self.cached_info.get("firmwareVer") == "1.11.1.18" and consecutive_errors == 0:
                    info = self.cgi_request("getDevInfo", timeout=2.5)
                    if info.get("result") == "0":
                        self.cached_info.update(info)
                    ports = self.cgi_request("getPortInfo", timeout=2.5)
                    if ports.get("result") == "0":
                        self.cached_ports.update(ports)

                # Only if unreachable for over 30 seconds, trigger UDP discovery
                if consecutive_errors >= 5:
                    self.is_online = False
                    new_ip = self.discover_ip()
                    if new_ip:
                        self.ip = new_ip
                        consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    self.is_online = False

    def discover_ip(self) -> Optional[str]:
        msg = bytes([
            0x4d, 0x4f, 0x5f, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
        ])
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(1.5)
        found_ip = None
        try:
            for target in ["255.255.255.255", "192.168.1.255"]:
                try:
                    s.sendto(msg, (target, 10000))
                    s.sendto(msg, (target, 10001))
                except Exception:
                    pass

            start = time.time()
            while time.time() - start < 1.5:
                try:
                    data, addr = s.recvfrom(1024)
                    if b"FI9821W" in data or b"00626E" in data:
                        found_ip = addr[0]
                        self.ip = found_ip
                        self.is_online = True
                        break
                except socket.timeout:
                    break
        except Exception as e:
            print(f"Discovery error: {e}")
        finally:
            s.close()
        return found_ip

    def cgi_request(self, cmd: str, params: Optional[Dict[str, Any]] = None, timeout: float = 3.0) -> Dict[str, str]:
        with self.cgi_lock:
            p = {"cmd": cmd, "usr": self.usr, "pwd": self.pwd}
            if params:
                p.update(params)
            
            query = urllib.parse.urlencode(p)
            url = f"http://{self.ip}:{self.port}/cgi-bin/CGIProxy.fcgi?{query}"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "FoscamControlCenter"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw_xml = resp.read().decode("utf-8", errors="ignore")
                    self.is_online = True
                    self.last_seen = time.time()
                    return self._parse_xml(raw_xml)
            except Exception as e:
                # Do NOT block with UDP discover here! Simply log and return error
                print(f"CGI request error ({cmd}): {e}")
                return {"result": "-1", "error": str(e)}

    def get_snapshot_bytes(self, timeout: float = 3.0) -> bytes:
        # 1. Ultra-fast instant snapshot directly from go2rtc RAM buffer (sub-5ms)
        if is_go2rtc_available():
            try:
                url_go2rtc = "http://127.0.0.1:1984/api/frame.jpeg?src=foscam"
                req_g = urllib.request.Request(url_go2rtc, headers={"User-Agent": "FoscamControlCenter"})
                with urllib.request.urlopen(req_g, timeout=0.8) as resp:
                    data = resp.read()
                    if data and len(data) > 1000:
                        self.is_online = True
                        return data
            except Exception:
                pass

        # 2. Camera hardware CGI fallback
        with self.cgi_lock:
            url = f"http://{self.ip}:{self.port}/cgi-bin/CGIProxy.fcgi?cmd=snapPicture2&usr={self.usr}&pwd={self.pwd}"
            req = urllib.request.Request(url, headers={"User-Agent": "FoscamControlCenter"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                self.is_online = True
                return resp.read()

    def _parse_xml(self, xml_text: str) -> Dict[str, str]:
        res = {}
        try:
            root = ET.fromstring(xml_text)
            for child in root:
                res[child.tag] = child.text or ""
        except Exception:
            matches = re.findall(r'<([a-zA-Z0-9_]+)>([^<]*)</\1>', xml_text)
            for tag, val in matches:
                res[tag] = val
        return res

    def _raw_cgi_call(self, cmd: str, params: Optional[Dict[str, Any]] = None, timeout: float = 1.5) -> Dict[str, str]:
        """Direct CGI call assuming cgi_lock is already held if atomic sequence."""
        p = {"cmd": cmd, "usr": self.usr, "pwd": self.pwd}
        if params:
            p.update(params)
        query = urllib.parse.urlencode(p)
        url = f"http://{self.ip}:{self.port}/cgi-bin/CGIProxy.fcgi?{query}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "FoscamControlCenter"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_xml = resp.read().decode("utf-8", errors="ignore")
                return self._parse_xml(raw_xml)
        except Exception as e:
            return {"result": "-1", "error": str(e)}

    def abort_ptz(self):
        """Immediately aborts any ongoing PTZ pulse and signals stop to camera hardware."""
        self.ptz_abort_event.set()
        try:
            self._raw_cgi_call("ptzStopRun", timeout=1.0)
        except Exception:
            pass

    def ptz_pulse_precise(self, cmd: str, duration_ms: int, target_speed: int = 0):
        """
        High-precision pipelined PTZ pulse using a single persistent socket.
        Sends start command, sleeps exactly duration_ms (or until aborted),
        and sends ptzStopRun down the exact same TCP stream with sub-millisecond precision.
        Eliminates 400ms+ HTTP response lag and Wi-Fi latency jitter.
        """
        dur_ms = max(30, min(4000, duration_ms))
        dur_sec = dur_ms / 1000.0

        with self.cgi_lock:
            self.ptz_abort_event.clear()
            # Ensure camera internal speed matches calibration
            if self.ptz_speed != target_speed:
                try:
                    self._raw_cgi_call("setPTZSpeed", {"speed": target_speed}, timeout=1.0)
                    self.ptz_speed = target_speed
                except Exception:
                    pass

            s = None
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.5)
                s.connect((self.ip, self.port))
                
                # 1. Send PTZ move command (starts motor immediately)
                req1 = f"GET /cgi-bin/CGIProxy.fcgi?cmd={cmd}&usr={self.usr}&pwd={self.pwd} HTTP/1.1\r\nHost: {self.ip}:{self.port}\r\n\r\n"
                s.sendall(req1.encode())

                # 2. Wait EXACT pulse duration (or exit instantly if aborted)
                self.ptz_abort_event.wait(timeout=dur_sec)

                # 3. Send stop command down same pipe without new TCP handshake
                req2 = f"GET /cgi-bin/CGIProxy.fcgi?cmd=ptzStopRun&usr={self.usr}&pwd={self.pwd} HTTP/1.1\r\nHost: {self.ip}:{self.port}\r\nConnection: close\r\n\r\n"
                s.sendall(req2.encode())

                try:
                    s.recv(512)
                except Exception:
                    pass
            except Exception as e:
                print(f"ptz_pulse_precise error: {e}")
                try:
                    self._raw_cgi_call("ptzStopRun", timeout=1.0)
                except Exception:
                    pass
            finally:
                if s:
                    try:
                        s.close()
                    except Exception:
                        pass

    def ptz_step(self, direction: str, duration_ms: int = 80, is_tracking: bool = False):
        direction_map = {
            "up": ("ptzMoveUp", 0, 1),
            "down": ("ptzMoveDown", 0, -1),
            "left": ("ptzMoveLeft", -1, 0),
            "right": ("ptzMoveRight", 1, 0),
            "upleft": ("ptzMoveTopLeft", -1, 1),
            "upright": ("ptzMoveTopRight", 1, 1),
            "downleft": ("ptzMoveBottomLeft", -1, -1),
            "downright": ("ptzMoveBottomRight", 1, -1),
        }
        item = direction_map.get(direction.lower())
        if not item:
            return
        cmd, pan_sign, tilt_sign = item

        dur_ms = max(35, min(600, duration_ms))
        target_spd = 0 if is_tracking else min(1, self.ptz_speed)

        def _worker():
            with self.ptz_motion_lock:
                self.ptz_pulse_precise(cmd, dur_ms, target_speed=target_spd)

                # Calibrated angular velocity (Speed 0: 0.018°/ms Pan, 0.014°/ms Tilt)
                pan_rate = 0.018 if target_spd == 0 else 0.036
                tilt_rate = 0.014 if target_spd == 0 else 0.028

                delta_pan = pan_sign * dur_ms * pan_rate
                delta_tilt = tilt_sign * dur_ms * tilt_rate

                self.pan_angle = max(-150.0, min(150.0, round(self.pan_angle + delta_pan, 1)))
                self.tilt_angle = max(-20.0, min(90.0, round(self.tilt_angle + delta_tilt, 1)))

        threading.Thread(target=_worker, daemon=True).start()

    def ptz_goto_degrees(self, target_pan: Optional[float] = None, target_tilt: Optional[float] = None, delta_pan: Optional[float] = None, delta_tilt: Optional[float] = None):
        """Moves camera accurately to target angles or by delta degrees using pipelined micro-stepping."""
        d_pan = delta_pan if delta_pan is not None else ((target_pan - self.pan_angle) if target_pan is not None else 0.0)
        d_tilt = delta_tilt if delta_tilt is not None else ((target_tilt - self.tilt_angle) if target_tilt is not None else 0.0)

        def _exec():
            with self.ptz_motion_lock:
                # Horizontal (Pan)
                if abs(d_pan) >= 0.5:
                    spd_pan = 0 if abs(d_pan) <= 20.0 else 1
                    pan_rate = 0.018 if spd_pan == 0 else 0.036
                    cmd = "ptzMoveRight" if d_pan > 0 else "ptzMoveLeft"
                    dur_ms = max(35, min(3500, int(abs(d_pan) / pan_rate)))
                    self.ptz_pulse_precise(cmd, dur_ms, target_speed=spd_pan)
                    self.pan_angle = max(-150.0, min(150.0, round(self.pan_angle + d_pan, 1)))
                    time.sleep(0.08)

                # Vertical (Tilt)
                if abs(d_tilt) >= 0.5:
                    spd_tilt = 0 if abs(d_tilt) <= 15.0 else 1
                    tilt_rate = 0.014 if spd_tilt == 0 else 0.028
                    cmd = "ptzMoveUp" if d_tilt > 0 else "ptzMoveDown"
                    dur_ms = max(35, min(2500, int(abs(d_tilt) / tilt_rate)))
                    self.ptz_pulse_precise(cmd, dur_ms, target_speed=spd_tilt)
                    self.tilt_angle = max(-20.0, min(90.0, round(self.tilt_angle + d_tilt, 1)))

        threading.Thread(target=_exec, daemon=True).start()

    def ptz_reset(self):
        res = self.cgi_request("ptzReset")
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        return res

    def get_immediate_frame(self) -> Optional[bytes]:
        try:
            return self.get_snapshot_bytes(timeout=2.0)
        except Exception:
            return None

MIC_GAIN = 5.0  # software boost applied to the browser microphone

camera = FoscamCamera()
talk_session = FoscamTalkSession(ip=camera.ip, port=camera.port, user=camera.usr, pwd=camera.pwd)

# Ultra-fast Low-Latency RTSP Stream Worker via FFmpeg Hardware Stream
class RTSPStreamWorker:
    def __init__(self, camera_obj: FoscamCamera):
        self.camera = camera_obj
        self.mode = "sub" # "sub" (fast 450p) or "main" (720p HD)
        self.latest_frame = None
        self.last_update = 0
        self.running = True
        self.lock = threading.Lock()
        self.proc = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def set_mode(self, mode: str):
        if mode in ["sub", "main"] and mode != self.mode:
            self.mode = mode
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except Exception:
                    pass

    def get_latest_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_frame

    def restart(self):
        """Immediately terminates current FFmpeg process and clears buffer to force fresh RTSP handshake."""
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        with self.lock:
            self.latest_frame = None
            self.last_update = 0

    def _run(self):
        while self.running:
            # Connect preferentially to local go2rtc proxy stream (protects camera from multiple RTSP connections)
            if is_go2rtc_available():
                stream_name = "foscam" if self.mode == "main" else "foscam_sub"
                rtsp_url = f"rtsp://127.0.0.1:8554/{stream_name}?video"
            else:
                rtsp_url = f"rtsp://{self.camera.usr}:{self.camera.pwd}@{self.camera.ip}:{self.camera.port}/videoMain"

            scale_filter = "scale=800:-1" if self.mode == "sub" else "scale=1280:-1"
            
            # Ultra low-latency flags:
            # -fflags nobuffer: eliminates internal FFmpeg buffering
            # -flags low_delay: forces lowest decoding latency
            # -avioflags direct: bypasses I/O buffering
            # -probesize 32768 -analyzeduration 0: starts stream instantly without delay
            cmd = [
                'ffmpeg',
                '-loglevel', 'warning',
                '-rtsp_transport', 'tcp',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-strict', 'experimental',
                '-avioflags', 'direct',
                '-fflags', '+discardcorrupt',
                '-probesize', '32768',
                '-analyzeduration', '0',
                '-i', rtsp_url,
                '-vf', scale_filter,
                '-f', 'mjpeg',
                '-q:v', '4',
                '-r', '30',
                '-flush_packets', '1',
                '-an',
                '-'
            ]
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0
                )
                buf = bytearray()
                while self.running and self.proc.poll() is None:
                    chunk = self.proc.stdout.read(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    
                    # Parse all complete frames in buffer, saving only the newest
                    newest_frame = None
                    while True:
                        soi = buf.find(b'\xff\xd8')
                        if soi == -1:
                            buf.clear()
                            break
                        eoi = buf.find(b'\xff\xd9', soi + 2)
                        if eoi == -1:
                            if soi > 0:
                                del buf[:soi]
                            break
                        newest_frame = bytes(buf[soi:eoi+2])
                        del buf[:eoi+2]

                    if newest_frame:
                        with self.lock:
                            self.latest_frame = newest_frame
                            self.last_update = time.time()

                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
            except Exception as e:
                print(f"StreamWorker error: {e}")
            time.sleep(1.0)

rtsp_worker = RTSPStreamWorker(camera)
ai_tracker.set_camera_and_worker(camera, rtsp_worker)

# Request Models
class AIConfigRequest(BaseModel):
    ai_enabled: Optional[bool] = None
    tracking_enabled: Optional[bool] = None
    target_classes: Optional[list] = None
    min_confidence: Optional[float] = None
    return_home_enabled: Optional[bool] = None
    home_preset: Optional[str] = None
    deadzone_x: Optional[float] = None
    deadzone_y: Optional[float] = None
    model_name: Optional[str] = None

class SwitchModelRequest(BaseModel):
    model_name: str

class TelegramConfigRequest(BaseModel):
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    enabled: Optional[bool] = None
    cooldown_seconds: Optional[int] = None
    include_photo: Optional[bool] = None
    app_url: Optional[str] = None

class PTZMoveRequest(BaseModel):
    direction: str
    mode: Optional[str] = "step"       # "step" (clean jog mode) or "continuous"
    duration: Optional[int] = 250      # milliseconds for step mode (e.g. 150, 250, 500)

class PTZGotoDegreesRequest(BaseModel):
    pan: Optional[float] = None
    tilt: Optional[float] = None
    delta_pan: Optional[float] = None
    delta_tilt: Optional[float] = None

class PTZSpeedRequest(BaseModel):
    speed: int

class PTZPresetRequest(BaseModel):
    name: str

class PTZCruiseRequest(BaseModel):
    cruise_type: str

class ImageSettingsRequest(BaseModel):
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None
    hue: Optional[int] = None
    sharpness: Optional[int] = None

class InfraLedRequest(BaseModel):
    mode: Optional[str] = None
    state: Optional[str] = None

class MotionDetectRequest(BaseModel):
    isEnable: int
    sensitivity: int
    triggerInterval: int

class WifiConnectRequest(BaseModel):
    ssid: str
    psk: str

class StreamModeRequest(BaseModel):
    mode: str  # "sub" or "main"

class PlaySoundRequest(BaseModel):
    sound: str  # "chime", "siren", "alert", "dog"

class RecordingConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    stream_mode: Optional[str] = None
    segment_duration: Optional[int] = None
    delete_after_upload: Optional[bool] = None

class GDriveConfigRequest(BaseModel):
    folder_name: Optional[str] = None
    auto_upload: Optional[bool] = None
    delete_after_upload: Optional[bool] = None

class GDrivePasteRequest(BaseModel):
    credentials_json: str
    filename: Optional[str] = "service_account.json"

# Endpoints
@app.get("/")
def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Foscam Control Center running."}

@app.get("/manifest.json")
def get_manifest():
    manifest_file = STATIC_DIR / "manifest.json"
    if manifest_file.exists():
        return FileResponse(manifest_file, media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="Manifest not found")

@app.get("/sw.js")
def get_sw():
    sw_file = STATIC_DIR / "sw.js"
    if sw_file.exists():
        return FileResponse(sw_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")

@app.get("/api/status")
def get_status():
    """Returns cached device info, state and Wi-Fi signal instantly in <1ms without burdening camera."""
    return {
        "online": camera.is_online,
        "ip": camera.ip,
        "port": camera.port,
        "model": camera.cached_info.get("productName", "FI9821W V2"),
        "devName": camera.cached_info.get("devName", "FI9821W"),
        "mac": camera.cached_info.get("mac", camera.mac),
        "firmware": camera.cached_info.get("firmwareVer", "1.11.1.18"),
        "hardware": camera.cached_info.get("hardwareVer", "1.4.1.8"),
        "streamMode": rtsp_worker.mode,
        "ports": camera.cached_ports,
        "state": camera.cached_state,
        "wifi": camera.cached_wifi
    }

@app.get("/api/wifi/signal")
def get_wifi_signal():
    """Queries camera for latest real-time Wi-Fi signal strength and details."""
    return camera.get_wifi_details()

@app.post("/api/camera/reconnect")
def reconnect_camera():
    """
    Safely restarts video streaming pipeline, tests camera connection,
    refreshes Wi-Fi signal, and ensures maximum audio volume.
    """
    is_alive = False
    try:
        state = camera.cgi_request("getDevState", timeout=2.0)
        if state.get("result") == "0" or "motionDetectAlarm" in state:
            is_alive = True
    except Exception:
        pass

    if not is_alive:
        new_ip = camera.discover_ip()
        if new_ip:
            camera.ip = new_ip
            is_alive = True

    # 1. Restart RTSP Stream Worker
    rtsp_worker.restart()

    # 2. Reset talk session if any
    try:
        talk_session.stop(drain_time=0.1)
    except Exception:
        pass

    # 3. Ensure volume, encoding profile, and get updated Wi-Fi details
    camera.ensure_max_volume()
    camera.ensure_optimal_video_encoding()
    wifi = camera.get_wifi_details()

    return {
        "status": "reconnected" if is_alive else "unreachable",
        "online": is_alive,
        "ip": camera.ip,
        "wifi": wifi
    }

@app.post("/api/stream/mode")
def set_stream_mode(req: StreamModeRequest):
    rtsp_worker.set_mode(req.mode)
    return {"mode": rtsp_worker.mode}

@app.get("/api/video/encoding")
def get_video_encoding():
    """Returns the current hardware DSP encoding profiles from the camera."""
    return camera.cgi_request("getVideoStreamParam")

@app.post("/api/video/encoding/optimize")
def optimize_video_encoding():
    """Forces 30 FPS, low GOP (15/10), and CBR for ultra-low latency."""
    camera.ensure_optimal_video_encoding()
    return {"status": "optimized", "params": camera.cgi_request("getVideoStreamParam")}

async def generate_mjpeg_stream(request: Request):
    """Ultra-responsive real-time stream generator backed by FFmpeg RTSP worker."""
    last_sent = 0
    try:
        while True:
            if await request.is_disconnected():
                break
            frame = rtsp_worker.latest_frame
            curr_time = rtsp_worker.last_update
            
            if frame and curr_time > last_sent:
                last_sent = curr_time
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame + b"\r\n"
                )
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.01)
    except (asyncio.CancelledError, GeneratorExit):
        pass

@app.get("/api/stream")
def get_stream(request: Request):
    return StreamingResponse(
        generate_mjpeg_stream(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0, no-transform",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "close"
        }
    )

@app.get("/api/go2rtc/status")
def get_go2rtc_status():
    available = is_go2rtc_available()
    streams_info = {}
    if available:
        try:
            req = urllib.request.Request("http://127.0.0.1:1984/api/streams", headers={"User-Agent": "FoscamControlCenter"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                streams_info = json.loads(resp.read().decode())
        except Exception:
            pass
    return {
        "running": available,
        "rtsp_local": "rtsp://127.0.0.1:8554/foscam",
        "api_local": "http://127.0.0.1:1984",
        "streams": list(streams_info.keys()) if streams_info else []
    }

@app.post("/api/webrtc")
async def post_webrtc_offer(request: Request):
    """Proxies WebRTC offer/answer negotiation directly to local go2rtc engine."""
    body = await request.body()
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:1984/api/webrtc?src=foscam",
            data=body,
            headers={"Content-Type": request.headers.get("Content-Type", "application/sdp")},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            answer = resp.read()
            return Response(content=answer, media_type=resp.headers.get_content_type() or "application/sdp")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"go2rtc WebRTC gateway error: {e}")

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    High-performance real-time video stream over WebSocket.
    Zero buffering through Cloudflare Tunnel, sub-50ms latency, 100% reliable on mobile!
    """
    await websocket.accept()
    last_sent = 0
    try:
        init_frame = rtsp_worker.latest_frame
        if init_frame:
            await websocket.send_bytes(init_frame)
            last_sent = rtsp_worker.last_update

        while True:
            frame = rtsp_worker.latest_frame
            curr_time = rtsp_worker.last_update
            if frame and curr_time > last_sent:
                last_sent = curr_time
                await websocket.send_bytes(frame)
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass

def generate_audio_stream():
    """Streams live microphone audio from the camera transcoded to MP3."""
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', f'rtsp://{camera.usr}:{camera.pwd}@{camera.ip}:{camera.port}/audio',
        '-f', 'mp3',
        '-acodec', 'libmp3lame',
        '-ab', '64k',
        '-ar', '16000',
        '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**5)
    try:
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                break
            yield chunk
    finally:
        if proc.poll() is None:
            proc.terminate()

@app.get("/api/audio/stream")
def get_audio_stream():
    return StreamingResponse(generate_audio_stream(), media_type="audio/mpeg")

@app.websocket("/ws/talk")
async def websocket_talk(websocket: WebSocket):
    """Real-time push-to-talk bidirectional audio channel between browser mic and camera speaker."""
    await websocket.accept()
    talk_session.ip = camera.ip
    talk_session.port = camera.port
    talk_session.user = camera.usr
    talk_session.pwd = camera.pwd

    await asyncio.to_thread(camera.ensure_max_volume)
    ok = await asyncio.to_thread(talk_session.start)
    if not ok:
        await websocket.send_json({"error": "Failed to open talk session on camera"})
        await websocket.close()
        return

    await websocket.send_json({"status": "connected", "message": "Camera speaker ready"})
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                talk_session.send_pcm_bytes(message["bytes"], gain=MIC_GAIN)
            elif "text" in message:
                if message["text"] == "stop":
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Talk WebSocket error: {e}")
    finally:
        # Gracefully drain queue and schedule socket teardown so camera speaker finishes playing
        talk_session.stop_async(drain_time=1.8)

_audio_playback_lock = threading.Lock()

def _play_pcm_task(pcm_bytes: bytes, drain_time: float = 1.8):
    """Safely serializes sound effects and file playback to prevent thread collisions."""
    with _audio_playback_lock:
        talk_session.ip = camera.ip
        talk_session.port = camera.port
        talk_session.user = camera.usr
        talk_session.pwd = camera.pwd
        camera.ensure_max_volume()
        if talk_session.start():
            talk_session.send_pcm_bytes(pcm_bytes)
            talk_session.stop(drain_time=drain_time)

@app.post("/api/audio/play_sound")
def play_sound(req: PlaySoundRequest):
    """Plays built-in sound effect (chime, siren, alert, dog) through camera speaker."""
    pcm = generate_sound_effect(req.sound)
    if not pcm:
        raise HTTPException(status_code=400, detail="Invalid sound")
    threading.Thread(target=_play_pcm_task, args=(pcm, 1.8), daemon=True).start()
    return {"status": "playing", "sound": req.sound}

@app.post("/api/audio/upload_and_play")
async def upload_and_play(file: UploadFile = File(...)):
    """Upload any audio file or recorded voice memo to play through camera speaker."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")
    pcm = transcode_audio_to_pcm(content)
    if not pcm:
        raise HTTPException(status_code=400, detail="Could not transcode audio file")

    threading.Thread(target=_play_pcm_task, args=(pcm, 1.8), daemon=True).start()
    return {"status": "playing", "size_pcm": len(pcm)}

@app.post("/api/audio/talk_stop")
def stop_talk():
    talk_session.stop(drain_time=0.2)
    return {"status": "stopped"}

@app.get("/api/audio/talk_status")
def talk_status():
    return {"active": talk_session.is_connected()}

@app.get("/api/snapshot")
def get_snapshot():
    if rtsp_worker.latest_frame:
        return Response(content=rtsp_worker.latest_frame, media_type="image/jpeg")
    try:
        data = camera.get_snapshot_bytes(3.0)
        return Response(content=data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/snapshot/save")
def save_snapshot():
    try:
        data = camera.get_snapshot_bytes(3.0)
        filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = SNAPSHOTS_DIR / filename
        with open(filepath, "wb") as f:
            f.write(data)
        return {"success": True, "filename": filename, "url": f"/snapshots/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gallery")
def list_snapshots():
    files = []
    for p in sorted(SNAPSHOTS_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True):
        files.append({
            "filename": p.name,
            "url": f"/snapshots/{p.name}",
            "size": p.stat().st_size,
            "time": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        })
    return files

@app.delete("/api/gallery/{filename}")
def delete_snapshot(filename: str):
    filepath = SNAPSHOTS_DIR / filename
    if filepath.exists() and filepath.is_file():
        filepath.unlink()
        return {"success": True}
    raise HTTPException(status_code=404, detail="File not found")

# ========================================================
# RESPONSIVE, LOW-LAG PTZ CONTROLS (STEP & CONTINUOUS)
# ========================================================
_ptz_step_thread_lock = threading.Lock()

@app.post("/api/ptz/move")
async def ptz_move(req: PTZMoveRequest):
    direction_map = {
        "up": "ptzMoveUp",
        "down": "ptzMoveDown",
        "left": "ptzMoveLeft",
        "right": "ptzMoveRight",
        "upleft": "ptzMoveTopLeft",
        "upright": "ptzMoveTopRight",
        "downleft": "ptzMoveBottomLeft",
        "downright": "ptzMoveBottomRight",
        "stop": "ptzStopRun",
        "center": "ptzReset"
    }
    cmd = direction_map.get(req.direction.lower())
    if not cmd:
        raise HTTPException(status_code=400, detail="Invalid direction")

    if req.direction.lower() == "center":
        camera.ptz_reset()
        return {"command": "ptzReset", "pan": 0.0, "tilt": 0.0}

    if req.direction.lower() == "stop":
        camera.abort_ptz()
        return {"command": "ptzStopRun", "status": "aborted"}

    if req.mode == "continuous":
        res = await asyncio.to_thread(camera.cgi_request, cmd)
        return {"command": cmd, "result": res, "mode": "continuous"}
    else:
        # Atomic step mode with real degree tracking
        dur_ms = req.duration or 200
        camera.ptz_step(req.direction, duration_ms=dur_ms)
        return {
            "command": cmd,
            "mode": "step",
            "duration_ms": dur_ms,
            "pan": camera.pan_angle,
            "tilt": camera.tilt_angle
        }

@app.post("/api/ptz/stop")
async def ptz_stop():
    camera.abort_ptz()
    return {"command": "ptzStopRun", "status": "aborted"}

@app.get("/api/ptz/coordinates")
def get_ptz_coordinates():
    """Devuelve las coordenadas espaciales estimadas de la cámara en 3D (Pan y Tilt en grados)."""
    return {
        "pan": camera.pan_angle,
        "tilt": camera.tilt_angle,
        "speed": camera.ptz_speed
    }

@app.post("/api/ptz/goto_degrees")
def ptz_goto_degrees(req: PTZGotoDegreesRequest):
    """Mueve la cámara con micro-pasos calibrados a un ángulo exacto o diferencial en grados."""
    camera.ptz_goto_degrees(req.pan, req.tilt, req.delta_pan, req.delta_tilt)
    return {
        "pan": camera.pan_angle,
        "tilt": camera.tilt_angle,
        "speed": camera.ptz_speed
    }

@app.post("/api/ptz/calibrate")
def ptz_calibrate():
    """Recalibra el centro de coordenadas en 0.0, 0.0."""
    camera.pan_angle = 0.0
    camera.tilt_angle = 0.0
    return {"status": "calibrated", "pan": 0.0, "tilt": 0.0}

@app.post("/api/ptz/speed")
def ptz_speed(req: PTZSpeedRequest):
    speed = max(0, min(4, req.speed))
    camera.ptz_speed = speed
    res = camera.cgi_request("setPTZSpeed", {"speed": speed})
    return {"speed": speed, "result": res}

@app.get("/api/ptz/presets")
def get_presets():
    res = camera.cgi_request("getPTZPresetPointList")
    points = []
    for i in range(16):
        pt = res.get(f"point{i}")
        if pt and pt.strip():
            points.append(pt.strip())
    return {"count": len(points), "presets": points}

@app.post("/api/ptz/preset/goto")
def goto_preset(req: PTZPresetRequest):
    res = camera.cgi_request("ptzGotoPresetPoint", {"name": req.name})
    return {"preset": req.name, "result": res}

@app.post("/api/ptz/preset/add")
def add_preset(req: PTZPresetRequest):
    res = camera.cgi_request("ptzAddPresetPoint", {"name": req.name})
    return {"preset": req.name, "result": res}

@app.post("/api/ptz/preset/delete")
def delete_preset(req: PTZPresetRequest):
    res = camera.cgi_request("ptzDeletePresetPoint", {"name": req.name})
    return {"preset": req.name, "result": res}

@app.post("/api/ptz/cruise")
def ptz_cruise(req: PTZCruiseRequest):
    if req.cruise_type == "horizontal":
        res = camera.cgi_request("ptzStartCruise", {"mapName": "Horizontal"})
    elif req.cruise_type == "vertical":
        res = camera.cgi_request("ptzStartCruise", {"mapName": "Vertical"})
    else:
        res = camera.cgi_request("ptzStopCruise")
    return {"cruise": req.cruise_type, "result": res}

@app.get("/api/image")
def get_image_settings():
    res = camera.cgi_request("getImageSetting")
    return {
        "brightness": int(res.get("brightness", 50)),
        "contrast": int(res.get("contrast", 50)),
        "saturation": int(res.get("saturation", 50)),
        "hue": int(res.get("hue", 50)),
        "sharpness": int(res.get("sharpness", 50)),
        "denoiseLevel": int(res.get("denoiseLevel", 50))
    }

@app.post("/api/image")
def set_image_settings(req: ImageSettingsRequest):
    results = {}
    if req.brightness is not None:
        results["brightness"] = camera.cgi_request("setBrightness", {"brightness": req.brightness})
    if req.contrast is not None:
        results["contrast"] = camera.cgi_request("setContrast", {"contrast": req.contrast, "constrast": req.contrast})
    if req.saturation is not None:
        results["saturation"] = camera.cgi_request("setSaturation", {"saturation": req.saturation})
    if req.hue is not None:
        results["hue"] = camera.cgi_request("setHue", {"hue": req.hue})
    if req.sharpness is not None:
        results["sharpness"] = camera.cgi_request("setSharpness", {"sharpness": req.sharpness})
    return {"success": True, "results": results}

@app.post("/api/image/reset")
def reset_image():
    res = camera.cgi_request("resetImageSetting")
    return {"result": res}

@app.post("/api/image/flip")
def flip_image():
    res = camera.cgi_request("flipVideo")
    return {"result": res}

@app.post("/api/image/mirror")
def mirror_image():
    res = camera.cgi_request("mirrorVideo")
    return {"result": res}

@app.get("/api/led")
def get_led_config():
    cfg = camera.cgi_request("getInfraLedConfig")
    mode = "auto" if cfg.get("mode") == "0" else "manual"
    return {"mode": mode, "raw": cfg}

@app.post("/api/led")
def set_led_config(req: InfraLedRequest):
    res = {}
    if req.mode == "auto":
        res["mode"] = camera.cgi_request("setInfraLedConfig", {"mode": "0"})
    elif req.mode == "manual":
        res["mode"] = camera.cgi_request("setInfraLedConfig", {"mode": "1"})

    if req.state:
        # Foscam firmware requires manual mode (mode: 1) before accepting open/closeInfraLed
        camera.cgi_request("setInfraLedConfig", {"mode": "1"})
        time.sleep(0.08)
        if req.state == "on":
            res["state"] = camera.cgi_request("openInfraLed")
        else:
            res["state"] = camera.cgi_request("closeInfraLed")
    return res

@app.get("/api/motion")
def get_motion():
    res = camera.cgi_request("getMotionDetectConfig")
    return {
        "isEnable": int(res.get("isEnable", 0)),
        "sensitivity": int(res.get("sensitivity", 1)),
        "triggerInterval": int(res.get("triggerInterval", 5)),
        "linkage": int(res.get("linkage", 0)),
        "raw": res
    }

@app.post("/api/motion")
def set_motion(req: MotionDetectRequest):
    params = {
        "isEnable": str(req.isEnable),
        "sensitivity": str(req.sensitivity),
        "triggerInterval": str(req.triggerInterval),
        "linkage": "0",
        "snapInterval": "2"
    }
    res = camera.cgi_request("setMotionDetectConfig", params)
    return {"result": res}

@app.get("/api/wifi/config")
def get_wifi_config():
    res = camera.cgi_request("getWifiConfig")
    ssid = res.get("ssid", "").replace("+", " ")
    return {
        "isEnable": res.get("isEnable"),
        "isUseWifi": res.get("isUseWifi"),
        "isConnected": res.get("isConnected"),
        "ssid": ssid,
        "encryptType": res.get("encryptType"),
        "raw": res
    }

@app.get("/api/wifi/scan")
def scan_wifi():
    camera.cgi_request("refreshWifiList", timeout=12.0)
    time.sleep(1.0)
    res = camera.cgi_request("getWifiList", timeout=8.0)
    networks = []
    cur = int(res.get("curCnt", 0))
    for i in range(cur):
        raw_ap = res.get(f"ap{i}")
        if raw_ap:
            decoded = urllib.parse.unquote(raw_ap)
            parts = decoded.split("+")
            if len(parts) >= 3:
                ssid = parts[0].replace("+", " ").strip()
                mac = parts[1] if len(parts) > 1 else ""
                signal = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                is_encrypt = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
                networks.append({
                    "ssid": ssid,
                    "mac": mac,
                    "signal": signal,
                    "encrypted": is_encrypt > 0
                })
    networks.sort(key=lambda x: x["signal"], reverse=True)
    return {"count": len(networks), "networks": networks}

@app.post("/api/wifi/connect")
def connect_wifi(req: WifiConnectRequest):
    params = {
        "isEnable": "1",
        "isUseWifi": "1",
        "ssid": req.ssid,
        "psk": req.psk,
        "encryptType": "3",
        "authMode": "2",
        "netType": "0"
    }
    res = camera.cgi_request("setWifiSetting", params)
    return {"result": res}

@app.post("/api/system/sync_time")
def sync_time():
    now = datetime.now()
    params = {
        "timeSource": "1",
        "year": str(now.year),
        "mon": str(now.month),
        "day": str(now.day),
        "hour": str(now.hour),
        "min": str(now.minute),
        "sec": str(now.second),
        "timeZone": "-10800",
        "isDst": "0"
    }
    res = camera.cgi_request("setSystemTime", params)
    return {"result": res, "syncedTo": now.isoformat()}

@app.post("/api/system/reboot")
def reboot_camera():
    res = camera.cgi_request("rebootSystem")
    return {"result": res}

# ========================================================
# CONTINUOUS RECORDING & GOOGLE DRIVE CLOUD ENDPOINTS
# ========================================================

@app.get("/api/recording/status")
def get_recording_status():
    """Estado en tiempo real de la grabación continua y cuota de Google Drive."""
    return recorder_manager.get_status()

@app.post("/api/recording/toggle")
def toggle_recording():
    """Inicia o detiene la grabación continua segmentada."""
    if recorder_manager.is_recording:
        recorder_manager.stop_recording()
        return {"is_recording": False, "message": "Grabación continua pausada"}
    else:
        success = recorder_manager.start_recording()
        return {"is_recording": success, "message": "Grabación continua iniciada" if success else "Error al iniciar"}

@app.post("/api/recording/config")
def update_recording_config(req: RecordingConfigRequest):
    """Actualiza configuración de grabación (calidad, duración de segmentos, borrado tras subida)."""
    if req.stream_mode:
        recorder_manager.stream_mode = req.stream_mode
    if req.segment_duration:
        recorder_manager.segment_duration = req.segment_duration
    if req.delete_after_upload is not None:
        recorder_manager.delete_after_upload = req.delete_after_upload
    if req.enabled is not None:
        if req.enabled and not recorder_manager.is_recording:
            recorder_manager.start_recording()
        elif not req.enabled and recorder_manager.is_recording:
            recorder_manager.stop_recording()
    recorder_manager.save_config()
    return recorder_manager.get_status()

@app.get("/api/recording/list")
def get_recordings_list():
    """Lista las grabaciones locales y su estado de subida a la nube."""
    return recorder_manager.list_recordings()

@app.delete("/api/recording/{filename}")
def delete_recording(filename: str):
    """Elimina una grabación local."""
    recorder_manager.delete_recording(filename)
    return {"status": "deleted", "filename": filename}

@app.get("/api/gdrive/status")
def get_gdrive_status():
    """Consulta la conexión activa y cuota (5 TB) en Google Drive."""
    return gdrive_manager.test_connection()

@app.post("/api/gdrive/config")
def update_gdrive_config(req: GDriveConfigRequest):
    """Actualiza opciones de Google Drive (nombre de carpeta, subida automática)."""
    if req.folder_name:
        gdrive_manager.folder_name = req.folder_name
    if req.auto_upload is not None:
        gdrive_manager.auto_upload = req.auto_upload
    if req.delete_after_upload is not None:
        gdrive_manager.delete_after_upload = req.delete_after_upload
        recorder_manager.delete_after_upload = req.delete_after_upload
        recorder_manager.save_config()
    gdrive_manager.save_config()
    return gdrive_manager.test_connection()

@app.post("/api/gdrive/upload_credentials")
async def upload_gdrive_credentials(file: UploadFile = File(...)):
    """Permite subir el archivo service_account.json o credentials.json directamente desde la app."""
    content = await file.read()
    text = content.decode("utf-8")
    filename = file.filename or "service_account.json"
    if not filename.endswith(".json"):
        filename = "service_account.json"
    res = gdrive_manager.save_credentials_data(text, filename)
    test = gdrive_manager.test_connection()
    return {"status": "saved", "type": res.get("type"), "test": test}

@app.post("/api/gdrive/paste_credentials")
def paste_gdrive_credentials(req: GDrivePasteRequest):
    """Permite pegar directamente el contenido JSON de la cuenta de Google Drive."""
    res = gdrive_manager.save_credentials_data(req.credentials_json, req.filename or "service_account.json")
    test = gdrive_manager.test_connection()
    return {"status": "saved", "type": res.get("type"), "test": test}

@app.post("/api/gdrive/sync_now")
def sync_gdrive_now():
    """Fuerza la subida inmediata de todas las grabaciones locales a Google Drive."""
    queued = 0
    for f in RECORDINGS_DIR.glob("rec_*.mp4"):
        fname = f.name
        if fname not in recorder_manager.upload_queue:
            rec_info = recorder_manager.db.get(fname, {})
            if rec_info.get("status") != "uploaded":
                recorder_manager.upload_queue.append(fname)
                queued += 1
    return {"queued": queued, "total_queue": len(recorder_manager.upload_queue)}

@app.get("/api/gdrive/files")
def get_gdrive_files():
    """Lista los videos respaldados en la carpeta de Google Drive."""
    return gdrive_manager.list_files()

# ========================================================
# AI VISION, PTZ AUTO-TRACKING & TELEGRAM ALERTS
# ========================================================

@app.get("/api/ai/status")
def get_ai_status():
    """Retorna el estado de la inferencia IA, FPS, detecciones y auto-tracking."""
    return ai_tracker.get_status()

@app.post("/api/ai/config")
def update_ai_config(req: AIConfigRequest):
    """Actualiza la configuración de detección de IA y Auto-Tracking PTZ."""
    with ai_tracker.lock:
        if req.ai_enabled is not None:
            ai_tracker.ai_enabled = req.ai_enabled
        if req.tracking_enabled is not None:
            ai_tracker.tracking_enabled = req.tracking_enabled
        if req.target_classes is not None:
            ai_tracker.target_classes = req.target_classes
        if req.min_confidence is not None:
            ai_tracker.min_confidence = req.min_confidence
        if req.return_home_enabled is not None:
            ai_tracker.return_home_enabled = req.return_home_enabled
        if req.home_preset is not None:
            ai_tracker.home_preset = req.home_preset
        if req.deadzone_x is not None:
            ai_tracker.deadzone_x = req.deadzone_x
        if req.deadzone_y is not None:
            ai_tracker.deadzone_y = req.deadzone_y
        if req.model_name is not None:
            ai_tracker.switch_model(req.model_name)
    ai_tracker.save_config()
    return ai_tracker.get_status()

@app.get("/api/ai/models")
def get_ai_models():
    """Retorna lista de modelos YOLO ONNX disponibles en models/ y el modelo activo."""
    return {
        "active_model": ai_detector.active_model_name,
        "model_type": ai_detector.model_type,
        "models": ai_detector.get_available_models()
    }

@app.post("/api/ai/model")
def switch_ai_model(req: SwitchModelRequest):
    """Cambia el modelo YOLO ONNX activo en tiempo real."""
    success = ai_tracker.switch_model(req.model_name)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo cargar el modelo '{req.model_name}'. Verifique que exista en models/"
        )
    return {
        "status": "success",
        "active_model": ai_detector.active_model_name,
        "model_type": ai_detector.model_type,
        "models": ai_detector.get_available_models()
    }

@app.post("/api/ai/toggle_tracking")
def toggle_ai_tracking():
    """Activa o desactiva rápidamente el auto-tracking de motores PTZ."""
    with ai_tracker.lock:
        ai_tracker.tracking_enabled = not ai_tracker.tracking_enabled
    ai_tracker.save_config()
    return {"tracking_enabled": ai_tracker.tracking_enabled}

@app.get("/api/ai/annotated_frame")
def get_ai_annotated_frame():
    """Devuelve el último fotograma con bounding boxes dibujados por la IA."""
    jpeg = ai_tracker.last_annotated_jpeg
    if not jpeg:
        jpeg = rtsp_worker.latest_frame
    if not jpeg:
        raise HTTPException(status_code=404, detail="No hay fotograma disponible aún")
    return Response(content=jpeg, media_type="image/jpeg")

@app.get("/api/telegram/status")
def get_telegram_status():
    """Estado y configuración actual del bot de alertas de Telegram."""
    return {
        "enabled": telegram_notifier.enabled,
        "configured": bool(telegram_notifier.bot_token and telegram_notifier.chat_id),
        "cooldown_seconds": telegram_notifier.cooldown_seconds,
        "include_photo": telegram_notifier.include_photo,
        "app_url": telegram_notifier.app_url,
        "is_silenced": telegram_notifier.is_silenced()
    }

@app.post("/api/telegram/config")
def update_telegram_config(req: TelegramConfigRequest):
    """Actualiza credenciales y opciones de alerta en Telegram."""
    if req.bot_token is not None:
        telegram_notifier.bot_token = req.bot_token.strip()
    if req.chat_id is not None:
        telegram_notifier.chat_id = req.chat_id.strip()
    if req.enabled is not None:
        telegram_notifier.enabled = req.enabled
    if req.cooldown_seconds is not None:
        telegram_notifier.cooldown_seconds = req.cooldown_seconds
    if req.include_photo is not None:
        telegram_notifier.include_photo = req.include_photo
    if req.app_url is not None:
        telegram_notifier.app_url = req.app_url.strip()
    telegram_notifier.save_config()
    return get_telegram_status()

@app.post("/api/telegram/test")
def test_telegram_alert():
    """Envía un mensaje de prueba a Telegram para verificar el Token y Chat ID."""
    if not telegram_notifier.bot_token or not telegram_notifier.chat_id:
        raise HTTPException(status_code=400, detail="Telegram no configurado (Token o Chat ID faltante)")
    
    photo = rtsp_worker.latest_frame
    res = telegram_notifier.send_message(
        "🔔 *Prueba de Alertas Telegram - Foscam Control Center*\n\n✅ La conexión entre el servidor de IA y Telegram funciona correctamente.",
        photo_bytes=photo
    )
    if res.get("ok"):
        return {"status": "ok", "message": "Mensaje enviado exitosamente a Telegram"}
    else:
        return {"status": "error", "error": res.get("description", "Error desconocido")}

app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
