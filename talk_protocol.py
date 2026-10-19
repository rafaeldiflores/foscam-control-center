"""
Foscam FI9821W V2 Two-Way Audio Talkback Protocol Module
Reverse-engineered from Foscam camera firmware and browser plugin DLLs (npIPcam.dll, fsnet_2.dll).

Protocol Architecture:
- Handshake: HTTP Upgrade (SERVERPUSH) -> FOSC Opcode 12 (Login, 164B) -> FOSC Opcode 4 (Start Talk, 160B)
- Codec: Mode 0 (Raw Linear PCM 16-bit 8000Hz mono little-endian)
- Chunking: 960 bytes per chunk (480 samples = 60ms)
- Inner Frame Header: 4-byte chunk-length prefix (struct.pack('<I', 960))
- Outer FOSC Header: 12-byte header (struct.pack('<I4sI', 6, b'FOSC', 964))
- Total Packet: 12 + 4 + 960 = 976 bytes per media packet streamed every 58-60ms
"""

import math
import os
import queue
import socket
import struct
import subprocess
import tempfile
import threading
import time
from typing import Optional

# Standard IMA-ADPCM Step Table used by Foscam (kept for compatibility/utilities)
STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544,
    598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878,
    2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894,
    6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818,
    18500, 20350, 22385, 24623, 27086, 29794, 32767
]

INDEX_TABLE = [-1, -1, -1, -1, 2, 4, 6, 8]

class FoscamAdpcmEncoder:
    """Encodes 16-bit 8000Hz mono linear PCM into Foscam IMA-ADPCM nibbles."""
    def __init__(self):
        self.predicted_sample = 0
        self.step_index = 0

    def encode_sample(self, sample: int) -> int:
        sample = max(-32768, min(32767, sample))
        diff = sample - self.predicted_sample
        sign = 0
        if diff < 0:
            sign = 8
            diff = -diff
        step = STEP_TABLE[self.step_index]
        delta = (4 * diff) // step
        if delta > 7:
            delta = 7

        vpdiff = (step * delta) // 4 + step // 8
        if sign:
            self.predicted_sample -= vpdiff
        else:
            self.predicted_sample += vpdiff
        self.predicted_sample = max(-32768, min(32767, self.predicted_sample))

        self.step_index += INDEX_TABLE[delta]
        self.step_index = max(0, min(88, self.step_index))
        return sign | delta

    def encode_pcm(self, pcm_samples) -> bytes:
        out = bytearray()
        for i in range(0, len(pcm_samples) - 1, 2):
            nib0 = self.encode_sample(pcm_samples[i])
            nib1 = self.encode_sample(pcm_samples[i+1])
            out.append((nib0 << 4) | (nib1 & 0x0F))
        return bytes(out)


class FoscamTalkSession:
    """
    Maintains a paced talkback socket session to the camera's physical speaker.
    Deadlock-free, thread-safe, and self-healing.
    Uses Mode 0 Raw Linear PCM (16-bit 8000Hz mono, 960B chunks, 4-byte chunk prefix).
    """
    CHUNK_SIZE = 960  # 480 samples = 60ms of 16-bit mono 8000Hz PCM

    def __init__(self, ip: str = "192.168.1.13", port: int = 88, user: str = "admin", pwd: str = ""):
        self.ip = ip
        self.port = port
        self.user = user
        self.pwd = pwd
        self.sock: Optional[socket.socket] = None
        self.is_active = False
        self.pcm_buffer = bytearray()
        self.lock = threading.RLock()
        self.audio_queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_timer: Optional[threading.Timer] = None
        self.last_audio_time = 0.0

    def is_connected(self) -> bool:
        with self.lock:
            return bool(self.is_active and self.sock and self.worker_thread and self.worker_thread.is_alive())

    def start(self) -> bool:
        with self.lock:
            # Cancel any pending auto-stop
            if self.stop_timer:
                try:
                    self.stop_timer.cancel()
                except Exception:
                    pass
                self.stop_timer = None

            # If already connected and worker thread is alive, reuse session
            if self.is_connected():
                return True

            # If partially open or worker dead, cleanly close first
            self._do_stop_locked()

            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)
                self.sock.connect((self.ip, self.port))

                # Step 1: Protocol Upgrade from HTTP to raw FOSC media socket
                upgrade_req = (
                    f"SERVERPUSH / HTTP/1.1\r\n"
                    f"Host: {self.ip}:{self.port}\r\n"
                    f"Accept:*/*\r\n"
                    f"Connection: Close\r\n\r\n"
                ).encode("ascii")
                self.sock.sendall(upgrade_req)
                time.sleep(0.08)

                # Step 2: FOSC Login Handshake (Opcode 12)
                user_bytes = self.user.encode("latin-1").ljust(64, b'\x00')
                pwd_bytes = self.pwd.encode("latin-1").ljust(64, b'\x00')
                token = struct.pack('<I', 0)
                login_payload = user_bytes + pwd_bytes + token + (b'\x00' * 32)
                hdr = struct.pack('<I4sI', 12, b'FOSC', len(login_payload))
                self.sock.sendall(hdr + login_payload)
                self.sock.recv(4096)

                # Step 3: Request Start Talk Channel (Opcode 4)
                talk_payload = user_bytes + pwd_bytes + token + (b'\x00' * 28)
                hdr4 = struct.pack('<I4sI', 4, b'FOSC', len(talk_payload))
                self.sock.sendall(hdr4 + talk_payload)
                time.sleep(0.05)
                resp4 = self.sock.recv(4096)
                if len(resp4) >= 16:
                    res = struct.unpack('<I', resp4[12:16])[0]
                    if res != 0:
                        print(f"Warning: Camera Start Talk returned non-zero result code {res}")

                self.pcm_buffer.clear()
                # Empty any leftover in audio_queue
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except Exception:
                        break

                self.is_active = True
                self.last_audio_time = time.time()
                self.worker_thread = threading.Thread(target=self._paced_sender_loop, daemon=True)
                self.worker_thread.start()
                return True
            except Exception as e:
                print(f"FoscamTalkSession start error: {e}")
                self._cleanup_socket()
                return False

    def _paced_sender_loop(self):
        """
        Worker thread that streams 960-byte PCM chunks with 4-byte chunk prefix at exact ~58ms pacing.
        Packet format:
        [12B FOSC header: opcode 6, b'FOSC', payload_len 964] + [4B chunk_len 960] + [960B audio data] = 976B total
        """
        hdr6 = struct.pack('<I4sII', 6, b'FOSC', self.CHUNK_SIZE + 4, self.CHUNK_SIZE)
        next_deadline = time.perf_counter()

        while self.is_active:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None or not self.is_active:
                break

            if len(chunk) < self.CHUNK_SIZE:
                chunk = chunk.ljust(self.CHUNK_SIZE, b'\x00')
            elif len(chunk) > self.CHUNK_SIZE:
                chunk = chunk[:self.CHUNK_SIZE]

            pkt = hdr6 + chunk

            try:
                if self.sock:
                    self.sock.sendall(pkt)
            except Exception as e:
                print(f"Error sending talk chunk: {e}")
                self.is_active = False
                break

            next_deadline += 0.058
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_deadline = time.perf_counter()

    @staticmethod
    def _apply_gain(pcm_data: bytes, gain: float) -> bytes:
        """Amplifies 16-bit PCM with soft clipping (tanh-style knee) to avoid harsh distortion."""
        if gain == 1.0 or len(pcm_data) < 2:
            return pcm_data
        n = len(pcm_data) // 2
        samples = struct.unpack(f'<{n}h', pcm_data[:n * 2])
        out = []
        for s in samples:
            v = s * gain
            if v > 24000:
                v = 24000 + (v - 24000) * 0.15
            elif v < -24000:
                v = -24000 + (v + 24000) * 0.15
            out.append(int(max(-32768, min(32767, v))))
        return struct.pack(f'<{n}h', *out)

    def send_pcm_bytes(self, pcm_data: bytes, gain: float = 1.0):
        """Processes 16-bit 8000Hz mono linear PCM bytes and enqueues 960-byte chunks."""
        if not pcm_data:
            return
        # Cancel any pending auto-stop because new audio is arriving
        if self.stop_timer:
            try:
                self.stop_timer.cancel()
            except Exception:
                pass
            self.stop_timer = None

        pcm_data = self._apply_gain(pcm_data, gain)
        with self.lock:
            if not self.is_active or not self.sock:
                return
            self.last_audio_time = time.time()
            self.pcm_buffer.extend(pcm_data)
            while len(self.pcm_buffer) >= self.CHUNK_SIZE:
                chunk = bytes(self.pcm_buffer[:self.CHUNK_SIZE])
                self.pcm_buffer = self.pcm_buffer[self.CHUNK_SIZE:]
                self.audio_queue.put(chunk)

    def stop_async(self, drain_time: float = 1.8):
        """
        Gracefully drains audio queue and schedules session teardown after drain_time.
        If user speaks again before drain_time expires, the session is kept alive seamlessly.
        """
        with self.lock:
            if not self.is_active:
                return
            self._flush_partial_pcm()
            # Trailing silence frames
            for _ in range(2):
                self.audio_queue.put(b'\x00' * self.CHUNK_SIZE)

            if self.stop_timer:
                try:
                    self.stop_timer.cancel()
                except Exception:
                    pass
            self.stop_timer = threading.Timer(drain_time, self._execute_stop)
            self.stop_timer.daemon = True
            self.stop_timer.start()

    def _execute_stop(self):
        with self.lock:
            self._do_stop_locked()

    def _flush_partial_pcm(self):
        if len(self.pcm_buffer) > 0:
            rem = len(self.pcm_buffer) % self.CHUNK_SIZE
            if rem > 0:
                self.pcm_buffer.extend(b'\x00' * (self.CHUNK_SIZE - rem))
            while len(self.pcm_buffer) >= self.CHUNK_SIZE:
                chunk = bytes(self.pcm_buffer[:self.CHUNK_SIZE])
                self.pcm_buffer = self.pcm_buffer[self.CHUNK_SIZE:]
                self.audio_queue.put(chunk)

    def stop(self, drain_time: float = 1.5):
        """
        Drains queue safely (with timeout, NEVER deadlocks) and stops talkback session cleanly.
        """
        with self.lock:
            if self.stop_timer:
                try:
                    self.stop_timer.cancel()
                except Exception:
                    pass
                self.stop_timer = None
            if not self.is_active:
                self._cleanup_socket()
                return
            self._flush_partial_pcm()
            # Trailing silence frames
            for _ in range(2):
                self.audio_queue.put(b'\x00' * self.CHUNK_SIZE)

        # Wait for queue to empty with 8-second hard timeout (NEVER deadlock!)
        deadline = time.time() + 8.0
        while not self.audio_queue.empty() and time.time() < deadline:
            time.sleep(0.04)

        # Allow camera hardware DAC buffer time to play out
        time.sleep(max(0.2, drain_time))

        with self.lock:
            self._do_stop_locked()

    def _do_stop_locked(self):
        self.is_active = False
        try:
            self.audio_queue.put_nowait(None)
        except Exception:
            pass
        if self.sock:
            try:
                user_bytes = self.user.encode("latin-1").ljust(64, b'\x00')
                pwd_bytes = self.pwd.encode("latin-1").ljust(64, b'\x00')
                token = struct.pack('<I', 0)
                stop_payload = user_bytes + pwd_bytes + token + (b'\x00' * 28)
                hdr = struct.pack('<I4sI', 5, b'FOSC', len(stop_payload))
                self.sock.sendall(hdr + stop_payload)
                time.sleep(0.05)
            except Exception:
                pass
        self._cleanup_socket()

    def _cleanup_socket(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        self.is_active = False
        self.pcm_buffer.clear()
        # Drain remaining items so queue is empty
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break


def generate_sound_effect(name: str) -> bytes:
    """Generates synthetic 16-bit 8000Hz mono PCM audio for sound effects."""
    SAMPLE_RATE = 8000
    samples = []

    if name == "siren":
        # Alternating European / Police siren (3.0s total, 0.25s per tone)
        # Resonates at maximum volume on the camera piezo speaker
        duration = 3.0
        tone_len = int(SAMPLE_RATE * 0.25)
        phase = 0.0
        for i in range(int(duration * SAMPLE_RATE)):
            freq = 960.0 if ((i // tone_len) % 2 == 0) else 720.0
            phase += 2.0 * math.pi * freq / SAMPLE_RATE
            if phase > 2.0 * math.pi:
                phase -= 2.0 * math.pi
            samples.append(int(30000.0 * math.sin(phase)))

    elif name == "chime":
        # Two-tone loud doorbell (Ding-Dong): 659.25 Hz (E5) -> 523.25 Hz (C5)
        # Note 1: E5 (0.5s)
        for t in range(int(SAMPLE_RATE * 0.5)):
            decay = math.exp(-t / (SAMPLE_RATE * 0.35))
            val = int(28000 * math.sin(2 * math.pi * 659.25 * t / SAMPLE_RATE) * decay)
            samples.append(val)
        # Silence (0.08s)
        samples.extend([0] * int(SAMPLE_RATE * 0.08))
        # Note 2: C5 (0.8s)
        for t in range(int(SAMPLE_RATE * 0.8)):
            decay = math.exp(-t / (SAMPLE_RATE * 0.50))
            val = int(30000 * math.sin(2 * math.pi * 523.25 * t / SAMPLE_RATE) * decay)
            samples.append(val)

    elif name == "alert":
        # Triple high-pitch warning beep (1050 Hz)
        for _ in range(3):
            for t in range(int(SAMPLE_RATE * 0.15)):
                val = int(28000 * math.sin(2 * math.pi * 1050 * t / SAMPLE_RATE))
                samples.append(val)
            samples.extend([0] * int(SAMPLE_RATE * 0.10))

    elif name == "dog":
        # Two synthetic barking impulses
        for _ in range(2):
            for t in range(int(SAMPLE_RATE * 0.22)):
                decay = math.exp(-t / (SAMPLE_RATE * 0.10))
                freq = 320 - (t / SAMPLE_RATE) * 300
                noise = ((t * 7919) % 200 - 100) * 80
                val = int(24000 * math.sin(2 * math.pi * freq * t / SAMPLE_RATE) * decay + noise * decay)
                samples.append(max(-32768, min(32767, val)))
            samples.extend([0] * int(SAMPLE_RATE * 0.18))

    else:
        # Default clean beep (880 Hz)
        for t in range(int(SAMPLE_RATE * 0.5)):
            samples.append(int(26000 * math.sin(2 * math.pi * 880 * t / SAMPLE_RATE)))

    return struct.pack(f'<{len(samples)}h', *samples)


def transcode_audio_to_pcm(audio_bytes: bytes) -> bytes:
    """Uses FFmpeg to transcode any audio file (mp3, wav, ogg, webm, m4a, aac, flac) to 8000Hz 16-bit mono PCM."""
    if not audio_bytes:
        return b""
    cmd = [
        'ffmpeg',
        '-i', 'pipe:0',
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ac', '1',
        '-ar', '8000',
        'pipe:1'
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out, _ = proc.communicate(input=audio_bytes)
    if out and len(out) > 0:
        return out

    # Fallback with temporary file for formats requiring seekable input
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(audio_bytes)
            temp_path = tf.name
        cmd2 = [
            'ffmpeg',
            '-y',
            '-i', temp_path,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-ar', '8000',
            'pipe:1'
        ]
        proc2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out2, _ = proc2.communicate()
        return out2 or b""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
