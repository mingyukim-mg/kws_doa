import numpy as np
from collections import deque
import time

# ===== 외부 모델 =====
from KWS.kws_infer import predict_from_waveform  # KWS
from MLPDOA.doa_infer import predict as doa_predict  # DOA

class AudioPipeline:
    def __init__(self,
                 chunk_size=0.1,
                 sample_rate=16000,
                 interval_sec=1.0):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.chunk_len = int(sample_rate * chunk_size)

        # 1초 buffer (10개)
        self.buffer = deque(maxlen=10)

        # interval control
        self.last_event_time = 0
        self.interval_sec = interval_sec

    # =========================
    # 1. energy 계산
    # =========================
    def compute_energy(self, audio):
        return np.mean(np.abs(audio))

    # =========================
    # 2. arrival feature
    # =========================
    def get_arrival_time(self, x, ratio=0.5):
        x = x - np.mean(x)
        x = np.abs(x)

        max_val = np.max(x)
        if max_val < 1e-6:
            return None

        threshold = max_val * ratio

        for i, v in enumerate(x):
            if v >= threshold:
                return i
        return None

    def extract_arrival_feature(self, audio_4ch):
        times = []
        for ch in audio_4ch:
            t = self.get_arrival_time(ch)
            if t is None:
                t = 0
            times.append(t)

        # normalize
        times = np.array(times, dtype=np.float32)
        if np.max(times) > 0:
            times = times / np.max(times)

        return times

    # =========================
    # 3. chunk 처리
    # =========================
    def process_chunk(self, audio_4ch):
        """
        audio_4ch: (4, T)
        """

        energy = self.compute_energy(audio_4ch)
        arrival = self.extract_arrival_feature(audio_4ch)

        self.buffer.append({
            "audio": audio_4ch,
            "energy": energy,
            "arrival": arrival
        })

        # buffer 부족하면 skip
        if len(self.buffer) < 10:
            return None

        # interval check
        now = time.time()
        if now - self.last_event_time < self.interval_sec:
            return None

        # =========================
        # 4. KWS
        # =========================
        audio_1s = self.merge_audio()

        kws_result = predict_from_waveform(audio_1s)

        if not kws_result["detected"]:
            return None

        if kws_result["label"] == "background":
            return {
                "label": "background",
                "direction": None,
                "score": kws_result["score"],
            }
        # =========================
        # 5. DOA
        # =========================
        chunk = self.select_chunk()

        direction = doa_predict(chunk["arrival"])

        self.last_event_time = now

        return {
            "label": kws_result["label"],
            "direction": direction,
            "score": kws_result["score"]
        }

    # =========================
    # 4. buffer → 1초 오디오
    # =========================
    def merge_audio(self):
        """
        (4, T) * 10 → (T_total,)
        mono로 합쳐서 KWS 입력
        """

        audio = [c["audio"] for c in self.buffer]
        audio = np.concatenate(audio, axis=1)  # (4, total)

        # mono 변환
        audio = np.mean(audio, axis=0)

        return audio

    # =========================
    # 5. chunk 선택 (핵심)
    # =========================
    def select_chunk(self):
        energies = np.array([c["energy"] for c in self.buffer])
        mean_E = np.mean(energies)

        # 너무 작은 값 제거
        valid = [
            c for c in self.buffer
            if c["energy"] > 0.4 * mean_E
        ]

        if len(valid) == 0:
            valid = list(self.buffer)

        # 가장 뒤 chunk
        return valid[-1]