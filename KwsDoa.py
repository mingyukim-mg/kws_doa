import numpy as np
from collections import deque
import time

from KWS.kws_infer import predict_from_waveform
#from MLPDOA.doa_infer import predict as doa_predict
from kws_doa.gcc_phat import estimate_direction

class AudioPipeline:
    def __init__(self,
                 chunk_size=0.1,
                 sample_rate=16000,
		 interval_sec=1.0):

        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.chunk_len = int(sample_rate * chunk_size)

        self.buffer = deque(maxlen=10)

	# interval control
        self.last_event_time = 0
        self.interval_sec = interval_sec

    # =========================
    # energy
    # =========================
    def compute_energy(self, audio):
        return np.mean(np.abs(audio))

    # =========================
    # arrival
    # =========================
    def get_arrival_time(self, x):
        x = x - np.mean(x)
        x = np.abs(x)

        if np.max(x) < 1e-6:
            return None

        # threshold 방식 제거 → peak 사용
        return np.argmax(x)

    def extract_arrival_feature(self, audio_4ch):
        times = []

        for ch in audio_4ch:
            t = self.get_arrival_time(ch)
            if t is None:
                return None
            times.append(t)

        raw_times = np.array(times, dtype=np.float32)

        # relative 변환
        rel = raw_times - np.min(raw_times)

        return rel

    # =========================
    # event detection
    # =========================
    def extract_event_segment(self, audio_4ch):
        # mono 변환
        mono = np.mean(audio_4ch, axis=0)

        energy = np.abs(mono)

        # peak 찾기
        peak_idx = np.argmax(energy)

        # ±20ms window
        window = int(0.02 * self.sample_rate)

        start = max(0, peak_idx - window)
        end = min(len(mono), peak_idx + window)

        segment = audio_4ch[:, start:end]

        return segment

    # =========================
    # main
    # =========================
    def process_chunk(self, audio_4ch):

        energy = self.compute_energy(audio_4ch)
        # buffer에는 audio, energy 저장
        self.buffer.append({
            "audio": audio_4ch,
            "energy": energy
        })

        if len(self.buffer) < 10:
            return None

        # interval check
        now = time.time()
        if now - self.last_event_time < self.interval_sec:
            return None

        #t0=time.perf_counter()
        # =========================
        # 1초 audio 생성
        # =========================
        audio_1s_4ch = self.merge_audio_4ch()
        #t1=time.perf_counter()
        audio_1s_mono = np.mean(audio_1s_4ch, axis=0)
        #t2=time.perf_counter()
        # =========================
        # KWS
        # =========================
        kws_result = predict_from_waveform(audio_1s_mono)

        #t3=time.perf_counter()
        # print(f"[PROFILE] "
        #     f"merge={t1-t0:.4f} "
        #     f"mono={t2-t1:.4f} "
        #     f"kws={t3-t2:.4f}"
        # )
        if kws_result["label"] == "background":
            return None



        # =========================
        # event segment 추출
        # =========================
        segment = self.extract_event_segment(audio_1s_4ch)

        # segment 추출 실패
        if segment is None:
            return None

        # =========================
        # GCC-PHAT DOA
        # =========================
        doa_result = estimate_direction(segment)

        final_dir = doa_result["direction"]

        self.last_event_time = now
        self.buffer.clear()
        return {
            "label": kws_result["label"],
            "direction": final_dir,
            "score": kws_result["score"]
        }

    # =========================
    # merge
    # =========================
    def merge_audio_4ch(self):
        audio = [c["audio"] for c in self.buffer]
        audio = np.concatenate(audio, axis=1)
        return audio
    
    def extract_multi_segments(self):
        """
        buffer에서 energy 기준으로 peak를 찾고
        S-2 ~ S+2 총 5개 segment 반환
        """

        # =========================
        # 1. energy 배열
        # =========================
        energies = [c["energy"] for c in self.buffer]

        # =========================
        # 2. peak index
        # =========================
        peak_idx = int(np.argmax(energies))

        # =========================
        # 3. segment 수집
        # =========================
        segments = []

        for offset in [-2, -1, 0, 1, 2]:
            idx = peak_idx + offset

            # 범위 벗어나면 None 처리
            if idx < 0 or idx >= len(self.buffer):
                segments.append(None)
            else:
                segments.append(self.buffer[idx]["audio"])

        return segments


# =========================
# preprocess
# =========================
def preprocess_arrival(rel):

    # 너무 flat → 제거
    if np.std(rel) < 1:
        return None

    # 값 범위 제한
    if np.max(rel) > 50:
        return None

    # 최소값이 여러 개면 제거
    if np.sum(rel == 0) >= 2:
        return None

    return rel.astype(np.float32)



DIR2DEG = {
    "R": 0,
    "FR": 45,
    "F": 90,
    "FL": 135,
    "L": 180,
    "BL": 225,
    "B": 270,
    "BR": 315
}

def vector_fusion(directions, weights):
    x_total = 0.0
    y_total = 0.0

    for d, w in zip(directions, weights):
        if d is None or d == "Nodir":
            continue

        theta = np.deg2rad(DIR2DEG[d])
        x_total += w * np.cos(theta)
        y_total += w * np.sin(theta)

    mag = np.sqrt(x_total**2 + y_total**2)

    if mag < 1e-3:
        return None, 0.0

    theta_final = np.rad2deg(np.arctan2(y_total, x_total))
    if theta_final < 0:
        theta_final += 360

    return theta_final, mag

def angle_to_direction(theta):
    bins = [
        (337.5, 360, "R"),
        (0, 22.5, "R"),
        (22.5, 67.5, "FR"),
        (67.5, 112.5, "F"),
        (112.5, 157.5, "FL"),
        (157.5, 202.5, "L"),
        (202.5, 247.5, "BL"),
        (247.5, 292.5, "B"),
        (292.5, 337.5, "BR"),
    ]

    for low, high, d in bins:
        if low <= theta < high:
            return d
    return "Nodir"