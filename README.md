# KWS + DOA Real-time Pipeline

이 프로젝트는 다음 3가지 구성 요소로 이루어집니다:

1. **KWS 모델 (HuggingFace)**
2. **MLP 기반 DOA 모델 (HuggingFace Git)**
3. **실시간 파이프라인 (kws_doa)**

---

## 📁 전체 디렉토리 구조

아래와 같은 구조를 맞춰주세요:

```
project/
├── KWS/
│   └── kws_infer.py
├── MLPDOA/
│   ├── MLPDOA.pth
│   └── ...
├── kws_doa/
│   ├── run.py
│   ├── KwsDoa.py
│   ├── mic_stream.py
│   └── ...
```

## MLPDOA 디렉토리는 따로 만들지 않고 바로 git clone (모델)하면 됩니다.

## 1️⃣ kws_doa (파이프라인 코드)

```bash
cd kws_doa
git clone https://github.com/mingyukim-mg/kws_doa.git
```

---

## 2️⃣ MLPDOA 모델 다운로드

```bash
git clone git clone https://huggingface.co/dbif/MLPDOA
```

MLPDOA디렉토리를 만들어서
cd MLPDOA한 후 하는 게 아닌 루트 디렉토리에서
git clone https://huggingface.co/dbif/MLPDOA를 해야합니다.

> 반드시 `MLPDOA/MLPDOA.pth` 파일이 존재해야 합니다.

---

## 3️⃣ KWS 모델 준비

HuggingFace에서 kws_infer.py 파일을 가져옵니다:

- 모델: dbif/kws_tuning_model(https://huggingface.co/dbif/kws_tuning_model)

다음 파일을 `KWS/` 폴더에 위치시킵니다:

```
KWS/kws_infer.py
```

> 해당 파일은 `AutoModelForAudioClassification`을 사용하는 코드입니다.

---

## 4️⃣ 환경 설정

```bash
pip install torch torchaudio transformers sounddevice numpy
```

---

## 5️⃣ 실행

```bash
python -m kws_doa.run
```

---

## FastAPI 예시 코드

```
from fastapi import FastAPI
import threading

from kws_doa.mic_stream import MicStream
from kws_doa.KwsDoa import AudioPipeline

app = FastAPI()

pipeline = AudioPipeline()
mic = MicStream()

latest_result = None # 최근 결과 저장

def mic_loop():
global latest_result

      print("Mic streaming started...")

      for chunk in mic.stream():
          try:
              result = pipeline.process_chunk(chunk)

              if result:
                  latest_result = result
                  print("result:", result)

          except Exception as e:
              print("❌ error in pipeline:", e)

@app.on_event("startup")
def start_mic():
thread = threading.Thread(target=mic_loop, daemon=True)
thread.start()

@app.get("/latest")
def get_latest():
return latest_result
```

## 실행 흐름

```
Mic Input (4채널)
    ↓
KWS (소리 종류 판단)
    ↓
DOA (방향 추정)
    ↓
결과 출력
```

---

## ⚠️ 주의사항

- 4채널 마이크 필요
- 샘플링레이트 16000Hz 기준
- CPU에서도 동작하지만 속도는 환경에 따라 다름

---

## 현재 상태

- KWS: 안정적으로 동작
- DOA: 정확도 개선 필요 (추후 연구 예정)

---
