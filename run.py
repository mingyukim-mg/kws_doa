from kws_doa.mic_stream import MicStream
from kws_doa.KwsDoa import AudioPipeline

pipeline = AudioPipeline()
mic = MicStream()

print("Start streaming...")

for chunk in mic.stream():

    try:
        result = pipeline.process_chunk(chunk)
        print("result:", result)

    except Exception as e:
        print("❌ error in pipeline:", e)