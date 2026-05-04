import sounddevice as sd
import numpy as np

DEVICE_INDEX = 1
class MicStream:
    def __init__(self,
                 sample_rate=16000,
                 channels=6,
                 chunk_size=0.1):

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        self.frames_per_chunk = int(sample_rate * chunk_size)
    
    def stream(self):

      print("stream 진입")

      try:
          with sd.InputStream(
              samplerate=self.sample_rate,
              channels=self.channels,
              dtype='float32',
              blocksize=self.frames_per_chunk,
              device=DEVICE_INDEX
          ) as stream:

              #print("InputStream 열림")

              while True:
                #print("read 시도")
                data, _ = stream.read(self.frames_per_chunk)
                data = data[:, 1:5]
                #print("chunk 받음:", data.shape)
                
                yield data.T

      except Exception as e:
          print("stream error:", e)