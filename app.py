import pyaudio
import numpy as np
import time
import sys
import torch

# Simple PyTorch binaural processor: apply interaural time difference (ITD)
class BinauralProcessor:
	def __init__(self, rate=44100, max_delay_ms=2.0, device=None):
		self.rate = rate
		self.max_delay = int(rate * max_delay_ms / 1000)
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

	def apply_itd(self, audio_np, azimuth=0.0):
		# audio_np: numpy array shape (n, 1) mono or (n,2) stereo
		# We'll treat input as mono and create stereo with ITD
		if audio_np.ndim == 2 and audio_np.shape[1] == 2:
			mono = audio_np.mean(axis=1)
		elif audio_np.ndim == 2 and audio_np.shape[1] == 1:
			mono = audio_np[:, 0]
		else:
			mono = audio_np

		x = torch.from_numpy(mono.astype(np.float32)).to(self.device)
		# ITD: map azimuth (-1..1) to delay samples (-max_delay..max_delay)
		delay = int(azimuth * self.max_delay)
		if delay > 0:
			# right delayed
			left = x
			right = torch.cat([torch.zeros(delay, device=self.device), x[:-delay]])
		elif delay < 0:
			d = -delay
			left = torch.cat([torch.zeros(d, device=self.device), x[:-d]])
			right = x
		else:
			left = x
			right = x

		stereo = torch.stack([left, right], dim=1).cpu().numpy()
		return stereo

# 3Dパンニング関数（左右・前後・上下の簡易処理）
def pan_audio(data, azimuth=0.0, elevation=0.0):
	# data: np.ndarray, shape=(n, 2) stereo
	# azimuth: -1.0(左)〜1.0(右), elevation: -1.0(下)〜1.0(上)
	left_gain = 1.0 - max(0, azimuth)
	right_gain = 1.0 + min(0, azimuth)
	# elevationはここでは未使用（HRTF等で拡張可能）
	data[:, 0] *= left_gain
	data[:, 1] *= right_gain
	return data

CHUNK = 1024
FORMAT = pyaudio.paFloat32
RATE = 44100

def open_streams():
	p = pyaudio.PyAudio()
	try:
		in_info = p.get_default_input_device_info()
		out_info = p.get_default_output_device_info()
	except Exception as e:
		print("入力または出力デバイスが見つかりません:", e)
		p.terminate()
		sys.exit(1)

	in_ch = int(in_info.get('maxInputChannels', 0))
	out_ch = int(out_info.get('maxOutputChannels', 0))

	print(f"Default input device: {in_info.get('name')} (maxInputChannels={in_ch})")
	print(f"Default output device: {out_info.get('name')} (maxOutputChannels={out_ch})")

	if in_ch == 0:
		print("入力デバイスがモノラル入力をサポートしていません。終了します。")
		p.terminate()
		sys.exit(1)
	if out_ch == 0:
		print("出力デバイスがありません。終了します。")
		p.terminate()
		sys.exit(1)

	# Open separate input and output streams to avoid channel mismatch errors
	input_stream = p.open(format=FORMAT,
						  channels=in_ch,
						  rate=RATE,
						  input=True,
						  frames_per_buffer=CHUNK)

	output_stream = p.open(format=FORMAT,
						   channels=out_ch,
						   rate=RATE,
						   output=True,
						   frames_per_buffer=CHUNK)

	return p, input_stream, output_stream, in_ch, out_ch


def run():
	p, input_stream, output_stream, in_ch, out_ch = open_streams()
	use_torch = True
	binaural = BinauralProcessor(rate=RATE)

	print("リアルタイム3D立体音響システム 起動中... Ctrl+Cで終了")
	try:
		while True:
			in_data = input_stream.read(CHUNK, exception_on_overflow=False)
			audio = np.frombuffer(in_data, dtype=np.float32)
			# reshape according to input channels
			audio = audio.reshape(-1, in_ch)

			# Convert input -> output channel count
			if in_ch == 1 and out_ch == 2:
				# mono -> stereo: duplicate channel
				audio = np.repeat(audio, 2, axis=1)
			elif in_ch == 2 and out_ch == 1:
				# stereo -> mono: average channels
				audio = audio.mean(axis=1, keepdims=True)
			elif in_ch != out_ch:
				# other cases: adapt by trimming or padding
				if in_ch > out_ch:
					audio = audio[:, :out_ch]
				else:
					pad_width = out_ch - in_ch
					audio = np.pad(audio, ((0, 0), (0, pad_width)), mode='constant')

			# If output is stereo, apply processing
			if out_ch == 2:
				if use_torch:
					# Convert to mono numpy for ITD processing
					mono = audio.mean(axis=1)
					stereo = binaural.apply_itd(mono, azimuth=0.5)
					audio = stereo
				else:
					audio = pan_audio(audio, azimuth=0.5)

			output_stream.write(audio.astype(np.float32).tobytes())
	except KeyboardInterrupt:
		print('\n停止中...')
	finally:
		input_stream.stop_stream()
		input_stream.close()
		output_stream.stop_stream()
		output_stream.close()
		p.terminate()


if __name__ == '__main__':
	run()
