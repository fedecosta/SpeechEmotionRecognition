import torchaudio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

path = "/home/usuaris/scratch/speaker_databases/RIRS_NOISES/pointsource_noises/noise-free-sound-0246.wav"
path = "/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files/MSP-PODCAST_0002_0033.wav"
waveform, waveform_sample_rate = torchaudio.load(path)
waveform = waveform.to(device)
print("Waveform len: {len(waveform)}")

bundle = torchaudio.pipelines.WAVLM_BASE
print("Sample Rate:", bundle.sample_rate)
if waveform_sample_rate != bundle.sample_rate:
    print("waveform resampled")
    waveform = torchaudio.functional.resample(waveform, waveform_sample_rate, bundle.sample_rate)

model = bundle.get_model().to(device)

with torch.inference_mode():
    features, _ = model.extract_features(waveform)

    for level, feats in enumerate(features):
        print(f"level {level}, feats.size():{feats.size()}")
        #feats dims: (#B, #num_vectors, #dim_vectors = 768)

