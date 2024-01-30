from utils import format_training_labels
import torchaudio
from random import randint

path = "/home/usuaris/veussd/federico.costa/datasets/data_augmentation/open_slr/slr_17/speech/us-gov/speech-us-gov-0249.wav"
noise, noise_sample_rate = torchaudio.load(path)






if False:
    
    utterances_paths = format_training_labels(
        labels_path = './labels/development_labels.tsv',
        prepend_directory = '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files',
        header = True,
    )

    index = 0
    sample_rate = 16000
    window_sample_size_secs = 1000000000000 #0.015
    random_crop_samples = window_sample_size_secs * sample_rate

    utterance_tuple = utterances_paths[index].strip().split('\t')
    utterance_path = utterance_tuple[0]
    utterance_label = utterance_tuple[1]

    waveform, sample_rate = torchaudio.load(utterance_path)
    print(f"waveform.size(): {waveform.size()}")

    waveform = waveform.squeeze(0)
    print(f"waveform.size(): {waveform.size()}")

    waveform_total_samples = waveform.size()[-1]
    print(f"waveform_total_samples: {waveform_total_samples}")
    print(f"random_crop_samples: {random_crop_samples}")

    random_start_index = randint(0, waveform_total_samples - random_crop_samples)
    print(f"random_start_index: {random_start_index}")



    end_index = None
    cropped_waveform =  waveform[random_start_index : end_index]
    print(f"cropped_waveform.size(): {cropped_waveform.size()}")

if False:

    waveform_samples = waveform.size()[-1]
    window_sample_size_samples = int(window_sample_size_secs * sample_rate)
    random_start_index = waveform_samples - window_sample_size_samples+1#randint(0, waveform_samples - window_sample_size_samples - 1)
    end_index = random_start_index + window_sample_size_samples

    croped_waveform = waveform[:, random_start_index : end_index]

    print(f"waveform_samples: {waveform_samples}")
    print(f"window_sample_size_samples: {window_sample_size_samples}")
    print(f"random_start_index: {random_start_index}")
    print(f"end_index: {end_index}")
    print(f"croped_waveform: {croped_waveform.size()}")

    #1.5 secs -> 147 frames
    #1.0 secs -> 97 frames