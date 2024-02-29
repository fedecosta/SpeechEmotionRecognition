from utils import format_training_labels
import torchaudio
import torch
from random import randint
import pandas as pd
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score
from data import TrainDataset
from utils import format_training_labels, pad_collate
from settings import LABELS_REDUCED_TO_IDS, TRAIN_DEFAULT_SETTINGS
import argparse
from torch.utils.data import DataLoader
from text_feature_extractor import TextBERTExtractor
import os


train_labels_lines = format_training_labels(
    labels_path = './labels/training_labels_reduced_8_classes.tsv',
    labels_to_ids = LABELS_REDUCED_TO_IDS,
    prepend_directory = '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files',
    header = True,
)

line_num = 51858
utterance_path = train_labels_lines[line_num].split("\t")[0]
file_name = utterance_path.split("/")[-1]
waveform, sample_rate = torchaudio.load(utterance_path)

bundle = torchaudio.pipelines.WAV2VEC2_XLSR_1B
feature_extractor = bundle.get_model()
features, _ = feature_extractor.extract_features(waveform)
print(len(features))
print(features[-1].size())


if False:
    line_num = 51858
    utterance_path = train_labels_lines[line_num].split("\t")[0]
    file_name = utterance_path.split("/")[-1]
    audio, sample_rate = torchaudio.load(utterance_path)
    import whisper
    transcriptor = whisper.load_model("base")

    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
    options = whisper.DecodingOptions(fp16 = False)
    results = whisper.decode(transcriptor, mel, options)

    print(results)


if False:

    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    for line_num, line in enumerate(train_labels_lines):

        utterance_path = train_labels_lines[line_num].split("\t")[0]

        file_name = utterance_path.split("/")[-1]

        #transcription_path = os.path.join("/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Transcripts/Transcripts", file_name)
        transcription_path = os.path.join("/home/usuaris/veussd/federico.costa/datasets/msp_podcast/custom_transcriptions/", file_name)
        transcription_path = transcription_path.replace(".wav", ".txt")

        with open(transcription_path, 'r') as data_labels_file:
            transcription = data_labels_file.readlines()

        if len(transcription) != 1:
            print("len(transcription) != 1")
            print(f"line {line_num}: {line}")
            print(f"file_name: {file_name}")
            break

        transcription = transcription[0]

        tokens = tokenizer.encode(transcription, add_special_tokens=True)
        
        if len(tokens) >= 512:
            print("len(transcription) >= 512")
            print(f"line {line_num}: {line}")
            print(f"file_name: {file_name}")
            print(f"transcription: {transcription}")
            print(f"tokens: {tokens}")

        #tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        #indexed_tokens = tokenizer.encode(transcription, add_special_tokens=True)
        #tokens_tensor = torch.tensor(indexed_tokens)



if False:

    train_labels_lines = format_training_labels(
        labels_path = './labels/training_labels_reduced_7_classes.tsv',
        labels_to_ids = LABELS_REDUCED_TO_IDS,
        prepend_directory = '/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files',
        header = True,
    )

    default_params_dict = TRAIN_DEFAULT_SETTINGS
    params = argparse.Namespace(**default_params_dict)
    params.padding_type = "repetition_pad"
    params.text_feature_extractor = "TextBERTExtractor"
    params.training_batch_size = 3

    training_dataset = TrainDataset(
        utterances_paths = train_labels_lines, 
        input_parameters = params,
        random_crop_secs = 5.5,
        augmentation_prob = 0,
        sample_rate = 16000,
        )

    if params.text_feature_extractor is not None:
        data_loader_parameters = {
            'batch_size': params.training_batch_size, 
            'shuffle': True, # FIX hardcoded True
            'num_workers': params.num_workers,
            'collate_fn': pad_collate,
        }
    else:
        data_loader_parameters = {
            'batch_size': params.training_batch_size, 
            'shuffle': True, # FIX hardcoded True
            'num_workers': params.num_workers,
        }

    # Instanciate a DataLoader class
    training_generator = DataLoader(
        training_dataset, 
        **data_loader_parameters,
        )

    #for batch_number, (input, label) in enumerate(training_generator):
    for batch_number, (input, label, transcription_tokens, transcription_tokens_lens) in enumerate(training_generator):

        # Assign input and label to device
        print(f"batch_number: {batch_number}")
        print(f"input: {input}")
        print(f"label: {label}")
        print(f"transcription_tokens_lens: {transcription_tokens_lens}")
        #print(f"transcription_tokens: {transcription_tokens}")

        text_feature_extractor = TextBERTExtractor()
        max_len = max(transcription_tokens_lens)
        padding_mask = [torch.nn.functional.pad(torch.tensor(torch.ones(len)), (0, max_len-len), mode = "constant", value = 0) for len in transcription_tokens_lens]
        padding_mask = torch.tensor(np.array(padding_mask))
        print(f"padding_mask: {padding_mask}")
        text_feature_extractor_output = text_feature_extractor(transcription_tokens, padding_mask)
        print(f"text_feature_extractor_output.size(): {text_feature_extractor_output.size()}")
        break










if False:
    from torch.nn.utils.rnn import pad_sequence
    transcription_tokens = (
        torch.tensor([ 101, 7994, 1146, 1397, 2106, 1105, 1122, 1125, 1176, 4674, 1104, 3697,1176, 2133, 1324,  102]), 
        torch.tensor([  101, 12373,  3948,  1222,  1103,  2965,  1104,   187,  2328,  3491,5412,  1863,  1755,  1602,  1234,  1110,  1136,  1126,  8050,   102]), 
        torch.tensor([  101,  1865,  1103,  9887,  1133,  1103,  5963,  1108, 24083,  1173,170, 22890,  1146,   170,  2046,  2560,  1131,  1865,  1103,  9887, 1133,  1103,   102]))

    transcription_tokens_padded = pad_sequence(transcription_tokens, batch_first=True, padding_value=0)

    print(transcription_tokens_padded)


if False:
    from text_feature_extractor import ASRModel, TextBERTExtractor
    path = "/home/usuaris/veussd/federico.costa/datasets/msp_podcast/Audios/audio_files/MSP-PODCAST_4685_0031.wav"
    waveform, sample_rate = torchaudio.load(path)
    waveform = torch.cat([waveform, waveform])
    print(f"waveform.size(): {waveform.size()}")

    feature_extractor = TextBERTExtractor()
    features = feature_extractor(waveform)
    print(f"features.size(): {features.size()}")










if False:
    path = "MSP-PODCAST_4685_0031.wav"
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