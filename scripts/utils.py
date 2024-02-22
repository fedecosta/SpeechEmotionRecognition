# Imports
import os
import logging
import psutil
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import numpy as np
# ---------------------------------------------------------------------
# Logging

# Set logging config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
# ---------------------------------------------------------------------

def format_training_labels(labels_path, labels_to_ids, prepend_directory = None, header = False):
        
    '''Format training type labels.'''

    # Expected labels line input format (tab separated): audio_file_path\tlabel_string
    # prepend_directory will be prepended to each audio file path

    # Read the paths of the audios and their labels
    with open(labels_path, 'r') as data_labels_file:
        labels_lines = data_labels_file.readlines()
    
    if header:
        labels_lines = labels_lines[1:]

    # Format labels lines
    formatted_labels_lines = []
    for labels_line in labels_lines:

        if len(labels_line.split("\t")) != 2:
            raise Exception(f'line {labels_line} has not 2 columns!')
        assert len(labels_line.split("\t")) == 2, f"line {labels_line} has not 2 columns!"
        
        file_path = labels_line.split("\t")[0]

        # We will assign each label a number using a fixed dictionary
        label = labels_line.split("\t")[1].replace("\n", "")
        label = labels_to_ids[label]

        # Prepend optional additional directory to the labels paths (but first checks if file exists)
        if prepend_directory is not None:
            file_path = os.path.join(prepend_directory, file_path) 
        data_founded = os.path.exists(file_path)
        assert data_founded, f"{file_path} not founded."

        labels_line = f"{file_path}\t{label}"
        
        formatted_labels_lines.append(labels_line)

    return formatted_labels_lines


def generate_model_name(params, start_datetime, wandb_run_id = None, wandb_run_name = None):

        # TODO add all neccesary components

        name_components = []

        formatted_datetime = start_datetime.replace(':', '_').replace(' ', '_').replace('-', '_')
        name_components.append(formatted_datetime)

        name_components.append(params.feature_extractor)
        name_components.append(params.front_end)
        name_components.append(params.adapter)
        name_components.append(params.seq_to_seq_method)
        name_components.append(params.seq_to_one_method)
        if wandb_run_id: name_components.append(wandb_run_id)
        if wandb_run_name: name_components.append(wandb_run_name)

        name_components = [str(component) for component in name_components]
        model_name = "_".join(name_components)

        return model_name


def get_memory_info(cpu = True, gpu = True):

    cpu_available_pctg, gpu_free = None, None

    # CPU memory info
    if cpu:
        cpu_memory_info = dict(psutil.virtual_memory()._asdict())
        cpu_total = cpu_memory_info["total"]
        cpu_available = cpu_memory_info["available"]
        cpu_available_pctg = cpu_available * 100 / cpu_total

    # GPU memory info
    if gpu:
        if torch.cuda.is_available():
            gpu_free, gpu_occupied = torch.cuda.mem_get_info()
            gpu_free = gpu_free/1000000000
        else:
            gpu_free = None

    return cpu_available_pctg, gpu_free


def pad_collate(batch_data):

    input, label, transcription_tokens = zip(*batch_data)

    # Input and labels processing

    # input is a tuple of tensors and label is a tuple of np.arrays of len 1.
    # we need to convert each of these to a tensor of arrays
    input = torch.stack(input)
    label = torch.tensor(np.array(label).flatten())

    # Tokens processing

    transcription_tokens_padded = pad_sequence(transcription_tokens, batch_first=True, padding_value=0)

    # We are going to define padding masks tensors because of the following suggestion:
    # We strongly recommend passing in an `attention_mask` since your input_ids may be padded. 
    # See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.

    transcription_tokens_lens = [len(x) for x in transcription_tokens]
    max_len = max(transcription_tokens_lens)
    transcription_tokens_mask = [torch.nn.functional.pad(torch.ones(len), (0, max_len-len), mode = "constant", value = 0) for len in transcription_tokens_lens]
    transcription_tokens_mask = torch.tensor(np.array(transcription_tokens_mask))

    return input, label, transcription_tokens_padded, transcription_tokens_mask


def get_waveforms_stats(labels_lines, sample_rate):

    audio_paths = [line.split("\t")[0] for line in labels_lines]

    logger.info(f"Computing {len(audio_paths)} waveforms statistics...")

    count = 0
    wav_sum = 0
    wav_sqsum = 0
    
    for audio_num, audio_path in enumerate(audio_paths):
        
        waveform, loaded_sample_rate = torchaudio.load(audio_path, sample_rate)

        assert loaded_sample_rate == sample_rate, f"loaded_sample_rate ({loaded_sample_rate}) should match sample_rate ({sample_rate})!"

        waveform = waveform.squeeze().numpy()
        wav_sum += np.sum(waveform)
        wav_sqsum += np.sum(waveform**2)
        count += len(waveform)

    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)

    logger.info(f"Waveforms statistics computed. Mean: {wav_mean}, std: {wav_std}")

    return wav_mean, wav_std

