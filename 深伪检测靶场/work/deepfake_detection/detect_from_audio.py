"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import os
import argparse
from os.path import join


import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from pathlib import Path

# from network.models import model_selection

from torch import package
from lfcc import LFCC
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchaudio
import sys

from miniio_op import check_and_download_audio_models

from log_utils import setup_logger
import csv
project_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_dir)
os.chdir(project_dir)

logger = setup_logger()

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

class AudioDataset(Dataset):
    """Torch dataset to load data from a provided directory.

    Args:
        directory_or_path_list: Path to the directory containing wav files to load. Or a list of paths.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        directory_or_path_list: str,
        sample_rate: int = 16_000,
        amount: Optional[int] = None,
        normalize: bool = True,
        trim: bool = True
    ) -> None:
        super().__init__()

        self.trim = trim
        self.sample_rate = sample_rate
        self.normalize = normalize

        # if os.path.is_file(directory_or_path_list):
        #     self._paths = [directory_or_path_list]

        self._paths = [directory_or_path_list]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self._paths[index]

        waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        # resamplling
        if sample_rate != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )

        if waveform.shape[0] > 1:
            waveform = waveform[0].unsqueeze(0)

        if self.trim:
            (
                waveform_trimmed,
                sample_rate_trimmed,
            ) = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, SOX_SILENCE
            )

            if waveform_trimmed.size()[1] > 0:
                waveform = waveform_trimmed
                sample_rate = sample_rate_trimmed

        audio_path = str(path)

        return waveform, sample_rate, str(audio_path)

    def __len__(self) -> int:
        return len(self._paths)

class PadDataset(Dataset):
    def __init__(self, dataset: Dataset, cut: int = 64600, label=None):
        self.dataset = dataset
        self.cut = cut  # max 4 sec (ASVSpoof default)
        self.label = label

    def __getitem__(self, index):
        waveform, sample_rate, audio_path = self.dataset[index]
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]
        if waveform_len >= self.cut:
            if self.label is None:
                return waveform[: self.cut], sample_rate, str(audio_path)
            else:
                return waveform[: self.cut], sample_rate, str(audio_path), self.label
        # need to pad
        num_repeats = int(self.cut / waveform_len) + 1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[:, : self.cut][0]

        if self.label is None:
            return padded_waveform, sample_rate, str(audio_path)
        else:
            return padded_waveform, sample_rate, str(audio_path), self.label

    def __len__(self):
        return len(self.dataset)

class TransformDataset(Dataset):
    """A generic transformation dataset.

    Takes another dataset as input, which provides the base input.
    When retrieving an item from the dataset, the provided transformation gets applied.

    Args:
        dataset: A dataset which return a (waveform, sample_rate)-pair.
        transformation: The torchaudio transformation to use.
        needs_sample_rate: Does the transformation need the sampling rate?
        transform_kwargs: Kwargs for the transformation.
    """

    def __init__(
        self,
        dataset: Dataset,
        transformation: Callable,
        needs_sample_rate: bool = False,
        transform_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self._dataset = dataset

        self._transform_constructor = transformation
        self._needs_sample_rate = needs_sample_rate
        self._transform_kwargs = transform_kwargs

        self._transform = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate, audio_path, label = self._dataset[index]

        if self._transform is None:
            if self._needs_sample_rate:
                self._transform = self._transform_constructor(
                    sample_rate, **self._transform_kwargs
                )
            else:
                self._transform = self._transform_constructor(**self._transform_kwargs)

        return self._transform(waveform), sample_rate, str(audio_path), label

def _build_preprocessing(
    directory_or_audiodataset: Union[Union[str, Path], AudioDataset],
    transform: torch.nn.Module,
    audiokwargs: dict = {},
    transformkwargs: dict = {},
) -> TransformDataset:
    """Generic function template for building preprocessing functions."""
    if isinstance(directory_or_audiodataset, AudioDataset) or isinstance(
        directory_or_audiodataset, PadDataset
    ):
        return TransformDataset(
            dataset=directory_or_audiodataset,
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )
    elif isinstance(directory_or_audiodataset, str) or isinstance(
        directory_or_audiodataset, Path
    ):
        return TransformDataset(
            dataset=AudioDataset(directory=directory_or_audiodataset, **audiokwargs),
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )
    else:
        raise TypeError("Unsupported type for directory_or_audiodataset!")

lfcc = functools.partial(_build_preprocessing, transform=LFCC)

class DoubleDeltaTransform(torch.nn.Module):
    """A transformation to compute delta and double delta features.

    Args:
        win_length (int): The window length to use for computing deltas (Default: 5).
        mode (str): Mode parameter passed to padding (Default: replicate).
    """

    def __init__(self, win_length: int = 5, mode: str = "replicate") -> None:
        super().__init__()
        self.win_length = win_length
        self.mode = mode

        self._delta = torchaudio.transforms.ComputeDeltas(
            win_length=self.win_length, mode=self.mode
        )

    def forward(self, X):
        """
        Args:
             specgram (Tensor): Tensor of audio of dimension (..., freq, time).
        Returns:
            Tensor: specgram, deltas and double deltas of size (..., 3*freq, time).
        """
        delta = self._delta(X)
        double_delta = self._delta(delta)

        return torch.hstack((X, delta, double_delta))

def double_delta(dataset: Dataset, delta_kwargs: dict = {}) -> TransformDataset:
    return TransformDataset(
        dataset=dataset,
        transformation=DoubleDeltaTransform,
        transform_kwargs=delta_kwargs,
    )


def load_our_model(model_name):
    root_path = os.getcwd()
    # cur_path = os.getcwd()
    # while True:
    #     if os.path.split(cur_path)[-1] == "deepmake-detection":
    #         root_path = os.path.split(cur_path)[0]
    #         break
    #     cur_path = os.path.split(cur_path)[0]
    weights_cache = os.path.join(root_path, "weights/%s.pt" % model_name)
    check_and_download_audio_models(model_name, weights_cache)

    if os.path.exists(weights_cache) is False:
        return None
    package_name = model_name
    resouce_name = model_name + ".pt"
    imp = package.PackageImporter(weights_cache)
    loaded_model = imp.load_pickle(package_name, resouce_name)
    return loaded_model


def test_full_image_network(audio_path, model_name, output_path,output_csv,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    # print('Starting: {}'.format(audio_path))
    device = "cuda"
    docker_root = "/df_detect/"
    # docker_root = "/home/sxt/deepmake-detection"

    test_dataset = AudioDataset(audio_path)
    test_dataset = PadDataset(test_dataset, label=1)
    dataset_test = lfcc(
        directory_or_audiodataset=test_dataset,
        transformkwargs={},
    )
    dataset_test = double_delta(dataset_test)

    test_loader = DataLoader(
        dataset_test,
        batch_size=1,
        drop_last=False,
    )
    # for batch_x, _, path in dataset_test:
    #     print(batch_x.shape)
    #     print(path)

    model = load_our_model(model_name)
    if model is None:
        logger.info("模型未找到")
        return
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    if cuda:
        model = model.cuda()
    else:
        device = "cpu"
        model = model.to("cpu")
    save_path = args.output_path
    csv_rows = []
    for batch_x, _, paths, _ in test_loader:

        batch_x = batch_x.to(device)

        pred = model(batch_x)
        bp = torch.sigmoid(pred.detach())
        batch_pred = (torch.sigmoid(pred) + 0.5).int()
        for i, path in enumerate(paths):
            p = os.path.split(path)[-1]
            if batch_pred[i][0].cpu().item() == 1:
                
                file_name = path.split("/")[-1]
                csv_rows.append([file_name, 0])
                file_name1 = file_name.split(".")[0]
                file_fix = file_name.split(".")[1]
                file_name = file_name1 + "_real" + "." + file_fix
                logger.info("置信度 %s: %s" % (file_name, 1-bp[i][0].cpu().item()))
            else:
                
                file_name = path.split("/")[-1]
                rows.append([file_name, 1])
                file_name1 = file_name.split(".")[0]
                file_fix = file_name.split(".")[1]
                file_name = file_name1 + "_fake" + "." + file_fix
                logger.info("置信度 %s: %s" % (file_name, 1-bp[i][0].cpu().item()))
            file_path = os.path.join(save_path, file_name)
            os.system("cp %s %s" % (path, file_path))
     # 保存csv
    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in csv_rows:
            csv_writer.writerow(row)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--audio_path', '-i', type=str)

    p.add_argument('--model_name', '-mi', type=str, default="Base")
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--output_csv', '-oc', type=str, default="./output_csv.csv")
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    csv_path = args.output_csv
    if os.path.exists(csv_path) and os.path.isfile(csv_path):
        # 如果路径存在且为文件，则删除文件
        os.remove(csv_path)

    audio_path = args.audio_path
    
    if audio_path.endswith('.mp3') or audio_path.endswith('.wav'):
        test_full_image_network(**vars(args))
    else:
        audios = os.listdir(audio_path)
        for audio in audios:
            args.audio_path = join(audio_path, audio)
            test_full_image_network(**vars(args))
