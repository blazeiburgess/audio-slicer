import numpy as np
import os.path
from argparse import ArgumentParser
import librosa
import soundfile


def get_rms(y, *, frame_length=2048, hop_length=512, pad_mode="constant"):
    """
    Calculate the root mean square (RMS) of a signal.
    This function is obtained from librosa.
    """
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)

    target_axis = axis - 1 if axis < 0 else axis + 1
    xw = np.moveaxis(xw, -1, target_axis)

    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


class AudioSlicer:
    def __init__(self, sr: int, threshold: float = -40., min_length: int = 5000,
                 min_interval: int = 300, hop_size: int = 20, max_sil_kept: int = 5000):
        """
        Initialize the AudioSlicer.

        Args:
            sr (int): Sample rate of the audio.
            threshold (float): The dB threshold for silence detection.
            min_length (int): Minimum length of a slice in milliseconds.
            min_interval (int): Minimum silence interval for slicing in milliseconds.
            hop_size (int): Hop size in milliseconds.
            max_sil_kept (int): Maximum silence to keep in milliseconds.
        """
        self._validate_parameters(min_length, min_interval, hop_size, max_sil_kept)

        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _validate_parameters(self, min_length, min_interval, hop_size, max_sil_kept):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('Condition not met: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('Condition not met: max_sil_kept >= hop_size')

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        """
        Slice the input waveform based on silence detection.

        Args:
            waveform (np.ndarray): Input audio waveform.

        Returns:
            list: List of sliced waveform chunks.
        """
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]

        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        silence_tags = self._detect_silence(rms_list)
        return self._create_chunks(waveform, silence_tags, rms_list.shape[0])

    def _detect_silence(self, rms_list):
        silence_tags = []
        silence_start = None
        clip_start = 0

        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue

            if silence_start is None:
                continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            silence_tags.extend(self._process_silence(i, silence_start, clip_start, rms_list))
            silence_start = None
            clip_start = silence_tags[-1][1]

        self._handle_trailing_silence(silence_start, rms_list, silence_tags)
        return silence_tags

    def _process_silence(self, i, silence_start, clip_start, rms_list):
        if i - silence_start <= self.max_sil_kept:
            return self._process_short_silence(i, silence_start, rms_list)
        elif i - silence_start <= self.max_sil_kept * 2:
            return self._process_medium_silence(i, silence_start, rms_list)
        else:
            return self._process_long_silence(i, silence_start, rms_list)

    def _process_short_silence(self, i, silence_start, rms_list):
        pos = rms_list[silence_start: i + 1].argmin() + silence_start
        return [(0, pos)] if silence_start == 0 else [(pos, pos)]

    def _process_medium_silence(self, i, silence_start, rms_list):
        pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin() + i - self.max_sil_kept
        pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
        pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
        if silence_start == 0:
            return [(0, pos_r)]
        else:
            return [(min(pos_l, pos), max(pos_r, pos))]

    def _process_long_silence(self, i, silence_start, rms_list):
        pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
        pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
        return [(0, pos_r)] if silence_start == 0 else [(pos_l, pos_r)]

    def _handle_trailing_silence(self, silence_start, rms_list, silence_tags):
        if silence_start is not None and len(rms_list) - silence_start >= self.min_interval:
            silence_end = min(len(rms_list), silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            silence_tags.append((pos, len(rms_list) + 1))

    def _create_chunks(self, waveform, silence_tags, total_frames):
        if not silence_tags:
            return [waveform]

        chunks = []
        if silence_tags[0][0] > 0:
            chunks.append(self._apply_slice(waveform, 0, silence_tags[0][0]))

        for i in range(len(silence_tags) - 1):
            chunks.append(self._apply_slice(waveform, silence_tags[i][1], silence_tags[i + 1][0]))

        if silence_tags[-1][1] < total_frames:
            chunks.append(self._apply_slice(waveform, silence_tags[-1][1], total_frames))

        return chunks


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('audio', type=str, help='The audio to be sliced')
    parser.add_argument('--out', '-o', type=str, dest='out',
                        help='Output directory of the sliced audio clips')
    parser.add_argument(
        '--db_thresh', '--thresh', type=float, default=-40, dest='db_thresh',
        help='The dB threshold for silence detection'
    )
    parser.add_argument('--min_length', '--min-length', type=int, default=5000,
                        help='The minimum milliseconds required for each sliced audio clip')
    parser.add_argument('--min_interval', '--min-interval', type=int, default=300, dest='min_interval',
                        help='The minimum milliseconds for a silence part to be sliced')
    parser.add_argument('--hop_size', '--hop-size', type=int, default=10, help='Frame length in milliseconds')
    parser.add_argument('--max_sil_kept', '--max-sil-kept', '--max-silence-kept',
                        type=int, default=500, dest='max_sil_kept',
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    return parser.parse_args()


def main():
    args = parse_arguments()
    out_dir = args.out or os.path.dirname(os.path.abspath(args.audio))
    os.makedirs(out_dir, exist_ok=True)

    audio, sr = librosa.load(args.audio, sr=None, mono=False)
    slicer = AudioSlicer(
        sr=sr,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept
    )
    chunks = slicer.slice(audio)

    base_filename = os.path.splitext(os.path.basename(args.audio))[0]
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        output_path = os.path.join(out_dir, f'{base_filename}_{i}.wav')
        soundfile.write(output_path, chunk, sr)


if __name__ == '__main__':
    main()
