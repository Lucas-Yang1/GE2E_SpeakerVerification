import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from params_data import *


class SpeakerVerificationDataset(Dataset):
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.speakers = [Speaker(speaker) for speaker in dataset_root.glob('*') if speaker.is_dir()]
        if len(self.speakers) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speaker_cycles = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)

    def __getitem__(self, item):
        return next(self.speaker_cycles)

class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker
        super(SpeakerVerificationDataLoader, self).__init__(
            dataset=dataset,
            batch_size=speakers_per_batch,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames)

class SpeakerBatch:
    def __init__(self, speakers: list, utterances_per_speaker, n_frames):
        self.speakers = speakers
        self.partial = {s:s.random_partial(utterances_per_speaker, n_frames) for s in speakers}

        self.data = np.array([frames for s in self.speakers for _, frames, _ in self.partial[s]])


class Speaker:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycle = None

    def _load_utterance(self):
        with open(self.root.joinpath('_sources.txt'), 'r') as source_files:
            sources = [l.split(',') for l in source_files]

        self.utterances = [Utterance(self.root.joinpath(x[0]), x[1]) for x in sources]
        self.utterance_cycle = RandomCycler(self.utterances)

    def random_partial(self, count: int, n_frames: int):
        if self.utterances == None:
            self._load_utterance()

        sample_utterances = self.utterance_cycle.sample(count)

        out = [(u,) + u.random_partial(n_frames) for u in sample_utterances]

        return out


class Utterance:
    def __init__(self, frame_fpath, wave_fpath):
        self.frame_fpath = frame_fpath
        self.wave_fpath = wave_fpath

    def get_frames(self):
        return np.load(self.frame_fpath)

    def random_partial(self, n_frames: int):

        frames = self.get_frames()

        if len(frames) == n_frames:
            start = 0

        else:
            start = random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)


class RandomCycler:
    """
    无限循环随机序列生成器，对输入进来的list，进行打乱。
    抽过的item，只有在其他的所有的items都抽完之后，才再有可能抽到。
    """
    def __init__(self, source):

        if len(source) == 0:
            raise Exception("Can't create RandomCycler from an empty collection")

        self.all_item = list(source)
        self.next_item = []

    def sample(self, count: int):
        shuffle = lambda l: random.sample(l, len(l))
        samples = []

        while count > 0:
            if count > len(self.all_item):
                samples.extend(shuffle(list(self.all_item)))
                count -= len(self.all_item)
            n = min(count, len(self.next_item))
            samples.extend(self.next_item[:n])
            count -= n
            self.next_item = self.next_item[n:]
            if len(self.next_item) == 0:
                self.next_item = shuffle(list(self.all_item))

        return samples

    def __next__(self):
        return self.sample(1)[0]

if __name__ == '__main__':
    root = Path('./post_data')
    dataset = SpeakerVerificationDataset(root)
    dataloader = SpeakerVerificationDataLoader(dataset, 10, 20)