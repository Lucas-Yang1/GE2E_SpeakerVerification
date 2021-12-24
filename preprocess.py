import logging
from pathlib import Path
from multiprocessing.pool import ThreadPool
import numpy as np
from tqdm import tqdm

import audio
from params_data import *

logger_name = 'GE2E_logger'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
log_path = './%s.log' % logger_name.replace('/', '_')
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)
fmt = '%(asctime)s-%(levelname)s\n%(message)s'
datefmt = "%a %d %b %Y %H:%M:%S"
fh.setFormatter(logging.Formatter(fmt, datefmt))
logger.addHandler(fh)


def _process_data(speaker_dirs, dataset_name, dataset_root:Path, out_dir: Path, extension, skip_existing):

    logger.info('\n'+'-'*50 + '\n%s : Processing data for %d Speakers.' % (dataset_name, len(speaker_dirs)))

    def process_data(speaker_dir: Path):
        speaker_name = '_'.join(speaker_dir.relative_to(dataset_root).parts)

        speaker_out_dir = out_dir.joinpath(speaker_name)
        speaker_out_dir.mkdir(parents=True, exist_ok=True)
        sources_fpath = speaker_out_dir.joinpath('_sources.txt')

        if sources_fpath.exists():
            try:
                with open(sources_fpath, 'r') as sources_file:
                    exsiting_fnames = {line.split(',')[0] for line in sources_file}
            except:
                exsiting_fnames = {}
        else:
            exsiting_fnames = {}

        sources_file = sources_fpath.open("a" if skip_existing else "w")
        for in_fpath in speaker_dir.glob('**/*.%s' % extension):
            out_fname = '_'.join(in_fpath.relative_to(speaker_dir).parts)
            out_fname = out_fname.replace(".%s" % extension, '.npy')
            if skip_existing and out_fname in exsiting_fnames:
                continue

            wav = audio.preprocess_wav(in_fpath)
            if len(wav) == 0:
                continue
            frames = audio.wav_to_mel_spectrogram(wav)
            if len(frames) < partials_n_frames:
                continue

            out_fpath = speaker_out_dir.joinpath(out_fname)
            np.save(out_fpath, frames)
            logger.info('{}, {} : {}'.format(out_fname, 'duaration', len(wav) / sampling_rate))
            sources_file.write("%s,%s\n" % (out_fname, in_fpath))

        sources_file.close()
    with ThreadPool(8) as pool:
        list(tqdm(pool.imap(process_data, speaker_dirs), dataset_name, len(speaker_dirs), unit='speakers'))
    logger.info('finished all')
    print("Done preprocessing %s.\n" % dataset_name)

dataset_root =Path('D:\\dataset\\aidatatang_200zh\\aidatatang_200zh~\\aidatatang_200zh')
out_dir =Path('./post_data')

def process_adatatang(dataset_root:Path , out_dir:Path, skip_existing=False):
    dataset_name = 'adatatang_200zh'
    speaker_dirs = list(Path(dataset_root).joinpath("corpus", "train").glob('G[0-9][0-9][0-9][0-9]'))

    _process_data(speaker_dirs, dataset_name, dataset_root, out_dir, 'wav', skip_existing)