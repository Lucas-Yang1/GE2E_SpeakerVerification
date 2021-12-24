import torch
from tqdm import tqdm

import audio
from audio import *
from model import SpeakerEncoder
from params_data import *
from functools import partial
from multiprocessing.pool import ThreadPool as Pool

_model: SpeakerEncoder = None
_device: torch.device = None


def _load_model(weight_fpath, device=None):
    global _model, _device
    if device == None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        _device = torch.device(device)

    _model = SpeakerEncoder(_device, 'cpu')
    _model.load_state_dict(torch.load(weight_fpath))
    _model.to(device)

    _model.eval()


def embed_from_frames_batch(frames_batch):
    frames = torch.from_numpy(frames_batch).to(_device)
    embed = _model.forward(frames).detach().cpu().numpy()
    return embed


def compute_partial_slices(n_samples, partial_utterance_n_frames=partials_n_frames,
                           min_pad_coverage=0.75, overlap=0.5):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to obtain
    partial utterances of <partial_utterance_n_frames> each. Both the waveform and the mel
    spectrogram slices are returned, so as to make each partial utterance waveform correspond to
    its spectrogram. This function assumes that the mel spectrogram parameters used are those
    defined in params_data.py.

    The returned ranges may be indexing further than the length of the waveform. It is
    recommended that you pad the waveform with zeros up to wave_slices[-1].stop.

    :param n_samples: the number of samples in the waveform
    :param partial_utterance_n_frames: the number of mel spectrogram frames in each partial
    utterance
    :param min_pad_coverage: when reaching the last partial utterance, it may or may not have
    enough frames. If at least <min_pad_coverage> of <partial_utterance_n_frames> are present,
    then the last partial utterance will be considered, as if we padded the audio. Otherwise,
    it will be discarded, as if we trimmed the audio. If there aren't enough frames for 1 partial
    utterance, this parameter is ignored so that the function always returns at least 1 slice.
    :param overlap: by how much the partial utterance should overlap. If set to 0, the partial
    utterances are entirely disjoint.
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
    respectively the waveform and the mel spectrogram with these slices to obtain the partial
    utterances.
    """
    assert 0 <= overlap < 1
    assert 0 < min_pad_coverage <= 1

    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partial_utterance_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
    """
    Computes an embedding for a single utterance.

    # TODO: handle multiple wavs to benefit from batching on GPU
    :param wav: a preprocessed (see audio.py) utterance waveform as a numpy array of float32
    :param using_partials: if True, then the utterance is split in partial utterances of
    <partial_utterance_n_frames> frames and the utterance embedding is computed from their
    normalized average. If False, the utterance is instead computed from feeding the entire
    spectogram to the network.
    :param return_partials: if True, the partial embeddings will also be returned along with the
    wav slices that correspond to the partial embeddings.
    :param kwargs: additional arguments to compute_partial_splits()
    :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If
    <return_partials> is True, the partial utterances as a numpy array of float32 of shape
    (n_partials, model_embedding_size) and the wav partials as a list of slices will also be
    returned. If <using_partials> is simultaneously set to False, both these values will be None
    instead.
    """
    # Process the entire utterance if not using partials
    if not using_partials:
        frames = wav_to_mel_spectrogram(wav)
        embed = embed_from_frames_batch(frames[None, ...])[0]
        if return_partials:
            return embed, None, None
        return embed

    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

    # Split the utterance into partials
    frames = wav_to_mel_spectrogram(wav)
    frames_batch = np.array([frames[s] for s in mel_slices])
    partial_embeds = embed_from_frames_batch(frames_batch)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    if return_partials:
        return embed, partial_embeds, wave_slices
    return embed


def inference(dataset_root: Path, out_dir: Path, with_speaker_dir: bool = True, using_partials: bool = True,
              n_process: int = 2):

    speaker_dirs = [speaker for speaker in dataset_root.glob("*") if speaker.is_dir()]

    func = partial(_embed_speaker_per_utterance, out_dir=out_dir, with_speaker_dir=with_speaker_dir, using_partials=using_partials)
    job = Pool(n_process).imap(func, speaker_dirs)

    list(tqdm(job, "Speaker_embedding", len(speaker_dirs), unit="speaker"))



def _embed_speaker_per_utterance(speaker_dir: Path, out_dir: Path, with_speaker_dir: bool,using_partials: bool):
    extensions = ["*.wav"]

    for extension in extensions:
        for wav_fpath in speaker_dir.glob(extension):
            wav = audio.preprocess_wav(wav_fpath)
            embed = embed_utterance(wav, using_partials=using_partials)
            sub_basename = "%s_%02d" % (wav_fpath.name, 0)
            embedding_name = "embedding-%s.npy" % sub_basename
            _out_dir = out_dir.joinpath(speaker_dir.name) if with_speaker_dir else out_dir
            _out_dir.mkdir(parents=True, exist_ok=True)
            np.save(_out_dir.joinpath(embedding_name), embed)
if __name__ == '__main__':
    _load_model("./archive/300000_softmax_model.pth")
    dataset_root = Path('D:/dataset/aidatatang_200zh/corpus/train')
    out_dir = Path("../Synthesis_Tacotron/postdata/aidatatang_200zh/embedding")
    inference(dataset_root, out_dir)