from datetime import datetime

from dataset import SpeakerVerificationDataset, SpeakerVerificationDataLoader
from params_model import *
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from model import SpeakerEncoder
import torch
import logging

def get_logger():
    logger = logging.getLogger('training')
    fh = logging.FileHandler('training.log')
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger
def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(dataset_root: Path, total_steps=250_000, schedule_fn=get_linear_schedule_with_warmup, loss_fn='softmax',
          logging_interval_steps=50, out_dir='archive', save_steps=5e4):

    dataloader = SpeakerVerificationDataLoader(SpeakerVerificationDataset(dataset_root),
                                               speakers_per_batch, utterances_per_speaker,
                                               num_workers=4)
    logger = get_logger()
    logger.info("-"*80)
    logger.info(datetime.now().strftime("%c"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_device = torch.device('cpu')


    model = SpeakerEncoder(device, loss_device)
    if loss_fn == 'contrast':
        model.get_loss = model.contrast_loss
    else:
        loss_fn = 'softmax'
        model.get_loss = model.softmax_loss

    optimizer = AdamW(model.parameters(), lr=learning_rate_init)
    p = 0.15
    if schedule_fn != None:
        schedule = schedule_fn(optimizer,
                               num_warmup_steps=int(total_steps*p),
                               num_training_steps=total_steps)

    logger.info("dataset_root: %s\ttotal_steps: %d\tloss_fn: %s\tschedule_fn: %s\tp: %f\tdevice: %s\tloss_device: %s" %
                (dataset_root, total_steps, loss_fn, schedule_fn.__name__, p, device, loss_device))
    init_step = 1
    model.train()
    for step, speaker_batch in enumerate(dataloader, init_step):
        t1 = datetime.now()
        # pre-process
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)

        # forward pass
        embeds = model(inputs)
        sync(device)
        embeds = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss = model.get_loss(embeds)
        sync(loss_device)

        # backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()

        optimizer.step()
        if schedule_fn != None:
            schedule.step()
        t2 = datetime.now()
        if step > total_steps:
            break
        if step % logging_interval_steps == 0:
            logger.info(f"[{step}/{total_steps}] loss: {loss:.4f}, time_cost: {(t2-t1).total_seconds():.2f}")

    logger.info("-"*40 + "final" + "-" * 40)

