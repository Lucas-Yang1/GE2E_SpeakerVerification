from datetime import datetime

from dataset import SpeakerVerificationDataset, SpeakerVerificationDataLoader
from params_model import *
from transformers import AdamW, get_linear_schedule_with_warmup
from pathlib import Path
from model import SpeakerEncoder
import torch
import logging
from visualization import Visualiztion
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

def train(dataset_root: Path, total_steps=300_000, schedule_fn=get_linear_schedule_with_warmup, loss_fn='softmax',
          logging_interval_steps=50, out_dir:Path = Path('archive'), save_steps=5_000, init_step=None):

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

    viz = Visualiztion(loss_fn)
    if init_step is not None:
        model.load_state_dict(torch.load(out_dir.joinpath('%d_%s_model.pth' % (init_step, loss_fn))))
    optimizer = AdamW(model.parameters(), lr=learning_rate_init)
    p = 0.15
    if schedule_fn != None:
        schedule = schedule_fn(optimizer,
                               num_warmup_steps=int(total_steps*p),
                               num_training_steps=total_steps)
        if init_step is not None:
            schedule._step_count = init_step
            schedule.step()

    logger.info("dataset_root: %s\ttotal_steps: %d\tloss_fn: %s\tschedule_fn: %s\tp: %f\tdevice: %s\tloss_device: %s\tcurrent_lr: %d" %
                (dataset_root, total_steps, loss_fn, schedule_fn.__name__, p, device, loss_device, schedule.get_lr()[0]))
    init_step = 1 if init_step == None else init_step+1
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

        viz.update(loss.item(), step)
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
            logger.info(f"[{step}/{total_steps}] loss: {loss.item():.4f}, time_cost: {(t2-t1).total_seconds():.2f}")

        if step % save_steps == 0:
            logger.info(f"save step : {step} model   out_path: {out_dir.joinpath(str(step)+'_'+ loss_fn+'_model.pth')}")
            torch.save(model.state_dict(), out_dir.joinpath(str(step)+'_'+ loss_fn+'_model.pth'))
            viz.save()

    logger.info("-"*40 + "final" + "-" * 40)
    viz.save()

if __name__ == '__main__':
    dataset_root = Path('./postdata')
    train(dataset_root, loss_fn='softmax', init_step=120_000)
    # train(dataset_root, loss_fn='contrast')