from datetime import datetime

import numpy as np
import visdom
from time import perf_counter as timer

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255


class Visualiztion:
    def __init__(self, env='GE2E', every_step=10):

        self.viz = visdom.Visdom(env=env)

        self.loss = []
        self.loss_win = None
        self.every_step = every_step
        self.env = env


    def update(self, loss, step):
        self.loss.append(loss)

        if step % self.every_step == 0:
            self.loss_win = self.viz.line(
                [np.mean(self.loss)],
                [step],
                win=self.loss_win,
                env=self.env,
                update='append' if self.loss_win else None,
                opts=dict(
                    legend=["Avg. loss"],
                    xlabel="Step",
                    ylabel="Loss",
                    title="Loss"
                )
            )

        else:
            return
        self.loss.clear()

    def save(self):
        self.viz.save([self.env])
