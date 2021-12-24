from params_model import *
import torch
import torch.nn as nn
import numpy as np
from params_data import *


class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device, eps=1e-5):
        super(SpeakerEncoder, self).__init__()
        self.loss_device = loss_device

        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=hidden_channel_size,
                            num_layers=model_num_layers,
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=hidden_channel_size,
                                out_features=model_embedding_size).to(device)

        self.relu = nn.ReLU().to(device)

        self.similarity_weight = nn.Parameter(torch.ones(1)).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.zeros(1)).to(loss_device)

        self.eps = eps

        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        self._init_weight()

    def forward(self, utterances, hidden_init=None):
        """
        compute the normed embedding
        utterance: shape of [Speaker_per_Batch * Utterance_per_Speaker, n_frames, features_per_Utterance]
        """

        out, (hidden, cell) = self.lstm(utterances, hidden_init)

        # take only the hidden state of the last layers
        embeds = self.relu(self.linear(hidden[-1]))

        # l2-norm
        embeds_norm = embeds / (torch.norm(embeds, keepdim=True, dim=1) + self.eps)

        return embeds_norm

    def similarity_matrix(self, embeds):
        """
        :param embeds: [Speakers_per_Batch, Utterances_per_Speaker, features_per_Utterance]
        :return:
        """

        speaker_per_batch, utterances_per_speaker = embeds.shape[:2]

        # include
        centroid_inc = torch.mean(embeds, dim=1, keepdim=True)
        centroid_inc = centroid_inc.clone() / (torch.norm(centroid_inc, dim=2, keepdim=True) + self.eps)

        # exclude
        centroid_exc = torch.sum(embeds, dim=1, keepdim=True) - embeds
        centroid_exc /= (utterances_per_speaker - 1)
        centroid_exc = centroid_exc.clone() / (torch.norm(centroid_exc, dim=2, keepdim=True) + self.eps)


        sim_mat = torch.zeros([speaker_per_batch, utterances_per_speaker, speaker_per_batch]).to(self.loss_device)
        mask_matrix = 1 - np.eye(speaker_per_batch)

        for j in range(speaker_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_mat[mask, :, j] = (embeds[mask] * centroid_inc[j]).sum(dim=2)
            sim_mat[j, :, j] = (embeds[j] * centroid_exc[j]).sum(dim=1)

        sim_mat = self.similarity_weight * sim_mat + self.similarity_bias
        return sim_mat

    def do_gradient_ops(self):
        # Scale gradient
        self.similarity_weight.grad *= 0.1
        self.similarity_bias.grad *= 0.1

        nn.utils.clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def softmax_loss(self, embeds):
        """
        :param embeds: [Speakers_per_Batch, Utterances_per_Speaker, Embedding_size]
        :return:
        """

        speaker_per_batch, utterances_per_speaker = embeds.shape[:2]

        sim_mat = self.similarity_matrix(embeds).reshape((speaker_per_batch * utterances_per_speaker, speaker_per_batch))
        ground_truth = np.repeat(np.arange(speaker_per_batch), utterances_per_speaker)
        ground_truth = torch.from_numpy(ground_truth).long().to(self.loss_device)

        loss = self.loss_fn(sim_mat, ground_truth)

        return loss


    def contrast_loss(self, embeds):
        """
        :param embeds: [Speakers_per_Batch, Utterances_per_Speaker, Embedding_size]
        :return:
        """

        speaker_per_batch, utterances_per_speaker = embeds.shape[:2]

        sim_mat = self.similarity_matrix(embeds)
        sim_mat = sim_mat.sigmoid()

        mask_matrix = 1 - np.eye(speaker_per_batch)
        loss = torch.zeros([1], dtype=torch.float).to(self.loss_device)
        for j in range(speaker_per_batch):
            mask = np.where(mask_matrix[j])[0]
            loss += torch.max(sim_mat[j, :, mask], dim=-1)[0].sum() -\
                sim_mat[j, :, j].sum()
        loss /= (speaker_per_batch * utterances_per_speaker)
        loss = 1-loss
        return loss

    def _init_weight(self):
        for m in self.children():
            for param in m.parameters():
                if param.dim() == 1:
                    nn.init.zeros_(param)
                else:
                    nn.init.xavier_normal_(param)




