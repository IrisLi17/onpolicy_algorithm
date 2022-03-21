import torch
import torch.nn as nn


class ActorCriticPolicy(nn.Module):
    # @property
    # def is_recurrent(self):
    #     return self._is_recurrent
    #
    # @property
    # def recurrent_hidden_state_size(self):
    #     return self._recurrent_hidden_state_size
    #
    # @is_recurrent.setter
    # def is_recurrent(self, value):
    #     self._is_recurrent = value
    #
    # @recurrent_hidden_state_size.setter
    # def recurrent_hidden_state_size(self, value):
    #     self._recurrent_hidden_state_size = value

    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        raise NotImplementedError
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        raise NotImplementedError

    def get_value(self, obs, rnn_hxs=None, rnn_masks=None):
        value, dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        return value

    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        raise NotImplementedError
