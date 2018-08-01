import os
import re
import time
import copy
import glob
import pickle

import torch
import numpy as np


class Logs:

    def __init__(self, run, keys, history=None):
        self.run = run
        self.keys = keys
        if history is None:
            history = []
        self.history = history

    @property
    def epoch_done(self):
        return self.run.epoch_done

    @property
    def batch_done(self):
        return self.run.batch_done

    def log(self, data):
        self.history.append(data)

    def log_τ(self):
        if 'τ' in self.keys:
            for i, layer in enumerate(self.run.network.layers):
                self.log(('τ', (i, self._copy_var(layer.τ)),
                         (self.epoch_done, self.batch_done), time.time()))

    def log_error_CL_train(self, error):
        if self.run.verbose:
            print('CL(train_dataset): {:.3f}'.format(error))
        if 'error_CL_train' in self.keys:
            self.log(('error_CL_train', self._copy_var(error),
                      (self.epoch_done, self.batch_done), time.time()))

    def log_loss(self, **kwargs):
        data = {k: self._copy_var(v) for k, v in kwargs.items()}
        if 'loss' in self.keys:
            self.log(('loss', data,
                      (self.epoch_done, self.batch_done), time.time()))

        if self.run.verbose:
            print('{}:{} L: {:.3f}, L_x: {:.3f}, L_z: {:.3f}'.format(
                  self.epoch_done, self.batch_done,
                  kwargs['L'], kwargs['L_x'], kwargs['L_z']))

    def log_array(self, key, value):
        if key in self.keys:
            self.log((key, self._copy_var(value),
                      (self.epoch_done, self.batch_done), time.time()))

    def log_network_state(self, network):
        if 'network_state' in self.keys:
            layer_states = []
            for layer in network.layers:
                layer_states.append(self._copy_var(layer.z))
            self.log(('network_state', layer_states,
                      (self.epoch_done, self.batch_done, network.t), time.time()))

    def log_τs(self, network):
        if 'τs' in self.keys:
            τs = []
            for layer in network.layers:
                τs.append(self._copy_var(layer.τ))
            self.log(('τs', τs,
                      (self.epoch_done, self.batch_done, network.t), time.time()))

    @classmethod
    def _copy_var(cls, v):
        if isinstance(v, torch.Tensor):
            return copy.deepcopy(v.data.cpu().numpy())
        elif isinstance(v, tuple):
            return tuple(cls._copy_var(v_i) for v_i in v)
        elif isinstance(v, list):
            return [cls._copy_var(v_i) for v_i in v]
        else:
            return copy.deepcopy(v)

    def logs_for(self, key):
        return [frame for frame in self.history if frame[0] == key]

    def flush(self, filepath_prefix):
        # don't overwrite data
        if len(self.history) > 0:
            filepath = '{}.logs.pickle'.format(filepath_prefix)
            assert not os.path.exists(filepath)
            with open(filepath, 'wb') as fd:
                pickle.dump(self.history, fd, protocol=pickle.HIGHEST_PROTOCOL)
            self.history = []

    def load_full(self, filepath_prefix):
        logs = []
        for filename in glob.glob('{}[[]e*[]].logs.pickle'.format(glob.escape(filepath_prefix))):
            search = re.search(r'.*\[e(?P<epoch>\d+)\].*.logs.pickle', filename)
            if search is not None:
                epoch = int(search.groupdict()['epoch'])
                logs.append((epoch, filename))
        log = []
        for epoch, filename in sorted(logs):
            with open(filename, 'rb') as fd:
                log_i = pickle.load(fd)
                log.extend(log_i)
        self.history = log + self.history
        return log
