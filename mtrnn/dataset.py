import random

import numpy as np
import torch

from . import utils


class Dataset:
    """Dataset class, to hold sample and create mini-batches.

    :param seed:  seed for the independent random generator of the Dataset
                  instance.
    :param random_state:  random state to set the independent random generator.
                          if not None, supersede the `seed` parameter.

    May be replaced by/extended into a more torch-integrated class in the
    future.
    """

    def __init__(self, seed:int=0, rng_state=None, device=None):
        self.device = utils.autodevice(device)
        self._shape_in  = None
        self._shape_out = None
        self.samples = {}

        self.random = random.Random(seed)
        if rng_state is not None:
            self.random.setstate(rng_state)

    def __len__(self):
        return len(self.samples)

    @property
    def datashape_in(self):
        return (len(self.samples),) + tuple(self._shape_in)

    @property
    def datashape_out(self):
        return (len(self.samples),) + tuple(self._shape_out)

    def _min_index(self, index=None):
        """Find the smallest (positive) available index for a new sample"""
        if index is not None:
            return index
        index = 0
        while True:
            if index not in self.samples:
                return index
            index += 1

    def add_sample(self, inputs, target=None, index:int=None):
        """Add a sample to the dataset.

        :param inputs:  input data, of size :math:`T, N_in`, with :math:`T` the
                        number of timesteps. We don't yet support different
                        number of timesteps for different samples.
        :param target:  target data. If None, the input is the target data.
        :param index:   if provided, this value will be the sample index.
                        Else, the smallest positive available index will be
                        used.
        """
        if target is None: # inputs are target, but with one timestep of prediction
            target = inputs[1:]
            inputs = inputs[:-1]

        inputs = np.array(inputs, dtype=np.float32)
        inputs_t = torch.from_numpy(inputs).to(self.device)
        if self._shape_in is None:
            self._shape_in = inputs_t.shape
        else:
            assert inputs_t.shape == self._shape_in

        target = np.array(target, dtype=np.float32)
        target_t = torch.tensor(target).to(self.device)
        if self._shape_out is None:
            self._shape_out = target_t.shape
        else:
            assert target_t.shape == self._shape_out
        assert self._shape_in == self._shape_out  # for the moment

        self.samples[self._min_index(index)] = {'inputs': inputs_t,
                                                'target': target_t}

    def sample(self, index: int):
        """Return a sample as a triplet (index, input, target).

        :param index:  the index of the sample to return.
        """
        sample = self.samples[index]
        return (index, sample['inputs'], sample['target'])

    def batch(self, indexes):
        inputs = torch.stack([self.samples[index]['inputs'] for index in indexes], 1)
        target = torch.stack([self.samples[index]['target'] for index in indexes], 1)
        return indexes, inputs, target

    def mini_batches(self, size):
        """Create mini-batch of size `size`.

        Return a list of 3-tuple, with a list of indexes, a matrix of inputs of
        shape :math:`(T, M, N_{in})`, with :math:`T` the number of timesteps and
        :math:`M` the number of samples in the mini-batch.
        """
        indexes = list(self.samples.keys())
        self.random.shuffle(indexes)
        offset, batches = 0, []
        for _ in range(len(self.samples)//size):
            b_indexes = indexes[offset:offset+size]
            offset += size
            batches.append(self.batch(b_indexes))
        return batches

    def load(self, filepath):
        """Load data from a file"""
        with open(filepath, 'rb') as fd:
            samples = np.load(fd)

        for sample in samples:
            self.add_sample(sample)
