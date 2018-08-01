import os
import re
import glob
import time
import resource
import importlib

import yaml
import torch
import numpy as np
import reproducible
from ensure import ensure, check

from . import Task, Dataset
from . import utils
from . import logs


class Run:

    def __init__(self, params, checkpoint=None, resume=True,
                       verbose=True, configpath=None):
        self.verbose = verbose
        self.beginning = time.time()
        self.context = reproducible.Context(repo_path=None)

        if params['computation']['num_thread'] is not None:
            torch.set_num_threads(params['computation']['num_thread'])

        if resume:
            checkpoint = self.load_checkpoint(params['checkpoints']['filepath'],
                                              must_exist=False)
            # if checkpoint is not None:
            #     # network parametrization *must* be the same
            #     for key in ['classname', 'layers']:
            #         ensure(params['network'][key]).equals(checkpoint['params']['network'][key])

        # even when loading from a checkpoint, the new parameters are used.
        self.params = params

        log_keys = params.get('logs', ['loss', 'error_CL_train'])
        if checkpoint is None:  # start from scratch
            self.batch_done = 0  # number of batches that have been computed.
            self.epoch_done = 0  # number of epoch that have been computed.
            self.logs = logs.Logs(self, log_keys, [])
            assert (params['seeds']['torch_seed']  <
                    params['seeds']['python_seed'] <
                    params['seeds']['dataset_seed'] )
            torch.random.manual_seed(params['seeds']['torch_seed'])
            self.configpath = configpath

        else:
            self.batch_done = checkpoint['batch_done']
            self.epoch_done = checkpoint['epoch_done']
            self.logs = logs.Logs(self, log_keys, checkpoint.get('logs', []))
            torch.random.set_rng_state(checkpoint['rng_states']['torch'])

        if verbose:
            print('Configuration:\n{}'.format(utils.yaml_pprint(self.params)))

        # device
        self.device = self.params['computation']['device']
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.context.add_data('device', str(self.device))

        # dataset & network
        datashape_in, datashape_out = self._create_datasets(
            self.params['seeds']['dataset_seed'], checkpoint=checkpoint)
        self._create_network(datashape_in, datashape_out, checkpoint=checkpoint)
        if verbose:
            print('Model loaded on {}.'.format(self.network.device))

        # task
        self.task = Task(self.network,
                         w=params['loss']['w'],
                         grad_clip=params['training'].get('grad_clip', 1.0),
                         grad_clip_norm_type=params['training'].get('grad_clip_norm_type', 'inf'),
                         loss_fun=params['loss']['loss_fun'],
                         learning_rate=params['training']['learning_rate'],
                         logs=self.logs)
        if checkpoint is not None:
            self.task.optimizer.load_state_dict(checkpoint['optimizer'])

        self.context.add_repo('.', allow_dirty=True)
        self.context.add_data('parameters', self.params)

    def _create_network(self, datashape_in, datashape_out, checkpoint=None):
        """Create the network"""

        net_cls = utils.import_str(self.params['network']['classname'])
        self.network = net_cls(self.params['network'],
                               datashape_in, datashape_out, device=self.device,
                               batch_size=self.params['training']['batch_size'],
                               jit=self.params['computation']['jit'])
        self.network.logs = self.logs
        if checkpoint is not None:
            self.network.load_state_dict(checkpoint['network'])

    def _create_datasets(self, seed, checkpoint=None):
        """Create the train, validate and test datasets"""
        # loading the data
        datapath = self._normpath(self.params['dataset']['filepath'])
        with open(datapath, 'rb') as fd:
            samples = np.load(fd)

        # shape of samples should be (n_samples, n_timesteps, datadim)
        assert len(samples.shape) == 3

        # create the datasets
        rng_states = checkpoint['rng_states'] if checkpoint is not None else {}

        self.train_dataset    = Dataset(seed=seed, device=self.device,
                                        rng_state=rng_states.get('train_dataset', None))
        self.validate_dataset = Dataset(seed=seed+1, device=self.device,
                                        rng_state=rng_states.get('validate_dataset', None))
        self.test_dataset     = Dataset(seed=seed+2, device=self.device,
                                        rng_state=rng_states.get('test_dataset', None))

        n_train    = self.params['training']['n_train']
        n_validate = self.params['training']['n_validate']
        n_test     = self.params['training']['n_test']
        for index, sample in zip(range(n_train), samples[:n_train]):
            self.train_dataset.add_sample(sample, index=index)
        for index, sample in zip(range(n_train, n_train+n_validate),
                                 samples[n_train:n_train+n_validate]):
            self.validate_dataset.add_sample(sample, index=index)
        for index, sample in zip(range(n_train+n_validate, n_train+n_validate+n_test),
                                 samples[n_train+n_validate:n_train+n_validate+n_test]):
            self.test_dataset.add_sample(sample, index=index)

        return ((n_train, samples.shape[1], samples.shape[2]),
                (n_train, samples.shape[1], samples.shape[2]))

    def load_logs(self):
        self.logs.load_full(self.params['checkpoints']['filepath'])


        # Save & Load

    def save(self, flush_logs=True):
        assert self.params['checkpoints']['filepath'] is not None
        filename = self._filename(create_dirs=True)
        if flush_logs:
            self.logs.flush(filename)
        checkpoint = {
            'params'    : self.params,
            'batch_done': self.batch_done,
            'epoch_done': self.epoch_done,
            'network'   : self.network.state_dict(),
            'optimizer' : self.task.optimizer.state_dict(),
            'logs'      : self.logs.history,
            'rng_states': {
                'torch': torch.random.get_rng_state(),
                'train_dataset': self.train_dataset.random.getstate(),
                'validate_dataset': self.validate_dataset.random.getstate(),
                'test_dataset': self.test_dataset.random.getstate()
            }
        }
        torch.save(checkpoint, filename + '.run.torch')
        self.context.add_file(filename + '.run.torch', category='output',
                              already=True)
        self.context.export_yaml(filename + '.tracks.yaml',
                                 update_timestamp=True)

    def _filename(self, create_dirs=False):
        filename = '{}[e{}]'.format(self.params['checkpoints']['filepath'],
                                     self.epoch_done)
        if create_dirs and os.path.dirname(filename) != '':
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        return filename

    @classmethod
    def load_checkpoint(cls, filepath, must_exist=True):
        """Find and load a checkpoint"""
        if os.path.isfile(filepath):  # assuming we got a complete filepath
            max_filename = filepath
        else:
            max_epoch, max_filename = -1, None
            for full_filename in glob.glob('{}[[]e*[]].run.torch'.format(glob.escape(filepath))):
                search = re.search(r'.*\[e(?P<epoch>\d+)\].*.run.torch', full_filename)
                if search is not None:
                    epoch = int(search.groupdict()['epoch'])
                    if epoch > max_epoch:
                        max_epoch, max_filename = epoch, full_filename
        if max_filename is not None:
            print('loading {}'.format(max_filename))
            return torch.load(max_filename)
        elif must_exist:
            raise FileNotFoundError('no file matching `{}[e*].run.torch` found'.format(filepath))
        return None  # if none found and allowed.

    @classmethod
    def _normpath(cls, filepath):
        return os.path.abspath(os.path.expanduser(filepath))

    @classmethod
    def from_configfile(cls, filepath, *args, **kwargs):
        filepath = cls._normpath(filepath)
        configpath = os.path.dirname(filepath)
        params = utils.load_configfile(filepath)
        rootdir = os.path.normpath(os.path.dirname(filepath))
        os.chdir(rootdir)
        instance = cls(params, *args, **kwargs)
        instance.context.add_file(filepath, category='input', already=True)
        return instance


        # Train & Test


    def _batches(self, dataset):
        return dataset.mini_batches(self.params['training']['batch_size'])

    def _check_nan(self, L):
        """Check if the loss is NaN."""
        if torch.isnan(L):
            #self.save(flush_logs=False)
            raise RuntimeError('The loss is NaN.')

    def epoch(self):
        """Run a training epoch"""
        for batch in self._batches(self.train_dataset):
            _, _, _, L, L_z, L_x = self.task.train(batch)
            self.batch_done += 1
            self._check_nan(L)
            self.logs.log_loss(L=L, L_x=L_x, L_z=L_z)
            self.logs.log_τ()
        self.epoch_done += 1

    def run(self, save:bool=True):
        """Train & test the network

        :param save:  if True, will save at the end.
        """
        start_time = time.time()

        while self.epoch_done < self.params['training']['n_epoch']:
            self.epoch()
            self.test()
            # saving
            if self.epoch_done % self.params['checkpoints']['period'] == 0:
                self.save()

            # computing speed display
            if self.verbose and self.epoch_done % 10 == 0:
                duration = time.time() - start_time
                print('{:.2f}s/epoch (last 10 epochs) total: {:.1f}s'.format(
                               duration/10, time.time() - self.beginning))
                start_time = time.time()

        if save:
            self.save()


    def test(self):
        """Run any scheduled test after an epoch"""
        test_cfg = self.params['testing']
        if 'error_CL_train' in test_cfg:
            if self.epoch_done % test_cfg['error_CL_train']['period'] == 0:
                error = self.error_closed_loop(self.train_dataset)
                self.logs.log_error_CL_train(error)

    def error_closed_loop(self, dataset):
        """Compute the closed-loop error over a dataset"""
        mse = 0
        for batch in self._batches(dataset):
            mse += self.task.test_closed_loop(batch)
        return mse / len(dataset)
