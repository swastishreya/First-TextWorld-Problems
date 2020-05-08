import os
import IPython
from time import time
import torch
from enum import Enum
import numpy as np

def make_path(path):
    prefix = ''
    if path[0] == '/':
        prefix = '/'
        path = path[1:]

    dirs = path.split("/")
    dirs = ['{}{}'.format(prefix, "/".join(dirs[:i+1])) for i in range(len(dirs))]
    for _dir in dirs:
        if not os.path.isdir(_dir):
            os.makedirs(_dir)


Event = Enum('Event', 'NEWEPOCH STARTTRAINING NEWEPISODE SAVEMODEL TRACKSTATS')

class EventHandler:
    Event = Event
    def __init__(self):
        self.handlers = {event_name:[] for event_name in [event.name for event in Event]}

    def add(self, handler, event):
        self.handlers[event.name].append(handler)
        return self

    def remove(self, handler, event):
        self.handlers[event.name].remove(handler)
        return self

    def __call__(self, event, **kwargs):
        for handler in self.handlers[event.name]:
            handler(**kwargs)

class StepCounter:
    def __init__(self, batch_size=1, max_nb_steps=100):
        self.batch_size = batch_size
        self.max_nb_steps = max_nb_steps
        self.counter = {
            'epoch': 0,
            'episode': 0,
            'global_steps': 0,
            'steps': 0,
            'steps_taken': np.array([0] * self.batch_size)
        }

    def __call__(self, key):
        return self.counter[key]

    def new_episode(self):
        self.counter['episode'] += 1
        self.counter['steps'] = 0
        self.counter['steps_taken'] = np.array([0] * self.batch_size)

    def step(self):
        self.counter['steps'] += 1
        self.counter['global_steps'] += 1

    def new_epoch(self):
        self.counter['epoch'] += 1

    def increase_steps_taken(self, idx):
        self.counter['steps_taken'][idx] += 1

    def recompute_steps_taken(self, just_finished_mask):
        self.counter['steps_taken'] = [self.counter['steps'] if jf else st for jf, st in zip(just_finished_mask, self.counter['steps_taken'])]

class Saver:
    def __init__(self, model, ckpt_path='NOPATH', experiment_tag='NONAME', load_pretrained=False,
                 pretrained_model_path=None, device='cpu', save_frequency=600):
        self.model = model
        self.device = device
        self.model_checkpoint_path = ckpt_path
        self.experiment_tag = experiment_tag

        self.last_save_time = time()
        self.save_frequency = save_frequency

        self.only_load = ckpt_path == 'NOPATH'

        if load_pretrained and pretrained_model_path is not None:
            self.pretrained_model_path = pretrained_model_path
            self._load_from_checkpoint()

    def save(self, epoch=None, episode=None):
        if time() - self.last_save_time > self.save_frequency and not self.only_load:
            self._save_checkpoint(epoch, episode)
            self.last_save_time = time()

    def _save_checkpoint(self, epoch=None, episode=None):
        """
        Save the model checkpoint.
        """
        if self.only_load:
            return
        make_path(self.model_checkpoint_path)
        save_to = "{}/{}".format(self.model_checkpoint_path, self.experiment_tag)
        if epoch is not None:
            save_to += '_epoch{}'.format(epoch)
        if episode is not None:
            save_to += '_episode{}'.format(episode)

        torch.save(self.model.state_dict(), save_to)
        print("Saved model to '{}'".format(save_to))
        self.last_save_time = time()


    def _load_from_checkpoint(self):
        load_from = self.pretrained_model_path
        # print("Trying to load model from {}.".format(load_from))
        try:
            if self.device == 'cpu':
                state_dict = torch.load(load_from, map_location='cpu')
            else:
                state_dict = torch.load(load_from, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            print("Loaded model from '{}'".format(load_from))
        except:
            print("Failed to load checkpoint {} ...".format(load_from))
            IPython.embed()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class flist(list):
    def append(self, object_):
        if isinstance(object_, list):
            [super(flist, self).append(o) for o in object_]
        else:
            super(flist, self).append(object_)
