import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
import os
import json
import math

class SGDRScheduler(Callback):
    def __init__(self,
                 path_to_checkpoints = 'Checkpoints',
                 min_lr = 7e-5,
                 start_cyclical = 100,
                 base_lr = 7e-3,
                 cyclical_base_lr = 7e-4,
                 cyclical_epoch=10,
                 warmup_epoch=10,
                 last_epoch=0):
        super(SGDRScheduler, self).__init__()

        self.path_to_checkpoints = path_to_checkpoints
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.start_cyclical = start_cyclical
        self.cyclical_epoch = cyclical_epoch
        self.cyclical_base_lr = cyclical_base_lr
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        self.next_checkpoint_epoch = 10

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        if self.last_epoch < self.warmup_epoch:
            return self.min_lr + self.last_epoch*(self.base_lr - self.min_lr)/self.warmup_epoch
        elif self.last_epoch < self.start_cyclical:
            return self.min_lr + (self.base_lr-self.min_lr)*(1+math.cos(math.pi*(self.last_epoch-self.warmup_epoch)/(self.start_cyclical-self.warmup_epoch))) / 2
        else:
            return self.min_lr + (self.cyclical_base_lr-self.min_lr)*(1+math.cos(math.pi* ((self.last_epoch-self.start_cyclical)% self.cyclical_epoch)/self.cyclical_epoch)) / 2

    def on_train_begin(self, start_epoch, logs={}):
        logs = logs or {}
        if start_epoch>0:
            self.last_epoch = start_epoch
            self.next_checkpoint_epoch = start_epoch
            if self.next_checkpoint_epoch <=80:
                self.next_checkpoint_epoch += 10
            else:
                self.next_checkpoint_epoch += 10
            K.set_value(self.model.optimizer.learning_rate, self.clr())
        else:
            K.set_value(self.model.optimizer.learning_rate, self.min_lr)

    def on_epoch_end(self, epoch, losses, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if self.last_epoch + 1 == self.next_checkpoint_epoch:
            folder_cp = os.path.join(self.path_to_checkpoints, 'epoch_{}'.format(epoch))
            path_cp = os.path.join(folder_cp, 'epoch_{}'.format(epoch))
            path_json = os.path.join(folder_cp, 'epoch_{}.json'.format(epoch))
            self.model.save_weights(path_cp)
            K.set_value(self.model.optimizer.learning_rate, self.clr())
            if self.next_checkpoint_epoch <=80:
                self.next_checkpoint_epoch += 10
            else:
                self.next_checkpoint_epoch += 10
            logs['epoch'] = epoch
            logs['path_cp'] = path_cp
            logs['l'] = losses
            with open(path_json, 'w') as f:
                json.dump(logs, f)
        K.set_value(self.model.optimizer.learning_rate, self.clr())
        print('Loss: ', losses[-1], 'next lr:', self.model.optimizer.learning_rate)
        self.last_epoch = self.last_epoch + 1
    def on_train_end(self, logs={}):
        logs= logs or {}
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        print('Finish training!')

if __name__=='__main__':
    import matplotlib.pyplot as plt
    scheduler_warmup = SGDRScheduler(min_lr=7e-5, warmup_epoch=10, start_cyclical=100, cyclical_base_lr=3.5e-3, cyclical_epoch=10)
    lr = []
    for epoch in range(0,150):
        lr.append(scheduler_warmup.on_epoch_end())
    plt.style.use('ggplot')
    plt.plot(list(range(0,150)), lr)
    plt.show()