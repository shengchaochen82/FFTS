from datetime import datetime
from collections import OrderedDict
from functools import partial
import builtins
from torch.utils.data import DataLoader
import numpy as np
import sys
import sklearn

class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()

class BaseRunner(object):

    def __init__(self, args, model, dataloader, device, loss_module, optimizer=None, l2_reg=0.01, 
                 print_interval=10, console=True, local_protos=None, global_protos=None, global_param=None):

        self.model = model

        self.train_loader = dataloader[0]
        self.test_loader = dataloader[1]

        # new client for ood testing if cluster self.train_loader[1] else: self.train_loader
        # self.ood_test_loader = self.train_loader
        self.ood_test_loader = dataloader[1]
        self.ood_test_data = dataloader[1]
        
        self.args = args
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = Printer(console=console)

        self.global_param = global_param

        # self.epoch_metrics = OrderedDict()
        self.epoch_output = OrderedDict()

        self.local_protos, self.global_protos = local_protos, global_protos

    def train(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.train_loader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)
