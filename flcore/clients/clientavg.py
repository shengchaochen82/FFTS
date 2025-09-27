import time
from flcore.clients.clientbase import Client
from flcore.trainers.trainer import SupervisedRunner
from flcore.trainers.unsup_trainer import UnsupervisedRunner
from flcore.losses.loss import get_loss_module


class clientAVG(Client):
    def __init__(self, args, id, data_name, train_data, test_data, **kwargs):
        super().__init__(args, id, data_name, train_data, test_data, **kwargs)

        if 'pretrain' in args.task:
            self.trainer = UnsupervisedRunner(args=args,
                                            model=self.model,
                                            dataloader=self.load_data(task=args.task),
                                            device=self.device,
                                            loss_module=get_loss_module(args),
                                            optimizer=self.optimizer,
                                            l2_reg=0)
        else: # train from scartch
            self.trainer = SupervisedRunner(args=args,
                                            model=self.model,
                                            dataloader=self.load_data(task=args.task),
                                            device=self.device,
                                            loss_module=get_loss_module(args),
                                            optimizer=self.optimizer,
                                            l2_reg=0)

    def train(self):

        start_time = time.time()

        self.trainer.train(max_local_epoch=self.local_epochs)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def evaluate_client(self):

        ood_metrics = self.trainer.test()

        return ood_metrics


