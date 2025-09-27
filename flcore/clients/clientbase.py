import copy
import torch
import os
from torch.utils.data import DataLoader
from utils.sup_tools import collate_unsuperv

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, data_name, train_data, test_data, **kwargs):
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_data, self.test_data = train_data, test_data
        self.train_samples, self.test_samples = len(train_data), len(test_data)
        
        print("\033[1m" + "Basic Information" + "\033[0m")
        print(f'  {"Client Number:":<20}{self.id:<20}')
        print(f'  {"Data:":<20}{data_name:<20}')
        print(f'  {"Data Samples (train/test):":<20}{self.train_samples:<10}{self.test_samples:<10}')
        print()

        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.local_epochs = args.local_epochs

        self.max_len = args.max_seq_len

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

    def load_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        train_dataloder = DataLoader(self.train_data, batch_size, 
                                        shuffle=False, pin_memory=True,
                                        collate_fn=lambda x: collate_unsuperv(x, max_len=self.max_len),
                                        drop_last=False)
            
        test_dataloader = DataLoader(self.test_data, batch_size, 
                                        shuffle=False, pin_memory=True,
                                        collate_fn=lambda x: collate_unsuperv(x, max_len=self.max_len),
                                         drop_last=False)

        return train_dataloder, test_dataloader

            
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
    

