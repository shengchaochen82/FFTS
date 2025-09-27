import os
import json
import torch
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# customized package
from data_provider.pre_loader import ShortForecastingDataset
from data_provider.pre_loader import ShortForecastingDataset_Pretrain
from data_provider.pre_loader_m4 import ShortForecastingDataset_M4FL, M4_loader


def load_filenames_from_json(input_file):
    with open(input_file, 'r') as json_file:
        filenames_dict = json.load(json_file)
    return filenames_dict

    
def read_client_data_monash(args, idx, is_train=True):
    """
    : The function is designed to read and load dataset from each client
    : 2-dataset for Monash Time Series
    """
    monash_ts_dict = load_filenames_from_json(os.path.join('dataset', args.dataset + '.json'))

    data_dir = os.path.join('dataset', args.dataset)

    if args.task == 'pretrain':
        data_path = os.path.join(data_dir, monash_ts_dict[str(idx)])

    elif args.task == 'pretrain_long':
        # selected dataset for pretraining
        monash_ts_dict = load_filenames_from_json(os.path.join('dataset', 'pretrain.json'))
        data_path = os.path.join(data_dir, monash_ts_dict[str(idx)])

    dataset_name = f"MonashTS-{monash_ts_dict[str(idx)]}" if args.task in ['pretrain', 'pretrain_long'] else 'm4'

    pretrain_config = {'masking_ratio': args.masking_ratio,
                        'mean_mask_length': args.mean_mask_length,
                        'mode': args.mask_mode,
                        'distribution': args.mask_distribution,
                        'exclude_feats': None}

    if args.task == 'pretrain':
        if is_train:
            data = ShortForecastingDataset(
                seq_len=args.seq_len,
                full_file_path_and_name=data_path,
                data_split='train',
                scale=True,
                task_name='pre-training',
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3,
                random_seed=42,
                masking_config=pretrain_config
            )

        else:
            data = ShortForecastingDataset(
                seq_len=args.seq_len,
                full_file_path_and_name=data_path,
                data_split='val',
                task_name='pre-training',
                scale=True,
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3,
                random_seed=42,
                masking_config=pretrain_config
            )

            
    elif args.task == 'pretrain_m4_full':
        data_path = m4_collection[idx-1]
        if is_train:
            data = ShortForecastingDataset(
                seq_len=args.seq_len,
                full_file_path_and_name=data_path,
                data_split='train',
                scale=True,
                task_name='pre-training',
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3,
                random_seed=42,
                masking_config=pretrain_config
            )

        else:
            data = ShortForecastingDataset(
                seq_len=args.seq_len,
                full_file_path_and_name=data_path,
                data_split='val',
                task_name='pre-training',
                scale=True,
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3,
                random_seed=42,
                masking_config=pretrain_config
            )

    elif args.task == 'pretrain_long':
        if is_train:
            data = ShortForecastingDataset_Pretrain(
                seq_len=args.seq_len,
                full_file_path_and_name=data_path,
                data_split='train',
                task_name='pre-training',
                scale=True,
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3,
                random_seed=42,
                masking_config=pretrain_config,
                step_size=args.step_size
            )

        else:
            data = ShortForecastingDataset_Pretrain(
                seq_len=args.seq_len,
                full_file_path_and_name=data_path,
                data_split='val',
                task_name='pre-training',
                scale=True,
                train_ratio=0.6,
                val_ratio=0.1,
                test_ratio=0.3,
                random_seed=42,
                masking_config=pretrain_config,
                step_size=args.step_size
            )


    return data, dataset_name

class CombinedDataset(Dataset):
    def __init__(self, data, masks):
        self.data = data
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.masks[idx])


