import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from data_provider.data_base import DataSplits, TaskDataset, TimeseriesData
from data_provider.monash_data import convert_tsf_to_dataframe
from utils.sup_tools import noise_mask
from sklearn.model_selection import train_test_split
from data_provider.utils import upsample_timeseries, downsample_timeseries

class ShortForecastingDataset(TaskDataset):
    def __init__(self, 
                 seq_len : int = 512,
                 full_file_path_and_name: str = '../TimeseriesDatasets/forecasting/monash/dominick_dataset.tsf',
                 data_split : str = 'train',
                 scale : bool = True,
                 task_name : str = 'short-horizon-forecasting',
                 train_ratio : float = 0.6,
                 val_ratio : float = 0.1,
                 test_ratio : float = 0.3,
                 random_seed : int = 42,
                 masking_config : dict = None,
                 upsampling_pad_direction = "backward",
                 upsampling_type = "pad",
                 downsampling_type = "last",
                 pad_mode = "constant",
                 pad_constant_values = 0,
                 **kwargs
                 ):
        super(ShortForecastingDataset, self).__init__()
        """
        Parameters (mainly on Monash Time Series Repo)
        ----------
        seq_len : int
            Length of the input sequence.
        full_file_path_and_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        scale : bool
            Whether to scale the dataset.
        task_name : str
            The task that the dataset is used for. One of
            'short-horizon-forecasting', 'pre-training', or 'imputation'.
        train_ratio : float
            Ratio of the training set.
        val_ratio : float
            Ratio of the validation set.
        test_ratio : float
            Ratio of the test set.
        random_seed : int
            Random seed for reproducibility.
        """
    
        self.seq_len = seq_len
        self.full_file_path_and_name = full_file_path_and_name

        self.dataset_name = full_file_path_and_name.split('/')[-1][:-4]
        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.masking_config = masking_config

        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values

        self.n_channels = 1 # All these time-series are univariate

        self.window_size = self.seq_len # Fix the length of windows size
        self.step_size = 3


        # Input checking
        self._check_inputs()
        
        # Read data
        self._read_data()

        # Sliding
        self.sliding_windows = self._create_sliding_windows()

    def _check_inputs(self):
        assert self.data_split in ['train', 'test', 'val'],\
            "data_split must be one of 'train', 'test' or 'val'"
        assert self.task_name in ['short-horizon-forecasting', 'pre-training', 'imputation'],\
            "task_name must be one of 'short-horizon-forecasting', 'pre-training', or 'imputation'"
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1,\
            "train_ratio + val_ratio + test_ratio must be equal to 1"
                    
    def __repr__(self):
        repr = f"ShortForecastingDataset(dataset_name={self.dataset_name}," + \
            f"length_dataset={self.__len__()}," + \
            f"seq_len={self.seq_len}," + \
            f"forecast_horizon={self.forecast_horizon}," + \
            f"data_split={self.data_split}," + \
            f"scale={self.scale}," + \
            f"task_name={self.task_name}," + \
            f"n_channels={self.n_channels}," + \
            f"train_ratio={self.train_ratio}," + \
            f"val_ratio={self.val_ratio}," + \
            f"test_ratio={self.test_ratio},"
        return repr

    def _get_borders(self):
        n_train = int(self.train_ratio * self.length_dataset)
        n_test = int(self.test_ratio*self.length_dataset)
        n_val = self.length_dataset - n_train - n_test

        train_end = n_train
        val_start = train_end 
        val_end = val_start + n_val
        test_start = val_end 
            
        return DataSplits(train=slice(0, train_end), 
                          val=slice(val_start, val_end), 
                          test=slice(test_start, None))

    def _read_data(self):
        if self.full_file_path_and_name.endswith('.tsf'):
            df, frequency, forecast_horizon,\
                contain_missing_values, contain_equal_length =\
                    convert_tsf_to_dataframe(
                        self.full_file_path_and_name, 
                        replace_missing_vals_with="NaN", 
                        value_column_name="series_value")

        elif self.full_file_path_and_name.endswith('.npy'):
            frequency = None
            forecast_horizon = None
            contain_missing_values = False
            contain_equal_length = False

            HORIZON_MAPPING = {
                'hourly': 48,
                'daily': 14,
                'weekly': 13,
                'monthly': 18,
                'quarterly': 8,
                'yearly': 6
            }
            for f in HORIZON_MAPPING:
                if f in self.full_file_path_and_name.lower():
                    frequency = f
                    forecast_horizon = HORIZON_MAPPING[f]
                    break
            if frequency is None:
                raise ValueError(
                    "Frequency not found in filename: {}".format(
                        self.full_file_path_and_name
                ))

            data = np.load(self.full_file_path_and_name, allow_pickle=True)
            data = data[()] # unpack the dictionary
            data = [(_id, series) for _id, series in data.items()]
            df = pd.DataFrame(data, columns=['series_name', 'series_value'])

        else:
            raise ValueError(f'Unknown file type: {self.full_file_path_and_name}')


        self.forecast_horizon = forecast_horizon
        # NOTE: What should we do if the forecast_horizon
        if self.forecast_horizon is None: self.forecast_horizon = 8
        assert self.forecast_horizon > 0, "forecast_horizon must be greater than 0"

        self.length_dataset = df.shape[0] 

        # Following line shuffles the dataset
        df = df.sample(frac=1, 
                random_state=self.random_seed).reset_index(drop=True)
        data_splits = self._get_borders()

        if self.scale: # z-score
            df.series_value = df.series_value.apply(lambda i: (i - i.mean())/(i.std(ddof=0) + 1e-7))    

        # if self.scale: # min-max
        #     df.series_value = df.series_value.apply(lambda i: (i - i.min()) / (i.max() - i.min() + 1e-7))       
                
        if len(df.iloc[:]) == 1: # only have one time series can not split
            self.data = df.iloc[:]

        else:
            if self.data_split == 'train':
                self.data = df.iloc[data_splits.train, :]
            elif self.data_split == 'val':
                self.data = df.iloc[data_splits.val, :]
            elif self.data_split == 'test':
                self.data = df.iloc[data_splits.test, :]        

        self.length_dataset = self.data.shape[0]

    
    def __len__(self):
        return len(self.sliding_windows)
    
    def __getitem__(self, index):
        timeseries = self.sliding_windows[index]

        assert timeseries.ndim == 1, "Time-series is not univariate"
        timeseries = timeseries[:, np.newaxis]
        
        if self.data_split == 'train':
            mask = noise_mask(timeseries, 
                            self.masking_config['masking_ratio'],
                            self.masking_config['mean_mask_length'],
                            self.masking_config['mode'],
                            self.masking_config['distribution'])
        else:
            # more serious val environment
            val_masking_ratio=0.75
            val_masking_length=256

            mask = noise_mask(timeseries, 
                            val_masking_ratio,
                            val_masking_length,
                            self.masking_config['mode'],
                            self.masking_config['distribution'])
    
        return torch.from_numpy(timeseries), torch.from_numpy(mask)
    
    def _create_sliding_windows(self):
        all_windows = []

        for index in range(len(self.data)):
            timeseries = np.asarray(self.data.iloc[index, :].series_value)
            assert timeseries.ndim == 1, "Time-series is not univariate"

            timeseries_len = len(timeseries)

            if timeseries_len <= self.seq_len:
                timeseries, _ =\
                    upsample_timeseries(timeseries,
                                        self.seq_len,
                                        direction=self.upsampling_pad_direction,
                                        sampling_type=self.upsampling_type,
                                        mode=self.pad_mode)
            elif timeseries_len > self.seq_len:

                timeseries = self.sliding_window(timeseries, self.window_size, self.step_size)
            all_windows.append(timeseries)
            
        return np.vstack(all_windows)

    def sliding_window(self, sequence, window_size, step_size=1):
        return np.array([sequence[i:i + window_size] 
                        for i in range(0, len(sequence) - window_size + 1, step_size)])
    

class ShortForecastingDataset_Pretrain(TaskDataset):
    def __init__(self, 
                 seq_len : int = 512,
                 full_file_path_and_name: str = '../TimeseriesDatasets/forecasting/monash/dominick_dataset.tsf',
                 data_split : str = 'train',
                 scale : bool = True,
                 task_name : str = 'short-horizon-forecasting',
                 train_ratio : float = 0.6,
                 val_ratio : float = 0.1,
                 test_ratio : float = 0.3,
                 random_seed : int = 42,
                 masking_config : dict = None,
                 upsampling_pad_direction = "backward",
                 upsampling_type = "pad",
                 downsampling_type = "last",
                 pad_mode = "constant",
                 pad_constant_values = 0,
                 step_size = 512,
                 **kwargs
                 ):
        super(ShortForecastingDataset_Pretrain, self).__init__()
        """
        Parameters (mainly on Monash Time Series Repo)
        ----------
        seq_len : int
            Length of the input sequence.
        full_file_path_and_name : str
            Name of the dataset.
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        scale : bool
            Whether to scale the dataset.
        task_name : str
            The task that the dataset is used for. One of
            'short-horizon-forecasting', 'pre-training', or 'imputation'.
        train_ratio : float
            Ratio of the training set.
        val_ratio : float
            Ratio of the validation set.
        test_ratio : float
            Ratio of the test set.
        random_seed : int
            Random seed for reproducibility.
        """
    
        self.seq_len = seq_len
        self.full_file_path_and_name = full_file_path_and_name

        self.dataset_name = full_file_path_and_name.split('/')[-1][:-4]
        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.masking_config = masking_config

        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values

        self.n_channels = 1 # All these time-series are univariate

        self.window_size = self.seq_len # Fix the length of windows size
        self.step_size = step_size

        # Input checking
        self._check_inputs()
        
        # Read data
        self._read_data()

        # Sliding
        self.sliding_windows = self._create_sliding_windows()

    def _check_inputs(self):
        assert self.data_split in ['train', 'test', 'val'],\
            "data_split must be one of 'train', 'test' or 'val'"
        assert self.task_name in ['short-horizon-forecasting', 'pre-training', 'imputation'],\
            "task_name must be one of 'short-horizon-forecasting', 'pre-training', or 'imputation'"
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1,\
            "train_ratio + val_ratio + test_ratio must be equal to 1"
                    
    def __repr__(self):
        repr = f"ShortForecastingDataset(dataset_name={self.dataset_name}," + \
            f"length_dataset={self.__len__()}," + \
            f"seq_len={self.seq_len}," + \
            f"forecast_horizon={self.forecast_horizon}," + \
            f"data_split={self.data_split}," + \
            f"scale={self.scale}," + \
            f"task_name={self.task_name}," + \
            f"n_channels={self.n_channels}," + \
            f"train_ratio={self.train_ratio}," + \
            f"val_ratio={self.val_ratio}," + \
            f"test_ratio={self.test_ratio},"
        return repr

    def _get_borders(self):
        n_train = int(self.train_ratio * self.length_dataset)
        n_test = int(self.test_ratio*self.length_dataset)
        n_val = self.length_dataset - n_train - n_test

        train_end = n_train
        val_start = train_end 
        val_end = val_start + n_val
        test_start = val_end 
            
        return DataSplits(train=slice(0, train_end), 
                          val=slice(val_start, val_end), 
                          test=slice(test_start, None))

    def _read_data(self):
        if self.full_file_path_and_name.endswith('.tsf'):
            df, frequency, forecast_horizon,\
                contain_missing_values, contain_equal_length =\
                    convert_tsf_to_dataframe(
                        self.full_file_path_and_name, 
                        replace_missing_vals_with="NaN", 
                        value_column_name="series_value")

        elif self.full_file_path_and_name.endswith('.npy'):
            frequency = None
            forecast_horizon = None
            contain_missing_values = False
            contain_equal_length = False

            HORIZON_MAPPING = {
                'hourly': 48,
                'daily': 14,
                'weekly': 13,
                'monthly': 18,
                'quarterly': 8,
                'yearly': 6
            }
            for f in HORIZON_MAPPING:
                if f in self.full_file_path_and_name.lower():
                    frequency = f
                    forecast_horizon = HORIZON_MAPPING[f]
                    break
            if frequency is None:
                raise ValueError(
                    "Frequency not found in filename: {}".format(
                        self.full_file_path_and_name
                ))

            data = np.load(self.full_file_path_and_name, allow_pickle=True)
            data = data[()] # unpack the dictionary
            data = [(_id, series) for _id, series in data.items()]
            df = pd.DataFrame(data, columns=['series_name', 'series_value'])

        else:
            raise ValueError(f'Unknown file type: {self.full_file_path_and_name}')


        self.forecast_horizon = forecast_horizon
        # NOTE: What should we do if the forecast_horizon
        if self.forecast_horizon is None: self.forecast_horizon = 8
        assert self.forecast_horizon > 0, "forecast_horizon must be greater than 0"

        self.length_dataset = df.shape[0] 

        # Following line shuffles the dataset
        df = df.sample(frac=1, 
                random_state=self.random_seed).reset_index(drop=True)
        data_splits = self._get_borders()

        if self.scale: # z-score
            df.series_value = df.series_value.apply(lambda i: (i - i.mean())/(i.std(ddof=0) + 1e-7))    
  
                
        if len(df.iloc[:]) == 1: # only have one time series can not split
            self.data = df.iloc[:]

        else:
            if self.data_split == 'train':
                self.data = df.iloc[data_splits.train, :]
            elif self.data_split == 'val':
                self.data = df.iloc[data_splits.val, :]
            elif self.data_split == 'test':
                self.data = df.iloc[data_splits.test, :]        

        self.length_dataset = self.data.shape[0]

    
    def __len__(self):
        return len(self.sliding_windows)
    
    def __getitem__(self, index):
        timeseries = self.sliding_windows[index]

        assert timeseries.ndim == 1, "Time-series is not univariate"
        timeseries = timeseries[:, np.newaxis]
        
        if self.data_split == 'train':
            mask = noise_mask(timeseries, 
                            self.masking_config['masking_ratio'],
                            self.masking_config['mean_mask_length'],
                            self.masking_config['mode'],
                            self.masking_config['distribution'])
        else:
            # more serious val environment
            val_masking_ratio=0.35
            val_masking_length=16

            mask = noise_mask(timeseries, 
                            val_masking_ratio,
                            val_masking_length,
                            self.masking_config['mode'],
                            self.masking_config['distribution'])
    
        return torch.from_numpy(timeseries), torch.from_numpy(mask)
    
    def _create_sliding_windows(self):
        all_windows = []

        for index in range(len(self.data)):
            timeseries = np.asarray(self.data.iloc[index, :].series_value)
            assert timeseries.ndim == 1, "Time-series is not univariate"

            timeseries_len = len(timeseries)

            if timeseries_len <= self.seq_len:
                timeseries, _ =\
                    upsample_timeseries(timeseries,
                                        self.seq_len,
                                        direction=self.upsampling_pad_direction,
                                        sampling_type=self.upsampling_type,
                                        mode=self.pad_mode)
            elif timeseries_len > self.seq_len:
                timeseries = self.sliding_window(timeseries, self.window_size, self.step_size)
            all_windows.append(timeseries)
            
        return np.vstack(all_windows)

    def plot(self, idx):
        timeseries_data = self.__getitem__(idx)
        forecast = timeseries_data.forecast
        timeseries = timeseries_data.timeseries

        plt.title(f"idx={idx}", fontsize=18)
        plt.plot(np.arange(self.seq_len), timeseries.squeeze(), 
                 label='Time-series', c='darkblue')
        if self.task_name == 'short-horizon-forecasting':
            plt.plot(np.arange(timeseries.shape[-1], timeseries.shape[-1] + self.forecast_horizon), 
                     forecast.squeeze(), label='Forecast', c='red', linestyle='--')
        
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Value', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()

    def sliding_window(self, sequence, window_size, step_size=1):
        return np.array([sequence[i:i + window_size] 
                        for i in range(0, len(sequence) - window_size + 1, step_size)])