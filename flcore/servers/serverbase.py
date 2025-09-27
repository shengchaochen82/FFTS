import torch
import os
import numpy as np
import h5py 
import copy
import time
import random

from data_provider.load_utils import read_client_data_monash


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 20

        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # self.uploaded_residual_proto = []
        self.uploaded_seasonal_proto = []
        self.uploaded_trend_proto = []
        
        self.rs_test_mae = []
        self.rs_test_rmse = []
        self.rs_test_mape = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.new_clients = [] # for out-of-distribution datasets and fine-tunning

        self.setting = 'note{}_task{}_al{}_len{}_data{}_maskr{}_maslen{}_ratio{}_k{}'.format(
            args.task_note,
            args.task,
            args.algorithm,
            args.max_seq_len,
            args.dataset,
            args.masking_ratio,
            args.mean_mask_length,
            args.join_ratio,
            args.topk_value)
        
        self.early_stopping_patience = 5
        self.test_results = []
        self.best_mse = float('inf')
        self.best_round = 0
        self.no_improvement_rounds = 0

    def set_clients(self, args, clientObj):
        for i in range(1, self.num_clients + 1):

            train_data, dataset_name = read_client_data_monash(args, i, is_train=True)
            test_data, dataset_name = read_client_data_monash(args, i, is_train=False)

                
            client = clientObj(args, 
                            id=i, 
                            data_name=dataset_name,
                            train_data=train_data,
                            test_data=test_data,
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            
            self.clients.append(client)


    def select_clients(self):
        num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, num_join_clients, replace=False))
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def send_prototypes(self):
        # customized server function used to send global trend/seasnal prototypes to clients
        pass

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(self.selected_clients, self.num_join_clients)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
            # print('ssss')

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        # print(self.uploaded_models)
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])

        # self.print_model_stats(self.global_model, 'Before aggregation')

        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

        # self.print_model_stats(self.global_model, "After Aggregation")

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("weights", self.dataset, self.setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("weights", self.dataset, self.setting)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("weights", self.dataset, self.setting)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_mae)):
            algo = algo + "_" + self.setting + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def set_new_clients(self, args, clientObj):
        """
            out-of-distribution generalization
        """
        for i in range(self.num_clients + 4, self.num_clients + 5):

            train_data, dataset_name = read_client_data_monash(args, i, is_train=True)
            test_data, dataset_name = read_client_data_monash(args, i, is_train=False)

            client = clientObj(args, 
                            id=i, 
                            data_name=dataset_name,
                            train_data=train_data,
                            test_data=test_data,
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            
            self.new_clients.append(client)

    def ood_evaluate_server(self):
        tot_mse = []

        eval_weight = copy.deepcopy(self.global_model)
        for c in self.new_clients:
            c.set_parameters(eval_weight)
            ood_metrics = c.evaluate_client()
            mse = ood_metrics['mse']
            tot_mse.append(mse)

        print("OOD Testing MSE: {:.3f}".format(np.average(tot_mse)))

    def regular_evaluate_server(self, current_round):
        tot_mse = []

        for c in self.clients:
            regular_metrics = c.evaluate_client()
            mse = regular_metrics['mse']
            tot_mse.append(mse)

        avg_mse = np.average(tot_mse)
        self.test_results.append(avg_mse)
        print(f"Round {current_round}: Regular Testing MSE: {avg_mse:.3f}")

        if avg_mse < self.best_mse:
            self.best_mse = avg_mse
            self.best_round = current_round
            self.no_improvement_rounds = 0

            # save the best model (temp)
            model_path = os.path.join("weights", self.dataset, self.setting)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            model_save_path = os.path.join(model_path, f'best_model_round{current_round}.pt')

            torch.save(self.global_model, model_save_path)
        else:
            self.no_improvement_rounds += 1

        if self.no_improvement_rounds >= self.early_stopping_patience:
            print(f"Early stopping at round {current_round}. Best MSE: {self.best_mse:.3f} at round {self.best_round}")
            return True
        
        return False

    def finetune_new_client(self):
        pass

    def print_model_stats(self, model, description="Model Statistics"):
        for name, param in model.named_parameters():
            print(f"{description} - {name}: mean={param.data.mean()}, std={param.data.std()}")

    def flatten_parameters(self):
        # flatten the parameters from uploaded model
        params = [p.data.cpu().numpy() for p in self.model.parameters()]
        shapes = [p.shape for p in self.model.parameters()]
        flat_params = np.concatenate([p.flatten() for p in params])
        # return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])
        return flat_params, shapes

    def recon_parameters(self, flat_params, shapes, model_template):
        # reconstructe the model shape from the aggregated model
        offset = 0
        new_params = []

        for shape in shapes:
            size = np.prod(shape)
            param = flat_params[offset:offset + size].reshape(shape)
            new_params.append(param)
            offset += size

        with torch.no_grad():
            for p, new_p in zip(model_template.parameters(), new_params):
                p.copy_(torch.from_numpy(new_p))
        
        return model_template

            

