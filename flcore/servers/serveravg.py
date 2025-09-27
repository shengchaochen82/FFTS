import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(args, clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("\nFinished creating server and clients.")

        self.Budget = []
        # self.set_new_clients(args, clientAVG)
        print("\nFinished creating new clients for ood testing.")

    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i + 1}-------------")
                print("\nEvaluate global model")
                # self.ood_evaluate_server()
                stop = self.regular_evaluate_server(current_round=i)
                if stop:
                    break

            for client in self.selected_clients:
                print("\033[1m" + f"Train {client.id}-th Client" + "\033[0m")
                client.train()

            self.receive_models()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)

        print(f"Training completed. Best MSE: {self.best_mse:.3f} at round {self.best_round}")

        self.save_results()
        self.save_global_model()



