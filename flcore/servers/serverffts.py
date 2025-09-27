import time
import copy
import torch
import numpy as np
from collections import defaultdict
from flcore.clients.clientffts import clientFFTS
from flcore.servers.serverbase import Server
from threading import Thread


class FedFFTS(Server):
    """FFTS Server with Adaptive Trend Module aggregation and alignment"""
    
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # FFTS-specific parameters
        self.lambda_atm = args.lambda_atm if hasattr(args, 'lambda_atm') else 0.01
        self.atm_aggregation_weight = args.atm_aggregation_weight if hasattr(args, 'atm_aggregation_weight') else 0.5
        self.enable_atm_alignment = args.enable_atm_alignment if hasattr(args, 'enable_atm_alignment') else True
        
        # Initialize clients
        self.set_clients(args, clientFFTS)
        
        # ATM-specific storage
        self.global_atm_params = None
        self.uploaded_atm_params = []
        self.atm_aggregation_history = []
        
        print(f"\n FFTS Server initialized:")
        print(f"   Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"   ATM regularization weight: {self.lambda_atm}")
        print(f"   ATM aggregation weight: {self.atm_aggregation_weight}")
        print(f"   ATM alignment: {'Enabled' if self.enable_atm_alignment else 'Disabled'}")
        print("Finished creating FFTS server and clients.")
        
        # Initialize global ATM parameters
        self._initialize_global_atm_params()
        
        # Performance tracking
        self.round_metrics = []
        self.atm_alignment_history = []

    def _initialize_global_atm_params(self):
        """Initialize global ATM parameters from the global model"""
        if hasattr(self.global_model, 'get_atm_parameters'):
            self.global_atm_params = [param.clone().detach() 
                                    for param in self.global_model.get_atm_parameters()]
            print(f"   Initialized {len(self.global_atm_params)} global ATM parameters")

    def train(self):
        """Main training loop for FFTS"""
        print(f"\n Starting FFTS training for {self.global_rounds} rounds...")
        
        for round_idx in range(self.global_rounds):
            round_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"FFTS Round {round_idx + 1}/{self.global_rounds}")
            print(f"{'='*60}")
            
            # Client selection
            self.selected_clients = self.select_clients()
            print(f"Selected {len(self.selected_clients)} clients: "
                  f"{[c.id for c in self.selected_clients]}")
            
            # Send global model and ATM parameters
            self.send_models()
            if self.enable_atm_alignment:
                self.send_atm_parameters()
            
            # Evaluation before training
            if round_idx % self.eval_gap == 0:
                print(f"\n Pre-training evaluation (Round {round_idx + 1})")
                stop = self.regular_evaluate_server(current_round=round_idx)
                if stop:
                    print(" Early stopping triggered!")
                    break
            
            # Client training
            print(f"\n Client training phase:")
            client_results = []
            for client in self.selected_clients:
                print(f"\n   Training Client {client.id}...")
                client_result = client.train()
                client_results.append(client_result)
            
            # Collect models and ATM parameters
            self.receive_models()
            if self.enable_atm_alignment:
                self.receive_atm_parameters()
            
            # Aggregation
            print(f"\n Aggregating models and ATM parameters...")
            self.aggregate_parameters()
            if self.enable_atm_alignment:
                self.aggregate_atm_parameters()
            
            # Round statistics
            round_time = time.time() - round_start_time
            self.Budget.append(round_time)
            
            avg_client_loss = np.mean([r['avg_loss'] for r in client_results])
            print(f"   Round {round_idx + 1} completed in {round_time:.2f}s")
            print(f"   Average client loss: {avg_client_loss:.4f}")
            
            # Store round metrics
            self.round_metrics.append({
                'round': round_idx + 1,
                'avg_client_loss': avg_client_loss,
                'round_time': round_time,
                'num_clients': len(self.selected_clients)
            })
        
        print(f"\n FFTS training completed!")
        print(f"   Best MSE: {self.best_mse:.4f} at round {self.best_round}")
        print(f"   Total training time: {sum(self.Budget):.2f}s")
        
        # Save results
        self.save_results()
        self.save_global_model()
        self.save_ffts_specific_results()

    def send_atm_parameters(self):
        """Send global ATM parameters to selected clients"""
        if self.global_atm_params is None:
            return
        
        print(f" Sending global ATM parameters to clients...")
        for client in self.selected_clients:
            client.set_global_atm_params(self.global_atm_params)

    def receive_atm_parameters(self):
        """Collect ATM parameters from clients"""
        print(f" Collecting ATM parameters from clients...")
        
        self.uploaded_atm_params = []
        for client in self.selected_clients:
            client_atm_params = client.get_atm_parameters()
            self.uploaded_atm_params.append(client_atm_params)
        
        print(f"   Collected ATM parameters from {len(self.uploaded_atm_params)} clients")

    def aggregate_atm_parameters(self):
        """Aggregate ATM parameters using weighted averaging"""
        if not self.uploaded_atm_params or not self.enable_atm_alignment:
            return
        
        print(f" Aggregating ATM parameters...")
        
        # Initialize aggregated ATM parameters
        aggregated_atm_params = []
        
        # Get the structure from the first client
        first_client_atm = self.uploaded_atm_params[0]
        
        for param_idx, param in enumerate(first_client_atm):
            # Initialize with zeros
            aggregated_param = torch.zeros_like(param)
            
            # Weighted aggregation
            total_weight = 0.0
            for client_idx, client_atm_params in enumerate(self.uploaded_atm_params):
                weight = self.uploaded_weights[client_idx]
                aggregated_param += weight * client_atm_params[param_idx].data
                total_weight += weight
            
            # Normalize
            if total_weight > 0:
                aggregated_param /= total_weight
            
            aggregated_atm_params.append(aggregated_param)
        
        # Update global ATM parameters with momentum
        if self.global_atm_params is not None:
            for global_param, aggregated_param in zip(self.global_atm_params, aggregated_atm_params):
                global_param.data = (1 - self.atm_aggregation_weight) * global_param.data + \
                                  self.atm_aggregation_weight * aggregated_param.data
        else:
            self.global_atm_params = aggregated_atm_params
        
        # Update global model's ATM parameters
        self._update_global_model_atm_params()
        
        # Track ATM alignment
        self._track_atm_alignment()
        
        print(f" ATM parameters aggregated successfully")

    def _update_global_model_atm_params(self):
        """Update the global model's ATM parameters"""
        if self.global_atm_params is None:
            return
        
        global_model_atm_params = self.global_model.get_atm_parameters()
        for global_param, new_param in zip(global_model_atm_params, self.global_atm_params):
            global_param.data = new_param.data.clone()

    def _track_atm_alignment(self):
        """Track ATM parameter alignment across clients"""
        if not self.uploaded_atm_params:
            return
        
        alignment_scores = []
        
        # Compute pairwise alignment between clients
        for i in range(len(self.uploaded_atm_params)):
            for j in range(i + 1, len(self.uploaded_atm_params)):
                client_i_params = torch.cat([p.flatten() for p in self.uploaded_atm_params[i]])
                client_j_params = torch.cat([p.flatten() for p in self.uploaded_atm_params[j]])
                
                cosine_sim = torch.nn.functional.cosine_similarity(
                    client_i_params.unsqueeze(0), client_j_params.unsqueeze(0))
                alignment_scores.append(cosine_sim.item())
        
        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
        self.atm_alignment_history.append(avg_alignment)
        
        print(f" ATM alignment score: {avg_alignment:.4f}")

    def aggregate_parameters(self):
        """Enhanced parameter aggregation for FFTS"""
        print(f" Aggregating global model parameters...")
        
        # Standard FedAvg aggregation
        super().aggregate_parameters()
        
        print(f" Global model parameters aggregated")

    def regular_evaluate_server(self, current_round):
        """Enhanced evaluation with FFTS-specific metrics"""
        print(f" Evaluating global model...")
        
        tot_mse = []
        tot_atm_alignment = []
        
        for client in self.clients:
            metrics = client.evaluate_client()
            tot_mse.append(metrics['mse'])
            
            if 'atm_alignment' in metrics:
                tot_atm_alignment.append(metrics['atm_alignment'])
        
        avg_mse = np.average(tot_mse)
        avg_atm_alignment = np.average(tot_atm_alignment) if tot_atm_alignment else 0.0
        
        self.test_results.append(avg_mse)
        
        print(f" Round {current_round + 1} Results:")
        print(f"      MSE: {avg_mse:.4f}")
        print(f"      ATM Alignment: {avg_atm_alignment:.6f}")
        
        # Early stopping check
        if avg_mse < self.best_mse:
            self.best_mse = avg_mse
            self.best_round = current_round
            self.no_improvement_rounds = 0
            
            # Save best model
            self._save_best_model(current_round)
            print(f" New best MSE: {self.best_mse:.4f}")
        else:
            self.no_improvement_rounds += 1
            print(f" No improvement for {self.no_improvement_rounds} rounds")
        
        if self.no_improvement_rounds >= self.early_stopping_patience:
            print(f" Early stopping triggered at round {current_round + 1}")
            return True
        
        return False

    def _save_best_model(self, current_round):
        """Save the best model with FFTS-specific components"""
        import os
        
        model_path = os.path.join("weights", self.dataset, self.setting)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save global model
        model_save_path = os.path.join(model_path, f'best_ffts_model_round{current_round}.pt')
        torch.save(self.global_model, model_save_path)
        
        # Save ATM parameters separately
        if self.global_atm_params:
            atm_save_path = os.path.join(model_path, f'best_atm_params_round{current_round}.pt')
            torch.save(self.global_atm_params, atm_save_path)

    def save_ffts_specific_results(self):
        """Save FFTS-specific results and metrics"""
        import os
        import json
        
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        
        # Save round metrics
        metrics_file = os.path.join(result_path, f"ffts_metrics_{self.setting}_{self.times}.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'round_metrics': self.round_metrics,
                'atm_alignment_history': self.atm_alignment_history,
                'best_mse': float(self.best_mse),
                'best_round': int(self.best_round),
                'total_rounds': len(self.round_metrics),
                'settings': {
                    'lambda_atm': self.lambda_atm,
                    'atm_aggregation_weight': self.atm_aggregation_weight,
                    'enable_atm_alignment': self.enable_atm_alignment
                }
            }, f, indent=2)
        
        print(f"   💾 FFTS metrics saved to {metrics_file}")

    def get_server_stats(self):
        """Get comprehensive server statistics"""
        stats = {
            'total_rounds': len(self.round_metrics),
            'best_mse': float(self.best_mse),
            'best_round': int(self.best_round),
            'avg_round_time': np.mean(self.Budget) if self.Budget else 0.0,
            'total_training_time': sum(self.Budget),
            'num_clients': self.num_clients,
            'join_ratio': self.join_ratio
        }
        
        if self.atm_alignment_history:
            stats['avg_atm_alignment'] = np.mean(self.atm_alignment_history)
            stats['final_atm_alignment'] = self.atm_alignment_history[-1]
        
        return stats

    def print_final_summary(self):
        """Print comprehensive training summary"""
        print(f"\n{'='*80}")
        print(f"🎯 FFTS TRAINING SUMMARY")
        print(f"{'='*80}")
        
        stats = self.get_server_stats()
        
        print(f"Performance Metrics:")
        print(f"   Best MSE: {stats['best_mse']:.4f} (Round {stats['best_round']})")
        print(f"   Total Rounds: {stats['total_rounds']}")
        print(f"   Average Round Time: {stats['avg_round_time']:.2f}s")
        print(f"   Total Training Time: {stats['total_training_time']:.2f}s")
        
        if 'avg_atm_alignment' in stats:
            print(f"   Average ATM Alignment: {stats['avg_atm_alignment']:.4f}")
            print(f"   Final ATM Alignment: {stats['final_atm_alignment']:.4f}")
        
        print(f"\n Configuration:")
        print(f"   Clients: {stats['num_clients']} (Join ratio: {stats['join_ratio']})")
        print(f"   ATM Regularization: {self.lambda_atm}")
        print(f"   ATM Aggregation Weight: {self.atm_aggregation_weight}")
        print(f"   ATM Alignment: {'Enabled' if self.enable_atm_alignment else 'Disabled'}")
        
        print(f"\n{'='*80}")
