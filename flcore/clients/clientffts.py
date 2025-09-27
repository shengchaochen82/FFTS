
import torch
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.trainers.unsup_trainer import UnsupervisedRunner
from flcore.losses.loss import get_loss_module


class FFTSTrainer(UnsupervisedRunner):
    """FFTS-specific trainer with masked MSE and ATM alignment"""
    
    def __init__(self, args, model, dataloader, device, loss_module, optimizer, l2_reg=0):
        super().__init__(args, model, dataloader, device, loss_module, optimizer, l2_reg)
        self.lambda_atm = args.lambda_atm if hasattr(args, 'lambda_atm') else 0.01
        self.global_atm_params = None
        
    def set_global_atm_params(self, global_atm_params):
        """Set global ATM parameters for alignment regularization"""
        self.global_atm_params = global_atm_params
    
    def train_one_epoch(self):
        """Train one epoch with FFTS-specific objectives"""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_atm_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, masks) in enumerate(self.train_dataloader):
            data = data.float().to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, features = self.model(data, return_features=True)
            
            # Masked reconstruction loss
            recon_loss = self.compute_masked_loss(predictions, data, masks)
            
            # ATM alignment regularization
            atm_loss = self.compute_atm_alignment_loss()
            
            # Total loss
            total_batch_loss = recon_loss + self.lambda_atm * atm_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_atm_loss += atm_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_atm_loss = total_atm_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'atm_loss': avg_atm_loss
        }
    
    def compute_masked_loss(self, predictions, targets, masks):
        """Compute masked MSE loss"""
        # Apply mask: 1 for masked positions, 0 for observed
        masked_predictions = predictions * masks
        masked_targets = targets * masks
        
        # Compute MSE only on masked positions
        mse_loss = F.mse_loss(masked_predictions, masked_targets, reduction='none')
        
        # Average over masked positions
        num_masked = masks.sum()
        if num_masked > 0:
            masked_loss = mse_loss.sum() / num_masked
        else:
            masked_loss = torch.tensor(0.0, device=predictions.device)
        
        return masked_loss
    
    def compute_atm_alignment_loss(self):
        """Compute ATM alignment regularization loss"""
        if self.global_atm_params is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        client_atm_params = self.model.get_atm_parameters()
        alignment_loss = self.model.compute_atm_alignment_loss(
            client_atm_params, self.global_atm_params)
        
        return alignment_loss


class clientFFTS(Client):
    """FFTS Client with Adaptive Trend Module and masked training"""
    
    def __init__(self, args, id, data_name, train_data, test_data, **kwargs):
        super().__init__(args, id, data_name, train_data, test_data, **kwargs)
        
        # FFTS-specific parameters
        self.lambda_atm = args.lambda_atm if hasattr(args, 'lambda_atm') else 0.01
        self.masking_ratio = args.masking_ratio if hasattr(args, 'masking_ratio') else 0.15
        self.mean_mask_length = args.mean_mask_length if hasattr(args, 'mean_mask_length') else 3
        
        # Initialize FFTS trainer
        if 'pretrain' in args.task:
            self.trainer = FFTSTrainer(
                args=args,
                model=self.model,
                dataloader=self.load_data(task=args.task),
                device=self.device,
                loss_module=get_loss_module(args),
                optimizer=self.optimizer,
                l2_reg=0
            )

        
        # Store global ATM parameters for alignment
        self.global_atm_params = None
        
        print(f"FFTS Client {self.id} initialized with:")
        print(f"  - ATM regularization weight: {self.lambda_atm}")
        print(f"  - Masking ratio: {self.masking_ratio}")
        print(f"  - Mean mask length: {self.mean_mask_length}")

    def set_global_atm_params(self, global_atm_params):
        """Receive global ATM parameters from server"""
        self.global_atm_params = [param.clone().detach() for param in global_atm_params]
        
        # Pass to trainer if it's FFTS trainer
        if isinstance(self.trainer, FFTSTrainer):
            self.trainer.set_global_atm_params(self.global_atm_params)

    def get_atm_parameters(self):
        """Get ATM-specific parameters for server aggregation"""
        return self.model.get_atm_parameters()

    def train(self):
        """Train the local model with FFTS objectives"""
        start_time = time.time()
        
        print(f"\n Training FFTS Client {self.id}")
        print(f"   Local epochs: {self.local_epochs}")
        print(f"   ATM alignment: {'Enabled' if self.global_atm_params else 'Disabled'}")
        
        # Train for specified local epochs
        epoch_losses = []
        for epoch in range(self.local_epochs):
            if isinstance(self.trainer, FFTSTrainer):
                # FFTS-specific training
                epoch_metrics = self.trainer.train_one_epoch()
                epoch_losses.append(epoch_metrics['total_loss'])
                
                if epoch % max(1, self.local_epochs // 3) == 0:
                    print(f"   Epoch {epoch+1}/{self.local_epochs}: "
                          f"Loss={epoch_metrics['total_loss']:.4f} "
                          f"(Recon={epoch_metrics['recon_loss']:.4f}, "
                          f"ATM={epoch_metrics['atm_loss']:.4f})")
            else:
                # Standard training
                self.trainer.train(max_local_epoch=1)
                
        # Learning rate decay
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        
        # Update training statistics
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f" Client {self.id} training completed. Average loss: {avg_loss:.4f}")
        
        return {
            'avg_loss': avg_loss,
            'num_samples': self.train_samples
        }

    def evaluate_client(self):
        """Evaluate the local model"""
        print(f"🔍 Evaluating FFTS Client {self.id}")
        
        # Use standard evaluation
        metrics = self.trainer.test()
        
        # Add FFTS-specific metrics if available
        if hasattr(self.trainer, 'compute_atm_alignment_loss'):
            with torch.no_grad():
                atm_alignment = self.trainer.compute_atm_alignment_loss().item()
                metrics['atm_alignment'] = atm_alignment
        
        print(f"   Evaluation MSE: {metrics.get('mse', 'N/A'):.4f}")
        print(f"   ATM Alignment: {metrics.get('atm_alignment', 'N/A'):.6f}")
        
        return metrics

    def compute_model_similarity(self, other_model):
        """Compute similarity between local and global model (for analysis)"""
        similarity_scores = {}
        
        with torch.no_grad():
            # Overall parameter similarity
            local_params = torch.cat([p.flatten() for p in self.model.parameters()])
            other_params = torch.cat([p.flatten() for p in other_model.parameters()])
            
            cosine_sim = F.cosine_similarity(local_params.unsqueeze(0), 
                                           other_params.unsqueeze(0))
            similarity_scores['overall'] = cosine_sim.item()
            
            # ATM-specific similarity
            local_atm_params = torch.cat([p.flatten() for p in self.get_atm_parameters()])
            other_atm_params = torch.cat([p.flatten() for p in other_model.get_atm_parameters()])
            
            atm_cosine_sim = F.cosine_similarity(local_atm_params.unsqueeze(0),
                                               other_atm_params.unsqueeze(0))
            similarity_scores['atm'] = atm_cosine_sim.item()
        
        return similarity_scores

    def get_model_stats(self):
        """Get model statistics for monitoring"""
        stats = {}
        
        with torch.no_grad():
            # Overall model stats
            all_params = [p for p in self.model.parameters()]
            total_params = sum(p.numel() for p in all_params)
            param_norm = torch.norm(torch.cat([p.flatten() for p in all_params]))
            
            stats['total_params'] = total_params
            stats['param_norm'] = param_norm.item()
            
            # ATM-specific stats
            atm_params = self.get_atm_parameters()
            atm_total_params = sum(p.numel() for p in atm_params)
            atm_param_norm = torch.norm(torch.cat([p.flatten() for p in atm_params]))
            
            stats['atm_params'] = atm_total_params
            stats['atm_param_norm'] = atm_param_norm.item()
            stats['atm_ratio'] = atm_total_params / total_params
        
        return stats
