from collections import OrderedDict
import torch
import numpy as np
from flcore.trainers.runner_base import BaseRunner
from flcore.losses.loss import l2_reg_loss
from utils.tools import EarlyStopping, adjust_learning_rate

class SupervisedRunner(BaseRunner):

    def train(self, max_local_epoch=None):

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        for epoch in range(max_local_epoch):
            train_loss = []
            epoch_loss = self.train_epoch(epoch_num=epoch)
            train_loss.append(epoch_loss)
            train_loss = np.average(train_loss)
            vali_loss = self.evaluate()

            print("\n Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):

            X, targets, padding_masks = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            predictions = self.model(X.to(self.device), padding_masks)

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch

        return epoch_loss
    
    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):

            X, targets, padding_masks = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            predictions = self.model(X.to(self.device), padding_masks)

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)  # mean loss (over samples)

            total_samples += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch

        # need to revise for adapting classification & regression evaluation
        return epoch_loss
    
    def test(self):
        
        ood_metrics = OrderedDict()
        self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        with torch.no_grad():
            for i, batch in enumerate(self.ood_test_loader):
 
                X, targets, target_masks, padding_masks = batch
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore

                predictions = self.model(X.to(self.device), padding_masks)

                # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
                target_masks = target_masks * padding_masks.unsqueeze(-1)

                loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

                total_active_elements += len(loss)
                epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch

        ood_metrics['mse'] = epoch_loss
        return ood_metrics