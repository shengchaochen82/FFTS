import torch
import numpy as np
from utils.tools import EarlyStopping_Nosaver, adjust_learning_rate
from collections import OrderedDict
from flcore.trainers.runner_base import BaseRunner
from flcore.losses.loss import l2_reg_loss



class UnsupervisedRunner(BaseRunner):

    def train(self, max_local_epoch=None):

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping_Nosaver(patience=self.args.patience, verbose=True)
        
        for epoch in range(max_local_epoch):
            train_loss = []
            epoch_loss = self.train_epoch(epoch_num=epoch)
            train_loss.append(epoch_loss)
            train_loss = np.average(train_loss)
            vali_loss = self.evaluate()

            print("\nEpoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
        
        # self.model.to('cpu')
        # torch.cuda.empty_cache()


    def train_epoch(self, epoch_num=None, global_protos=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        for i, batch in enumerate(self.train_loader):

            X, targets, target_masks, padding_masks = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            # loss = self.loss_module(predictions, targets) 
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization
            # self.print_gradient_statistics(self.model)
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # metrics = {"loss": mean_loss.item()}
            # if i % self.print_interval == 0:
            #     ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            #     self.print_callback(i, metrics, prefix='Training ' + ending)
        
            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch

        return epoch_loss

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch


        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):

                X, targets, target_masks, padding_masks = batch
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore

                predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

                # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
                target_masks = target_masks * padding_masks.unsqueeze(-1)
                loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
                # loss = self.loss_module(predictions, targets) 
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

                total_active_elements += len(loss)
                epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch

        return epoch_loss
    
    def test(self):
        
        ood_metrics = OrderedDict()
        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        # -Testing 1
        # for name, param in self.model.named_parameters():
        #         print(f"Layer: {name} | Weights: {param.data.norm()} | Gradients: {param.grad.norm() if param.grad is not None else 'None'}")

        # -Testing 2
        # hooks = []
        # for layer in self.model.children():
        #     hook = layer.register_forward_hook(self.forward_hook)
        #     hooks.append(hook)

        # zero indicates without test loader [drop last]
        with torch.no_grad():
            for i, batch in enumerate(self.ood_test_loader):
 
                X, targets, target_masks, padding_masks = batch
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                predictions = self.model(X.to(self.device), padding_masks)
                # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
                target_masks = target_masks * padding_masks.unsqueeze(-1)

                # print(X[0,0], predictions[0, 0], targets[0, 0])
                loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
                # loss = self.loss_module(predictions, targets) 
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

                total_active_elements += len(loss)
                epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch

        ood_metrics['mse'] = epoch_loss
        return ood_metrics
    
    def forward_hook(self, module, input, output):
        print(f"Layer: {module.__class__.__name__}")
        print(f"Output: {output}\n")

    def check_input_threshold(self, input_tensor, threshold):
        """
        检查输入张量中是否存在大于阈值的值。

        参数:
        - input_tensor (torch.Tensor): 输入张量
        - threshold (float): 阈值

        返回:
        - bool: 如果存在大于阈值的值，返回 True；否则返回 False
        """
        return torch.any(input_tensor > threshold).item()
    
    def print_gradient_statistics(self, model):
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad_data = parameter.grad.data
                print(f'Layer {name} | Grad Max: {grad_data.max()} | Grad Min: {grad_data.min()} | Grad Mean: {grad_data.mean()}')