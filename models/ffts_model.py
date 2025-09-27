import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flcore.layers.Transformer_EncDec import Encoder, EncoderLayer, GatingNetworkWithDecompWithTopK
from flcore.layers.SelfAttention_Family import FullAttention, AttentionLayer
from flcore.layers.Embed import PatchEmbedding
from flcore.layers.Autoformer_EncDec import series_decomp


def _get_activation_fn(activation):
    """Get activation function"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class TimescaleExpert(nn.Module):
    """Individual timescale expert for different temporal patterns"""
    def __init__(self, d_model, timescale='hour'):
        super(TimescaleExpert, self).__init__()
        self.timescale = timescale
        
        # Expert-specific FFN
        self.expert_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )
        
        # Timescale-specific parameters for different temporal patterns
        if timescale == 'sec':
            self.temporal_weight = nn.Parameter(torch.ones(1) * 0.1)
        elif timescale == 'min':
            self.temporal_weight = nn.Parameter(torch.ones(1) * 0.3)
        elif timescale == 'hour':
            self.temporal_weight = nn.Parameter(torch.ones(1) * 0.5)
        elif timescale == 'day':
            self.temporal_weight = nn.Parameter(torch.ones(1) * 1.0)
        else:
            self.temporal_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Args:
            x: [B, L, D] - input representation
        Returns:
            expert_output: [B, L, D] - expert-specific transformation
        """
        expert_output = self.expert_ffn(x)
        expert_output = expert_output * self.temporal_weight
        return expert_output


class AdaptiveTrendModule(nn.Module):
    """Adaptive Trend Module with multiple timescale experts integrated in Transformer layer"""
    def __init__(self, d_model, num_experts=4, top_k=2):
        super(AdaptiveTrendModule, self).__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Create timescale experts (sec/min/hour/day)
        timescales = ['sec', 'min', 'hour', 'day']
        self.experts = nn.ModuleList([
            TimescaleExpert(d_model, timescale=timescales[i % len(timescales)])
            for i in range(num_experts)
        ])
        
        # Gating network with decomposition and top-k selection
        self.gating_network = GatingNetworkWithDecompWithTopK(
            input_size=d_model,
            hidden_dim=d_model // 2,
            num_experts=num_experts,
            capacity_factore=2.0,
            k_value=top_k
        )

    def forward(self, x):
        """
        Args:
            x: [B, L, D] - input representation
        Returns:
            output: [B, L, D] - ATM processed representation
        """
        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        
        # Get gating weights with decomposition and top-k
        gate_scores = self.gating_network(x)  # [B, L, num_experts]
        
        # Handle NaN values
        if torch.isnan(gate_scores).any():
            gate_scores[torch.isnan(gate_scores)] = 0
        
        # Stack expert outputs
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, D, num_experts]
        
        # Handle NaN values in expert outputs
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0
        
        # Weighted combination of expert outputs
        atm_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )  # [B, L, D]
        
        return atm_output


class EncoderLayerFFTS(nn.Module):
    """FFTS Encoder Layer with integrated Adaptive Trend Module"""
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu", 
                 num_experts=4, top_k=2):
        super(EncoderLayerFFTS, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)  # Additional norm for ATM
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # Adaptive Trend Module
        self.atm = AdaptiveTrendModule(d_model, num_experts, top_k)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Self-attention
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Standard FFN
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        x = self.norm2(x)
        
        # Adaptive Trend Module
        atm_out = self.atm(x)
        x = x + atm_out
        x = self.norm3(x)

        return x, attn


class FlattenHead(nn.Module):
    """Prediction head for time series forecasting"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FFTSModel(nn.Module):
    """FFTS: Federated Foundation Time Series model with Adaptive Trend Module"""
    
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, 
                 patch_len=16, stride=8, dropout=0.1, activation='gelu',
                 num_experts=4, top_k=2):
        """
        Args:
            feat_dim: number of features/variables
            max_len: maximum sequence length
            d_model: model dimension
            n_heads: number of attention heads
            num_layers: number of transformer layers
            patch_len: patch length for patch embedding
            stride: stride for patch embedding
            dropout: dropout rate
            activation: activation function
            num_experts: number of experts in ATM
            top_k: top-k expert selection
        """
        super().__init__()
        
        self.seq_len = max_len
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        padding = stride
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)
        
        self.act = _get_activation_fn(activation)
        
        # Transformer encoder with integrated ATM
        self.encoder = Encoder(
            [
                EncoderLayerFFTS(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=dropout), 
                        d_model, n_heads),
                    d_model,
                    d_model * 4,
                    dropout=dropout,
                    activation=activation,
                    num_experts=num_experts,
                    top_k=top_k
                ) for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Prediction head
        self.head_nf = d_model * int((self.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(feat_dim, self.head_nf, max_len, head_dropout=dropout)
        
        # Loss regularization parameter
        self.lambda_reg = 0.01

    def forward(self, x_enc, padding_masks=None, return_features=False):
        """
        Args:
            x_enc: [B, L, D] - input time series
            padding_masks: padding masks (optional)
            return_features: whether to return intermediate features
        Returns:
            dec_out: [B, L, D] - predicted time series
            features: intermediate features (if return_features=True)
        """
        # Normalization (Non-stationary Transformer style)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, D, L]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*D, patch_num, d_model]
        
        # Transformer encoder with integrated ATM
        enc_out, attns = self.encoder(enc_out)  # [B*D, patch_num, d_model]
        
        # Reshape for prediction head
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, D, d_model, patch_num]
        
        # Prediction head
        dec_out = self.head(enc_out)  # [B, D, target_window]
        dec_out = dec_out.permute(0, 2, 1)  # [B, target_window, D]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        if return_features:
            features = {
                'encoder_output': enc_out,
                'attention_weights': attns
            }
            return dec_out, features
        
        return dec_out

    def get_atm_parameters(self):
        """Get ATM-specific parameters for regularization"""
        atm_params = []
        for name, param in self.named_parameters():
            if 'atm' in name:
                atm_params.append(param)
        return atm_params

    def compute_atm_alignment_loss(self, client_atm_params, global_atm_params):
        """Compute ATM alignment regularization loss"""
        if global_atm_params is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        alignment_loss = 0.0
        for client_param, global_param in zip(client_atm_params, global_atm_params):
            alignment_loss += F.mse_loss(client_param, global_param.detach())
        
        return self.lambda_reg * alignment_loss


# Alias for backward compatibility
Model = FFTSModel
