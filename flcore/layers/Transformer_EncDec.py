import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.layers.Autoformer_EncDec import series_decomp

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
    
class GatingNetwork(nn.Module):
    def __init__(self, input_size, hidden_dim, num_experts=4):
        super(GatingNetwork, self).__init__()
        # channel-wise gating
        self.linear = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x
    
class GatingNetworkWithTopK(nn.Module):
    def __init__(self, input_size, hidden_dim, num_experts=4, capacity_factore=1.0):
        super(GatingNetworkWithTopK, self).__init__()

        self.capacity_factor = capacity_factore
        self.linear = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        gate_scores = self.softmax(self.linear2(self.relu(self.linear(x))))

        # determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # combine gating scores with the mask
        masked_gate_scores = gate_scores * mask
        
        # denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + 1e-4
        )
        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        return gate_scores
   
class GatingNetworkWithDecomp(nn.Module):
    def __init__(self, input_size, hidden_dim, num_experts=4):
        super(GatingNetworkWithDecomp, self).__init__()

        self.decom = series_decomp(kernel_size=25)
        self.trend_linear = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.seasonal_linear = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        trend_init, seasonal_init = self.decom(x)

        trend_init, seasonal_init = self.trend_linear(trend_init), self.seasonal_linear(seasonal_init)
        
        med_series = self.relu(trend_init + seasonal_init)
        med_series = self.linear2(med_series)
        scores = self.softmax(med_series)

        return scores
    
class GatingNetworkWithDecompWithTopK(nn.Module):
    def __init__(self, input_size, hidden_dim, num_experts=4, capacity_factore=2.0, k_value=1):
        super(GatingNetworkWithDecompWithTopK, self).__init__()

        self.capacity_factor = capacity_factore
        self.decom = series_decomp(kernel_size=25)
        self.trend_linear = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.seasonal_linear = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

        self.k_value = k_value
        
    def forward(self, x):
        trend_init, seasonal_init = self.decom(x)

        trend_init, seasonal_init = self.trend_linear(trend_init), self.seasonal_linear(seasonal_init)
        
        med_series = self.relu(trend_init + seasonal_init)
        med_series = self.linear2(med_series)
        gate_scores = self.softmax(med_series)

        # print(f'timescale weights are {gate_scores}')
        # determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(self.k_value, dim=-1)

        # mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # combine gating scores with the mask
        masked_gate_scores = gate_scores * mask
        
        # denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + 1e-4
        )
        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        return gate_scores
    
    
class FFNMoe(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts, k_value):
        super(FFNMoe, self).__init__()

        self.experts = nn.ModuleList(
            [
                nn.Linear(d_model, d_model)
                for _ in range(num_experts)
            ]
        ) 

        # self.gate = GatingNetwork(
        #     input_size=d_model,
        #     hidden_dim=hidden_dim,
        #     num_experts=num_experts
        # )

        # self.gate = GatingNetworkWithDecomp(
        #     input_size=d_model,
        #     hidden_dim=hidden_dim,
        #     num_experts=num_experts
        # )

        self.gate = GatingNetworkWithDecompWithTopK(
            input_size=d_model,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            k_value=k_value
        )

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]

        # (batch_size, seq_len, num_experts)
        gate_scores = self.gate(x)

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output

    
class EncoderLayerMoE(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", k_value=1):
        super(EncoderLayerMoE, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.ffn = FFNMoe(d_model=d_model, hidden_dim=d_model//2, num_experts=4, k_value=k_value)
        # self.add_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        #### MoE ####
        moe_y = self.ffn(y)
        moe_y = moe_y + x

        return self.norm2(moe_y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
