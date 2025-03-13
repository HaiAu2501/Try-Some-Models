import torch
import torch.nn as nn

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.feature_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(input_dim)])
        self.variable_weights = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim)
        )
    
    def forward(self, x):
        # x: (batch, input_dim)
        projected = [proj(x[:, i].unsqueeze(1)) for i, proj in enumerate(self.feature_proj)]
        projected = torch.stack(projected, dim=1)  # (batch, input_dim, d_model)
        weights = self.variable_weights(x)         # (batch, input_dim)
        weights = torch.softmax(weights, dim=1)
        weighted_sum = torch.sum(projected * weights.unsqueeze(2), dim=1)  # (batch, d_model)
        return weighted_sum, weights

class NHITSBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, forecast_dim, backcast_dim):
        """
        Mỗi block nhận input vector (flattened time series) có kích thước input_dim = window_size*num_series.
        Xuất ra:
          - backcast: phần “giải thích” của input (shape: [batch, backcast_dim])
          - forecast: dự báo (shape: [batch, forecast_dim])
        """
        super(NHITSBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.backcast_layer = nn.Linear(hidden_dim, backcast_dim)
        self.forecast_layer = nn.Linear(hidden_dim, forecast_dim)
    
    def forward(self, x):
        # x: (batch, input_dim)
        h = self.fc(x)
        backcast = self.backcast_layer(h)
        forecast = self.forecast_layer(h)
        return backcast, forecast

class NHITSBranch(nn.Module):
    def __init__(self, window_size, num_series, n_blocks, block_hidden_dim, n_block_layers):
        super(NHITSBranch, self).__init__()
        self.window_size = window_size
        self.num_series = num_series
        self.input_dim = window_size * num_series  # chỉ phần chuỗi thời gian
        self.backcast_dim = self.input_dim
        self.forecast_dim = num_series  # dự báo cho mỗi biến (2 giá trị)
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList([
            NHITSBlock(self.input_dim, block_hidden_dim, n_block_layers, self.forecast_dim, self.backcast_dim)
            for _ in range(n_blocks)
        ])
    
    def forward(self, x_seq):
        # x_seq: (batch, window_size, num_series)
        batch_size = x_seq.size(0)
        x_flat = x_seq.view(batch_size, -1)  # (batch, window_size*num_series)
        residual = x_flat
        forecast_sum = 0
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum  # (batch, num_series)

class DeepARBranch(nn.Module):
    def __init__(self, seq_input_dim, d_model, num_layers, dropout):
        super(DeepARBranch, self).__init__()
        self.lstm = nn.LSTM(seq_input_dim, d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
    
    def forward(self, x_seq):
        # x_seq: (batch, window_size, seq_input_dim)
        output, (hn, cn) = self.lstm(x_seq)  # hn: (num_layers, batch, d_model)
        # Lấy hidden state của layer cuối cùng
        return hn[-1]  # (batch, d_model)

class InvertedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, expansion=4, dropout=0.1):
        super(InvertedTransformerEncoderLayer, self).__init__()
        self.expanded_dim = d_model * expansion
        # Expand projection
        self.fc1 = nn.Linear(d_model, self.expanded_dim)
        # Multi-head attention operating in expanded space
        self.attn = nn.MultiheadAttention(embed_dim=self.expanded_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        # Project back to d_model
        self.fc2 = nn.Linear(self.expanded_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # Feedforward (optional)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src):
        # src: (batch, seq_len, d_model)
        residual = src
        x = self.fc1(src)  # (batch, seq_len, expanded_dim)
        attn_out, _ = self.attn(x, x, x)  # (batch, seq_len, expanded_dim)
        x = self.fc2(attn_out)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        x = self.norm1(residual + x)
        # Feedforward with residual
        residual2 = x
        x_ff = self.ffn(x)
        x = self.norm2(residual2 + x_ff)
        return x

class InvertedTransformerBranch(nn.Module):
    def __init__(self, seq_input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(InvertedTransformerBranch, self).__init__()
        self.input_proj = nn.Linear(seq_input_dim, d_model)
        self.layers = nn.ModuleList([
            InvertedTransformerEncoderLayer(d_model, nhead, expansion=4, dropout=dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x_seq):
        # x_seq: (batch, window_size, seq_input_dim) where seq_input_dim = num_series (2)
        x = self.input_proj(x_seq)  # (batch, window_size, d_model)
        for layer in self.layers:
            x = layer(x)
        # Lấy giá trị của time step cuối cùng
        return x[:, -1, :]  # (batch, d_model)

class HybridForecastingModel(nn.Module):
    def __init__(self, window_size, num_series, static_dim, d_model, nhead, num_layers_transformer,
                 n_blocks_nhits, nhits_hidden_dim, nhits_n_layers, deepar_num_layers, dropout=0.1, output_dim=2):
        super(HybridForecastingModel, self).__init__()
        # NHITS Branch
        self.nhits_branch = NHITSBranch(window_size, num_series, n_blocks_nhits, nhits_hidden_dim, nhits_n_layers)
        # Inverted Transformer Branch
        self.inverted_transformer_branch = InvertedTransformerBranch(num_series, d_model, nhead, num_layers_transformer, dropout)
        # DeepAR Branch
        self.deepar_branch = DeepARBranch(num_series, d_model, deepar_num_layers, dropout)
        
        # Fusion các nhánh thời gian: concatenate output của 3 nhánh
        fusion_input_dim = (num_series) + d_model + d_model  # NHITS branch (forecast: (batch, num_series)) + Inverted Transformer + DeepAR (each: (batch, d_model))
        self.time_fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Static branch: xử lý qua Variable Selection Network
        self.variable_selection = VariableSelectionNetwork(static_dim, d_model, dropout)
        
        # Fusion cuối cùng qua gating: kết hợp time fusion và static
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # AR skip: từ giá trị cuối của x_seq (các biến ban đầu)
        self.ar_layer = nn.Linear(num_series, d_model)
        
        # Head cuối cùng: kết hợp fusion và AR để dự báo
        self.final_fc = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x_seq, x_cal):
        # x_seq: (batch, window_size, num_series)
        # x_cal: (batch, static_dim)
        nhits_out = self.nhits_branch(x_seq)  # (batch, num_series)
        inverted_trans_out = self.inverted_transformer_branch(x_seq)  # (batch, d_model)
        deepar_out = self.deepar_branch(x_seq)  # (batch, d_model)
        
        # Fusion time: concatenate outputs của 3 nhánh
        time_concat = torch.cat([nhits_out, inverted_trans_out, deepar_out], dim=1)  # (batch, num_series + 2*d_model)
        time_fused = self.time_fusion_fc(time_concat)  # (batch, d_model)
        
        # Xử lý static features
        static_feat, _ = self.variable_selection(x_cal)  # (batch, d_model)
        
        # Fusion qua gating
        fusion_concat = torch.cat([time_fused, static_feat], dim=1)  # (batch, 2*d_model)
        gate = self.gate_fc(fusion_concat)  # (batch, d_model)
        fused = gate * time_fused + (1 - gate) * static_feat  # (batch, d_model)
        fused = self.fc_fusion(fused) + fused  # residual
        
        # AR skip từ giá trị cuối của x_seq
        ar_input = x_seq[:, -1, :]  # (batch, num_series)
        ar_feat = self.ar_layer(ar_input)  # (batch, d_model)
        
        combined_final = torch.cat([fused, ar_feat], dim=1)  # (batch, 2*d_model)
        output = self.final_fc(combined_final)  # (batch, output_dim)
        return output
