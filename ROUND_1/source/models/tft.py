import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, seq_input_dim, d_model, kernel_sizes=[3, 5, 7], dropout=0.1):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.branches = nn.ModuleList()
        num_branches = len(kernel_sizes)
        out_channels = d_model // num_branches
        # Nếu không chia hết, ta có thể tính theo float, sau đó chuyển thành int
        for k in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=seq_input_dim, out_channels=out_channels, kernel_size=k, padding=k//2),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout(dropout)
                )
            )
        self.proj = nn.Linear(out_channels * num_branches, d_model)
        
    def forward(self, x_seq):
        # x_seq: (batch, seq_len, seq_input_dim)
        x = x_seq.transpose(1, 2)  # (batch, seq_input_dim, seq_len)
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))  # mỗi tensor: (batch, out_channels, seq_len)
        combined = torch.cat(branch_outputs, dim=1)  # (batch, out_channels*num_branches, seq_len)
        pooled = combined.mean(dim=2)  # Global average pooling trên chiều seq_len -> (batch, out_channels*num_branches)
        multi_scale_feature = self.proj(pooled)  # (batch, d_model)
        return multi_scale_feature

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        """
        Cho mỗi đặc trưng static (input_dim), học một biểu diễn riêng (chiếu lên d_model)
        và tính trọng số qua một MLP để tổng hợp.
        """
        super(VariableSelectionNetwork, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Từng đặc trưng được chiếu lên d_model
        self.feature_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(input_dim)])
        
        # Mạng tính trọng số (sẽ cho ra vector kích thước input_dim)
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

class TemporalFusionTransformer(nn.Module):
    def __init__(self, seq_input_dim, cal_input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        """
        Mô hình kết hợp:
         - Xử lý chuỗi thời gian qua Linear, Positional Encoding và TransformerEncoder.
         - Bổ sung Multi-Scale Feature Extractor để trích xuất đặc trưng cục bộ từ chuỗi.
         - Xử lý đặc trưng static qua Variable Selection Network.
         - Cơ chế gating để kết hợp thông tin từ chuỗi (từ Transformer và Multi-Scale) và static.
         - Bổ sung Autoregressive Skip Connection từ giá trị cuối của chuỗi đầu vào.
         - Hợp nhất với residual connection và dự báo qua tầng FC cuối.
        """
        super(TemporalFusionTransformer, self).__init__()
        # Xử lý chuỗi thời gian với Transformer
        self.input_projection = nn.Linear(seq_input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-scale branch
        self.multi_scale_extractor = MultiScaleFeatureExtractor(seq_input_dim, d_model, kernel_sizes=[3,5,7], dropout=dropout)
        
        # Xử lý đặc trưng static qua Variable Selection Network
        self.variable_selection = VariableSelectionNetwork(cal_input_dim, d_model, dropout=dropout)
        
        # Cơ chế gating để kết hợp thông tin từ chuỗi và static
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # Fusion cuối cùng với residual connection
        self.fc_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Autoregressive layer: sử dụng giá trị cuối của chuỗi đầu vào
        self.ar_layer = nn.Linear(seq_input_dim, d_model)
        
        # Lớp kết hợp cuối cùng: kết hợp fusion và AR component
        self.final_fc = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x_seq, x_cal):
        # Xử lý chuỗi thời gian qua Transformer
        temporal = self.input_projection(x_seq)         # (batch, seq_len, d_model)
        temporal = self.positional_encoding(temporal)     # (batch, seq_len, d_model)
        temporal = self.transformer_encoder(temporal)     # (batch, seq_len, d_model)
        temporal_feature = temporal[:, -1, :]             # (batch, d_model)
        
        # Multi-scale features từ x_seq
        multi_scale_feature = self.multi_scale_extractor(x_seq)  # (batch, d_model)
        
        # Kết hợp các đặc trưng chuỗi: transformer + multi-scale
        combined_temporal = temporal_feature + multi_scale_feature  # (batch, d_model)
        
        # Xử lý đặc trưng static qua Variable Selection Network
        static_feature, _ = self.variable_selection(x_cal)  # (batch, d_model)
        
        # Kết hợp thông tin từ chuỗi và static qua gating
        combined = torch.cat([combined_temporal, static_feature], dim=1)  # (batch, 2*d_model)
        gate = self.gate_fc(combined)                     # (batch, d_model)
        fused = gate * combined_temporal + (1 - gate) * static_feature  # (batch, d_model)
        
        # Fusion với residual connection
        fused_out = self.fc_fusion(fused) + fused          # (batch, d_model)
        
        # Autoregressive component: từ giá trị cuối của x_seq
        ar_input = x_seq[:, -1, :]                         # (batch, seq_input_dim)
        ar_feature = self.ar_layer(ar_input)               # (batch, d_model)
        
        # Kết hợp fusion và AR component
        combined_final = torch.cat([fused_out, ar_feature], dim=1)  # (batch, 2*d_model)
        output = self.final_fc(combined_final)             # (batch, output_dim)
        return output