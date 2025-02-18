import torch
import torch.nn as nn
from torchdiffeq import odeint

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv,
            self.relu,
            self.dropout
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
    
    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        out = self.net(x)
        out = out[:, :, :x.size(2)]  # đảm bảo output có cùng seq_len (causal convolution)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNTimeSeriesBranch(nn.Module):
    def __init__(self, seq_input_dim, num_channels, kernel_size, dropout):
        super(TCNTimeSeriesBranch, self).__init__()
        layers = []
        num_levels = len(num_channels)
        in_channels = seq_input_dim
        for i in range(num_levels):
            layers.append(
                TCNBlock(in_channels, num_channels[i], kernel_size, dilation=2**i, dropout=dropout)
            )
            in_channels = num_channels[i]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, seq_len, seq_input_dim)
        x = x.transpose(1,2)  # (batch, seq_input_dim, seq_len)
        out = self.network(x)  # (batch, channels, seq_len)
        # Lấy giá trị của time step cuối cùng (causal)
        out = out[:, :, -1]    # (batch, channels)
        return out

class ODEFunc(nn.Module):
    def __init__(self, d_model):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, t, h):
        return self.net(h)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, t0=0, t1=1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        # Đăng ký t làm buffer để nó tự động chuyển theo device của mô hình
        self.register_buffer('t', torch.tensor([t0, t1]).float())
    
    def forward(self, x):
        # Đảm bảo t nằm trên cùng device với x
        t = self.t.to(x.device)
        out = odeint(self.odefunc, x, t, method='dopri5')
        return out[-1]

class ODETransformerBranch(nn.Module):
    def __init__(self, seq_input_dim, d_model, nhead, num_layers, dropout):
        super(ODETransformerBranch, self).__init__()
        self.input_proj = nn.Linear(seq_input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.odefunc = ODEFunc(d_model)
        self.odeblock = ODEBlock(self.odefunc, t0=0, t1=1)
    
    def forward(self, x):
        # x: (batch, seq_len, seq_input_dim)
        x_proj = self.input_proj(x)                   # (batch, seq_len, d_model)
        trans_feat = self.transformer_encoder(x_proj)   # (batch, seq_len, d_model)
        trans_out = trans_feat[:, -1, :]               # (batch, d_model)
        # Dùng trung bình của x_proj làm trạng thái ban đầu cho ODE
        x0 = x_proj.mean(dim=1)                        # (batch, d_model)
        ode_out = self.odeblock(x0)                    # (batch, d_model)
        return trans_out, ode_out

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

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, seq_input_dim, cal_input_dim, d_model, nhead, num_layers, tcn_channels, kernel_size, dropout, output_dim):
        super(TemporalConvolutionalNetwork, self).__init__()
        # TCN branch
        self.tcn_branch = TCNTimeSeriesBranch(seq_input_dim, tcn_channels, kernel_size, dropout)
        self.tcn_proj = nn.Linear(tcn_channels[-1], d_model)
        # ODE-Transformer branch
        self.ode_trans_branch = ODETransformerBranch(seq_input_dim, d_model, nhead, num_layers, dropout)
        # Fusion gating cho 2 nhánh thời gian
        self.gate_fc_time = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        # Static branch
        self.variable_selection = VariableSelectionNetwork(cal_input_dim, d_model, dropout)
        self.gate_fc_static = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        # Fusion cuối cùng với residual
        self.fc_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # AR skip connection: từ giá trị cuối của x_seq
        self.ar_layer = nn.Linear(seq_input_dim, d_model)
        # Head dự báo
        self.final_fc = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x_seq, x_cal):
        # x_seq: (batch, seq_len, seq_input_dim); x_cal: (batch, cal_input_dim)
        # TCN branch
        tcn_feat = self.tcn_branch(x_seq)         # (batch, tcn_channels[-1])
        tcn_feat = self.tcn_proj(tcn_feat)          # (batch, d_model)
        # ODE-Transformer branch
        trans_out, ode_out = self.ode_trans_branch(x_seq)  # mỗi thứ: (batch, d_model)
        # Hòa trộn hai nhánh thời gian
        combined_time = torch.cat([tcn_feat, trans_out + ode_out], dim=1)  # (batch, 2*d_model)
        gate_time = self.gate_fc_time(combined_time)  # (batch, d_model)
        fused_time = gate_time * (tcn_feat + trans_out + ode_out)  # (batch, d_model)
        
        # Static branch
        static_feat, _ = self.variable_selection(x_cal)  # (batch, d_model)
        combined_static = torch.cat([fused_time, static_feat], dim=1)  # (batch, 2*d_model)
        gate_static = self.gate_fc_static(combined_static)  # (batch, d_model)
        fused_all = gate_static * fused_time + (1 - gate_static) * static_feat  # (batch, d_model)
        fused_all = self.fc_fusion(fused_all) + fused_all  # residual, (batch, d_model)
        
        # Autoregressive component: từ giá trị cuối của x_seq
        ar_input = x_seq[:, -1, :]  # (batch, seq_input_dim)
        ar_feat = self.ar_layer(ar_input)  # (batch, d_model)
        
        # Kết hợp final
        combined_final = torch.cat([fused_all, ar_feat], dim=1)  # (batch, 2*d_model)
        output = self.final_fc(combined_final)  # (batch, output_dim)
        return output