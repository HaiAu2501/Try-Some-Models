import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

##########################################
# Positional Encoding (batch_first=True)
##########################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

##########################################
# Conditional Quantile VAE (CQuantVAE)
##########################################
class ConditionalQuantileVAE(nn.Module):
    def __init__(self, window_size, num_series, static_dim,
                 latent_dim=32, hidden_dim=128, dropout=0.1, output_dim=2, num_quantiles=3):
        """
        window_size: số bước lịch sử.
        num_series: số biến chuỗi (ví dụ: 2).
        static_dim: số cột static.
        latent_dim: kích thước không gian latent.
        hidden_dim: kích thước tầng ẩn trong FC.
        output_dim: số biến dự báo (2).
        num_quantiles: số lượng quantile cần dự báo (ví dụ: [0.05, 0.50, 0.95] → 3).
        
        => Decoder sẽ xuất ra output_dim * num_quantiles giá trị.
        """
        super(ConditionalQuantileVAE, self).__init__()
        self.window_size = window_size
        self.num_series = num_series
        self.static_dim = static_dim
        self.num_quantiles = num_quantiles
        self.output_dim = output_dim
        
        # Encoder: input = flatten(x_seq) (window_size*num_series) concat x_cal (static_dim)
        self.encoder_input_dim = window_size * num_series + static_dim
        self.fc_enc = nn.Sequential(
            nn.Linear(self.encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu_z = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_z = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: input = [z; x_cal]
        self.decoder_input_dim = latent_dim + static_dim
        self.fc_dec = nn.Sequential(
            nn.Linear(self.decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Skip connection từ flatten(x_seq)
        self.fc_skip = nn.Linear(window_size * num_series, hidden_dim)
        
        # Final head: xuất ra output_dim * num_quantiles (cho mỗi biến, các quantile)
        self.final_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * num_quantiles)
        )
    
    def encode(self, x_seq, x_cal):
        batch_size = x_seq.size(0)
        x_seq_flat = x_seq.view(batch_size, -1)  # (batch, window_size*num_series)
        enc_input = torch.cat([x_seq_flat, x_cal], dim=1)  # (batch, encoder_input_dim)
        h_enc = self.fc_enc(enc_input)  # (batch, hidden_dim)
        mu_z = self.fc_mu_z(h_enc)      # (batch, latent_dim)
        logvar_z = self.fc_logvar_z(h_enc)  # (batch, latent_dim)
        return mu_z, logvar_z, x_seq_flat
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, x_cal, skip_flat):
        dec_input = torch.cat([z, x_cal], dim=1)  # (batch, decoder_input_dim)
        h_dec = self.fc_dec(dec_input)  # (batch, hidden_dim)
        skip_feat = self.fc_skip(skip_flat)  # (batch, hidden_dim)
        combined = torch.cat([h_dec, skip_feat], dim=1)  # (batch, 2*hidden_dim)
        out = self.final_fc(combined)  # (batch, output_dim*num_quantiles)
        return out
    
    def forward(self, x_seq, x_cal):
        mu_z, logvar_z, skip_flat = self.encode(x_seq, x_cal)
        z = self.reparameterize(mu_z, logvar_z)
        out = self.decode(z, x_cal, skip_flat)
        return out, mu_z, logvar_z