import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Positional Encoding (batch_first=True)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # sin cho các chiều chẵn
        pe[:, 1::2] = torch.cos(position * div_term)  # cos cho các chiều lẻ
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Mô hình ProbabilisticForecastTransformerVAE
class ProbabilisticForecastTransformer(nn.Module):
    def __init__(self, window_size, num_series, static_dim, 
                 latent_dim=32, d_model=64, nhead=4, num_layers=2,
                 hidden_dim=128, dropout=0.1, output_dim=2):
        """
        window_size: số bước thời gian của chuỗi lịch sử.
        num_series: số biến trong chuỗi (ví dụ: 2: Units, Revenue)
        static_dim: số đặc trưng ngoại lai.
        latent_dim: kích thước không gian latent.
        d_model: kích thước embedding của Transformer.
        nhead: số head của multi-head attention.
        num_layers: số lớp Transformer encoder.
        hidden_dim: kích thước tầng ẩn trong encoder/decoder FC.
        output_dim: số biến dự báo (2).
        """
        super(ProbabilisticForecastTransformer, self).__init__()
        self.window_size = window_size
        self.num_series = num_series
        self.static_dim = static_dim
        
        # Encoder:
        # Chúng ta flatten x_seq: (batch, window_size*num_series)
        # Sau đó, kết hợp với static features: (batch, window_size*num_series + static_dim)
        self.encoder_input_dim = window_size * num_series + static_dim
        
        # Sử dụng một tầng Linear để chiếu đầu vào lên không gian d_model
        self.enc_fc = nn.Linear(self.encoder_input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Sau Transformer, ta dùng một tầng FC để lấy biểu diễn ẩn
        self.enc_fc_out = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Từ đó, tính μ_z và logvar_z
        self.fc_mu_z = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar_z = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder:
        # Decoder nhận vào latent vector (latent_dim) và static features (static_dim)
        self.decoder_input_dim = latent_dim + static_dim
        self.dec_fc = nn.Sequential(
            nn.Linear(self.decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Thêm một skip connection từ flattened x_seq (để giữ thông tin chi tiết)
        self.dec_skip = nn.Linear(window_size * num_series, hidden_dim)
        
        # Kết hợp decoder và skip, sau đó ra đầu ra dự báo phân phối: mu_y và logvar_y cho mỗi biến
        self.final_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * 2)
        )
    
    def encode(self, x_seq, x_cal):
        batch_size = x_seq.size(0)
        # Flatten x_seq
        x_seq_flat = x_seq.view(batch_size, -1)  # (batch, window_size * num_series)
        # Kết hợp với static features
        enc_input = torch.cat([x_seq_flat, x_cal], dim=1)  # (batch, encoder_input_dim)
        # Chiếu lên d_model
        x = self.enc_fc(enc_input)  # (batch, d_model)
        # Ta cần đưa thành chuỗi cho Transformer, giả sử seq_len=1
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, 1, d_model)
        x = x.squeeze(1)  # (batch, d_model)
        h_enc = self.enc_fc_out(x)  # (batch, hidden_dim)
        mu_z = self.fc_mu_z(h_enc)  # (batch, latent_dim)
        logvar_z = self.fc_logvar_z(h_enc)  # (batch, latent_dim)
        return mu_z, logvar_z, x_seq_flat
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, x_cal, skip_flat):
        # z: (batch, latent_dim), x_cal: (batch, static_dim), skip_flat: (batch, window_size*num_series)
        dec_input = torch.cat([z, x_cal], dim=1)  # (batch, decoder_input_dim)
        h_dec = self.dec_fc(dec_input)  # (batch, hidden_dim)
        skip_feat = self.dec_skip(skip_flat)  # (batch, hidden_dim)
        combined = torch.cat([h_dec, skip_feat], dim=1)  # (batch, 2*hidden_dim)
        out = self.final_fc(combined)  # (batch, output_dim*2)
        return out
    
    def forward(self, x_seq, x_cal):
        mu_z, logvar_z, skip_flat = self.encode(x_seq, x_cal)
        z = self.reparameterize(mu_z, logvar_z)
        out = self.decode(z, x_cal, skip_flat)
        return out, mu_z, logvar_z

def distributional_vae_loss(out, y_true, mu_z, logvar_z, kl_weight=0.001):
    # out: (batch, output_dim*2) -> tách ra μ_y và logvar_y
    batch_size, out_dim2 = out.shape
    output_dim = out_dim2 // 2
    mu_y = out[:, :output_dim]
    logvar_y = out[:, output_dim:]
    
    sigma_y = torch.exp(0.5 * logvar_y)
    nll = 0.5 * (np.log(2 * np.pi) + logvar_y + ((y_true - mu_y)**2 / (sigma_y**2)))
    nll = torch.mean(torch.sum(nll, dim=1))
    
    kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    
    return nll + kl_weight * kl