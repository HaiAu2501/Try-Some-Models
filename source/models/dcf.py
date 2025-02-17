import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DistributionalConditionalForecast(nn.Module):
    def __init__(self, window_size, num_series, static_dim,
                 latent_dim=32, hidden_dim=128, dropout=0.1, output_dim=2):
        """
        - Giống ConditionalForecastVAE, nhưng decoder xuất ra (mu_y, logvar_y) thay vì 1 vector cố định.
        - Qua đó mô hình có thể tự điều chỉnh độ bất định (sigma_y) cho những điểm dao động mạnh (spike).
        """
        super(DistributionalConditionalForecast, self).__init__()
        self.window_size = window_size
        self.num_series = num_series
        self.static_dim = static_dim
        
        # Encoder input = flatten(x_seq) + static
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
        
        # Decoder input = latent + static
        self.decoder_input_dim = latent_dim + static_dim
        self.fc_dec = nn.Sequential(
            nn.Linear(self.decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Skip connection (flatten(x_seq)) -> fc_skip
        self.fc_skip = nn.Linear(window_size * num_series, hidden_dim)
        
        # Cuối cùng, xuất (mu_y, logvar_y) => output_dim * 2
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * 2)
        )
    
    def encode(self, x_seq, x_cal):
        batch_size = x_seq.size(0)
        x_seq_flat = x_seq.view(batch_size, -1)  # (batch, window_size * num_series)
        enc_input = torch.cat([x_seq_flat, x_cal], dim=1)  # (batch, encoder_input_dim)
        h_enc = self.fc_enc(enc_input)
        mu_z = self.fc_mu_z(h_enc)
        logvar_z = self.fc_logvar_z(h_enc)
        return mu_z, logvar_z, x_seq_flat
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, x_cal, skip_flat):
        dec_input = torch.cat([z, x_cal], dim=1)  # (batch, decoder_input_dim)
        h_dec = self.fc_dec(dec_input)
        skip_feat = self.fc_skip(skip_flat)  # (batch, hidden_dim)
        combined = torch.cat([h_dec, skip_feat], dim=1)  # (batch, hidden_dim*2)
        out = self.final_fc(combined)  # (batch, output_dim*2) => (mu_y, logvar_y)
        return out
    
    def forward(self, x_seq, x_cal):
        mu_z, logvar_z, skip_flat = self.encode(x_seq, x_cal)
        z = self.reparameterize(mu_z, logvar_z)
        out = self.decode(z, x_cal, skip_flat)
        return out, mu_z, logvar_z

def dcf_loss(out, y_true, mu_z, logvar_z, kl_weight=0.001):
    """
    out: (batch, output_dim*2) => tách (mu_y, logvar_y)
    y_true: (batch, output_dim)
    mu_z, logvar_z: latent
    """
    batch_size, out_dim2 = out.shape
    output_dim = out_dim2 // 2
    mu_y = out[:, :output_dim]
    logvar_y = out[:, output_dim:]
    
    # Negative log-likelihood của y ~ N(mu_y, sigma_y^2)
    # => nll = 0.5 * [log(2pi) + logvar_y + (y_true - mu_y)^2 / exp(logvar_y)]
    # sum over output_dim, mean over batch
    sigma_y = torch.exp(0.5 * logvar_y)
    nll = 0.5 * (np.log(2 * np.pi) + logvar_y + ((y_true - mu_y)**2 / (sigma_y**2)))
    nll = torch.sum(nll, dim=1).mean()  # sum each sample's output_dim, then average
    
    # KL for latent z
    kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    
    return nll + kl_weight * kl
