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

##########################################
# Quantile Loss (Pinball Loss)
##########################################
def quantile_loss(y_pred, y_true, quantiles):
    """
    y_pred: (batch, output_dim*num_quantiles) → reshape thành (batch, num_quantiles, output_dim)
    y_true: (batch, output_dim)
    quantiles: list hoặc array các quantile (ví dụ [0.05, 0.50, 0.95])
    """
    batch_size, out_dim_times_q = y_pred.shape
    num_quantiles = len(quantiles)
    output_dim = out_dim_times_q // num_quantiles
    y_pred = y_pred.view(batch_size, num_quantiles, output_dim)
    loss = 0
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, i, :]
        loss += torch.max((q - 1) * errors, q * errors).unsqueeze(1)
    loss = torch.mean(loss)
    return loss

def conditional_quantile_vae_loss(out, y_true, mu_z, logvar_z, quantiles, kl_weight=0.001):
    q_loss = quantile_loss(out, y_true, quantiles)
    kl_loss = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    return q_loss + kl_weight * kl_loss

# window_size = 30
# static_dim = 18
# num_series = 2

# quantiles = [0.05, 0.50, 0.95]  # Dự báo 3 quantiles

# latent_dim = 32
# hidden_dim = 128
# output_dim = 2

# model = ConditionalQuantileVAE(window_size, num_series, static_dim,
#                                 latent_dim=latent_dim, hidden_dim=hidden_dim,
#                                 dropout=0.1, output_dim=output_dim, num_quantiles=len(quantiles))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# kl_weight = 0.001  # cố định

# epochs = 150
# train_losses = []
# val_losses = []
# best_val_loss = float('inf')

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     kl_weight = min(0.001, 0.001 * (epoch / 50))
#     for x_seq, x_cal, y in train_loader:
#         x_seq = x_seq.to(device)
#         x_cal = x_cal.to(device)
#         y = y.to(device)
        
#         optimizer.zero_grad()
#         out, mu_z, logvar_z = model(x_seq, x_cal)
#         loss = conditional_quantile_vae_loss(out, y, mu_z, logvar_z, quantiles, kl_weight=kl_weight)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         running_loss += loss.item() * x_seq.size(0)
#     epoch_train_loss = running_loss / len(train_loader.dataset)
#     train_losses.append(epoch_train_loss)
    
#     model.eval()
#     running_val_loss = 0.0
#     with torch.no_grad():
#         for x_seq, x_cal, y in val_loader:
#             x_seq = x_seq.to(device)
#             x_cal = x_cal.to(device)
#             y = y.to(device)
#             out, mu_z, logvar_z = model(x_seq, x_cal)
#             loss = conditional_quantile_vae_loss(out, y, mu_z, logvar_z, quantiles, kl_weight=kl_weight)
#             running_val_loss += loss.item() * x_seq.size(0)
#     epoch_val_loss = running_val_loss / len(val_loader.dataset)
#     val_losses.append(epoch_val_loss)
    
#     scheduler.step(epoch_val_loss)
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, KL Weight: {kl_weight:.6f}")
    
#     if epoch_val_loss < best_val_loss:
#         best_val_loss = epoch_val_loss
#         checkpoint = {
#             'epoch': epoch+1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'val_loss': epoch_val_loss,
#         }
#         torch.save(checkpoint, 'checkpoints/CQV.pth')
#         print(f"Best model updated at epoch {epoch+1} with validation loss {epoch_val_loss:.4f}")