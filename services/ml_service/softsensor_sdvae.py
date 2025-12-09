import torch
import torch.nn as nn
import torch.nn.functional as F


class SDVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.1, alpha=1e-4):
        super().__init__()
        self.alpha = alpha

        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)

        decoder_layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim

        self.decoder_layers = nn.Sequential(*decoder_layers)
        self.fc_decoder = nn.Linear(current_dim, input_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_layers(z)
        return self.fc_decoder(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, xy_batch):
        h = self.encoder(xy_batch)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        L = 20
        z_samples = []
        for _ in range(L):
            z_sample = self.reparameterize(mu, logvar)
            z_samples.append(z_sample)

        z = torch.stack(z_samples).mean(dim=0)
        recon_xy = self.fc_decoder(self.decoder_layers(z))
        return recon_xy, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, kld_weight=0.01):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        kl_div = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        l2 = torch.tensor(0.0, device=x.device)
        for param in self.parameters():
            l2 += torch.norm(param, p=2)

        return recon_loss + kl_div * kld_weight + l2 * self.alpha


class MUDVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout_rate=0.1, alpha=1e-4):
        super().__init__()
        self.alpha = alpha

        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(current_dim, latent_dim)
        self.fc_logvar = nn.Linear(current_dim, latent_dim)

        decoder_layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim

        self.decoder_layers = nn.Sequential(*decoder_layers)
        self.fc_decoder = nn.Linear(current_dim, input_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_layers(z)
        return self.fc_decoder(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, mu_prior, logvar_prior, kld_weight=0.01):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        var_prior = torch.exp(logvar_prior)
        var = torch.exp(logvar)

        kl_div = 0.5 * torch.sum(
            logvar_prior - logvar +
            (var + (mu - mu_prior) ** 2) / var_prior - 1
        )

        l2 = torch.tensor(0.0, device=x.device)
        for param in self.parameters():
            l2 += torch.norm(param, p=2)

        return recon_loss + kl_div * kld_weight + l2 * self.alpha


class SoftSensor(nn.Module):
    def __init__(self, encoder: MUDVAE, decoder: SDVAE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, logvar = self.encoder.encode(x)
        z = self.encoder.reparameterize(mu, logvar)
        recon_xy = self.decoder.decode(z)
        return recon_xy[:, -1]
