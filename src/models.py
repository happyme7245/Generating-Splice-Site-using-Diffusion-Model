import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    '''
    U-net is used to denoise the given data during  reverse process
    It predicts the noise which is used to calculate the mean of the Gaussian distribution
    '''
    def __init__(self, input_dim, num_features):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_features, input_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=64):
        super(UNet1D, self).__init__()

        # Encoding path
        self.enc1 = ConvBlock(in_channels, n_filters)
        self.enc2 = ConvBlock(n_filters, n_filters * 2)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8)

        # Max-pooling
        self.pool = nn.MaxPool1d(2, padding=1)

        # Bottleneck
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16)

        # Decoding path
        self.upconv4 = nn.ConvTranspose1d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(n_filters * 16, n_filters * 8)
        self.upconv3 = nn.ConvTranspose1d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 4)
        self.upconv2 = nn.ConvTranspose1d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(n_filters * 4, n_filters * 2)
        self.upconv1 = nn.ConvTranspose1d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(n_filters * 2, n_filters)

        # Final output layer
        self.final_conv = nn.Conv1d(n_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = F.pad(dec4, (0, enc4.size(2) - dec4.size(2)))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = F.pad(dec3, (0, enc3.size(2) - dec3.size(2)))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = F.pad(dec2, (0, enc2.size(2) - dec2.size(2)))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = F.pad(dec1, (0, enc1.size(2) - dec1.size(2)))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        # Final output
        out = self.final_conv(dec1)
        return out


class DDPM(nn.Module):
    def __init__(self, betas, unet):
        super(DDPM, self).__init__()
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # cumulative product of all alphas. This is used to calculate noise (probability q)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.unet = unet

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1).to(x_start.device) * x_start +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).to(x_start.device) * noise
        )  # math.sqrt(a)*x + math.sqrt(1-a)*noise

    def p_losses(self, x_start, t, noise=None, loss_type='l2'):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)  # assign noise as q_sample
        predicted_noise = self.unet(x_noisy)  # predict noise by using Unet
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()  # set model to not calculate gradient
    def p_sample(self, x, t):  # x: sequence tensor
        betas_t = self.betas[t].to(x.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x.device)
        sqrt_recip_alphas_t = torch.sqrt(1 / self.alphas[t]).to(x.device)
        sqrt_recipm1_alphas_t = torch.sqrt(1 / (1 - self.alphas_cumprod[t])).to(x.device)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * sqrt_recipm1_alphas_t * self.unet(x))
        model_var = betas_t * sqrt_one_minus_alphas_cumprod_t
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        return model_mean + torch.sqrt(model_var) * noise

    @torch.no_grad()
    def sample(self, seq_length, batch_size, device):
        x = torch.randn((batch_size, 4, seq_length), device=device)  # x: initial noise, 4: dimension of one-hot encoded sequence
        for t in reversed(range(len(self.betas))):  # reverse process
            x = self.p_sample(x, t)  # update x by using preduction
        return x
