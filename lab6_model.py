import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear Noise Scheduler 

class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas         = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas        = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)

    def add_noise(self, x0, noise, t):
        B = x0.shape[0]
        sqrt_alpha_cum = self.sqrt_alpha_cum_prod.to(x0.device)[t].reshape(B)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cum_prod.to(x0.device)[t].reshape(B)
        while len(sqrt_alpha_cum.shape) < len(x0.shape):
            sqrt_alpha_cum = sqrt_alpha_cum.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha_cum * x0 + sqrt_one_minus * noise

    def sample_prev_timestep(self, xt, noise_pred, t_scalar):
        t_int = int(t_scalar.item()) if isinstance(t_scalar, torch.Tensor) else int(t_scalar)
        betas         = self.betas.to(xt.device)
        alphas        = self.alphas.to(xt.device)
        alpha_cum     = self.alpha_cum_prod.to(xt.device)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)

        beta_t           = betas[t_int]
        alpha_t          = alphas[t_int]
        alpha_bar_t      = alpha_cum[t_int]
        sqrt_one_minus_t = sqrt_one_minus[t_int]

        x0_pred = (xt - sqrt_one_minus_t * noise_pred) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        mean    = (xt - beta_t * noise_pred / sqrt_one_minus_t) / torch.sqrt(alpha_t)

        if t_int == 0:
            return mean, x0_pred
        else:
            alpha_bar_prev = alpha_cum[t_int - 1]
            var   = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
            sigma = torch.sqrt(var)
            return mean + sigma * torch.randn_like(xt), x0_pred


def get_time_embedding(time_steps, temb_dim):
    half_dim = temb_dim // 2
    exponent = torch.arange(half_dim, dtype=torch.float32, device=time_steps.device) / half_dim
    factor   = 10000 ** exponent
    t = time_steps.float().unsqueeze(1) / factor.unsqueeze(0)
    return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.norm1   = nn.GroupNorm(8, in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj  = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))
        self.norm2   = nn.GroupNorm(8, out_ch)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H*W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, down_sample=True, num_layers=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(in_ch if i == 0 else out_ch, out_ch, t_emb_dim) for i in range(num_layers)
        ])
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x, t_emb):
        for blk in self.res_blocks:
            x = blk(x, t_emb)
        return self.down(x)


class MidBlock(nn.Module):
    def __init__(self, ch, t_emb_dim, num_heads=4):
        super().__init__()
        self.res1  = ResBlock(ch, ch, t_emb_dim)
        self.attn  = AttentionBlock(ch, num_heads)
        self.res2  = ResBlock(ch, ch, t_emb_dim)

    def forward(self, x, t_emb):
        return self.res2(self.attn(self.res1(x, t_emb)), t_emb)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, t_emb_dim, up_sample=True, num_layers=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1) if up_sample else nn.Identity()
        self.res_blocks = nn.ModuleList([
            ResBlock((in_ch + skip_ch) if i == 0 else out_ch, out_ch, t_emb_dim) for i in range(num_layers)
        ])

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for blk in self.res_blocks:
            x = blk(x, t_emb)
        return x


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        t_dim = config["time_emb_dim"]
        self.t_dim = t_dim
        self.t_proj = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.cond_proj = nn.Sequential(nn.Linear(config["num_classes"], t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        
        self.conv_in = nn.Conv2d(config["im_channels"], config["down_channels"][0], 3, padding=1)

        self.downs = nn.ModuleList()
        chs = config["down_channels"]
        ds = config["down_sample"]
        for i in range(len(chs)-1):
            self.downs.append(DownBlock(chs[i], chs[i+1], t_dim, ds[i], config["num_down_layers"]))

        self.mid = MidBlock(chs[-1], t_dim, config["num_heads"])

        self.ups = nn.ModuleList()
        rev_chs = list(reversed(chs))
        rev_ds = list(reversed(ds))
        for i in range(len(rev_chs)-1):
            self.ups.append(UpBlock(rev_chs[i], rev_chs[i+1], rev_chs[i+1], t_dim, rev_ds[i], config["num_up_layers"]))

        self.norm_out = nn.GroupNorm(8, chs[0])
        self.conv_out = nn.Conv2d(chs[0], config["im_channels"], 3, padding=1)

    def forward(self, x, t, cond):
        t_emb = get_time_embedding(t, self.t_dim)
        emb   = self.t_proj(t_emb) + self.cond_proj(cond)

        out = self.conv_in(x)
        skips = [out]
        for down in self.downs:
            out = down(out, emb)
            skips.append(out)

        out = self.mid(out, emb)
        
        
        skips.pop() 

        for up in self.ups:
            skip = skips.pop()
            out  = up(out, skip, emb)

        return self.conv_out(F.silu(self.norm_out(out)))

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.named_parameters():
                if v.requires_grad: self.shadow[k].mul_(self.decay).add_(v.data, alpha=1-self.decay)
    def state_dict(self): return self.shadow
    def load_state_dict(self, sd): self.shadow = sd

MODEL_CONFIG = {
    "im_channels": 3, "im_size": 64, "down_channels": [64, 128, 256, 256],
    "down_sample": [True, True, False], "time_emb_dim": 128, "num_down_layers": 2,
    "num_up_layers": 2, "num_heads": 4, "num_classes": 24
}