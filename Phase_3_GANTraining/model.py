import torch
import torch.nn as nn
import json

class IDSGenerator(nn.Module):
    """
    Residual Generator (Policy-Aware): 
    Nhận vào Mã độc thật + Noise ngẫu nhiên.
    Sinh ra Nhiễu (Perturbation) đã được giới hạn bởi Mask (Không dùng Alpha cứng).
    """
    def __init__(self, input_dim, noise_dim=32, policy_path=None):
        super(IDSGenerator, self).__init__()
        self.noise_dim = noise_dim
        
        # ARCHITECTURAL MASK: Khóa cứng giới hạn theo Policy
        mask = torch.ones(input_dim) * 0.1 
        if policy_path:
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy = json.load(f)
            
            for zone_name, zone_data in policy['zones'].items():
                val = zone_data['allowed_variance']
                for idx in zone_data['features']:
                    mask[idx] = val
                    
        self.register_buffer('variance_mask', mask)
        
        # NETWORK CAPACITY
        self.net = nn.Sequential(
            nn.Linear(input_dim + noise_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim),
            
            # SOFTSIGN: Thay thế Tanh() để giảm rủi ro Hard Saturation
            # Softsign: x / (1 + |x|), tiệm cận mượt hơn Tanh
            nn.Softsign() 
        )

    def forward(self, x_real, z):
        # Ghép Malware thật và Noise
        inp = torch.cat([x_real, z], dim=1)
        
        # Sinh ra nhiễu nguyên thủy bằng Softsign
        raw_perturbation = self.net(inp)
        
        # Khóa bằng Mask của Policy (Tôn trọng ranh giới từng feature)
        scaled_perturbation = raw_perturbation * self.variance_mask
        
        # Kiến trúc Residual thuần túy: Giao phó việc ép norm cho L1 Loss
        x_fake = x_real + scaled_perturbation
        
        # Trả về x_fake và scaled_perturbation để tính Loss
        return x_fake, scaled_perturbation


class Discriminator(nn.Module):
    """
    Discriminator / Critic (Dành cho WGAN-GP).
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        # Tăng Capacity (512 -> 256 -> 128) để Critic đủ thông minh
        # XÓA TOÀN BỘ DROPOUT để bảo toàn Lipschitz Continuity cho WGAN-GP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512), 
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1) # Đầu ra Score thực
        )

    def forward(self, x):
        return self.net(x)