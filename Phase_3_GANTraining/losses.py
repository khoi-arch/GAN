import torch
import torch.nn as nn
import json

class WGANLoss(nn.Module):
    """
    Hàm Loss cơ bản của WGAN.
    Critic: max D(real) - D(fake) <=> min D(fake) - D(real)
    Generator: max D(fake) <=> min -D(fake)
    """
    def __init__(self):
        super(WGANLoss, self).__init__()

    def get_critic_loss(self, score_real, score_fake):
        return torch.mean(score_fake) - torch.mean(score_real)
        
    def get_generator_loss(self, score_fake):
        return -torch.mean(score_fake)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Tính Gradient Penalty cho WGAN-GP để chống Mode Collapse và duy trì 1-Lipschitz.
    """
    # Lấy alpha ngẫu nhiên để nội suy giữa real và fake
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand(real_samples.size())
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake_targets = torch.ones(real_samples.size(0), 1).to(device)
    
    # Tính Gradient của D đối với các điểm nội suy
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_targets,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    # Tính Norm bậc 2 và ép nó về gần 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

class CovarianceMatchingLoss(nn.Module):
    """
    Ép Ma trận hiệp phương sai của Fake Malware phải giống hệt Real Malware.
    Giữ vững cấu trúc Semantic Manifold.
    """
    def __init__(self):
        super(CovarianceMatchingLoss, self).__init__()

    def forward(self, real_data, fake_data):
        real_mean = torch.mean(real_data, dim=0, keepdim=True)
        fake_mean = torch.mean(fake_data, dim=0, keepdim=True)
        
        real_centered = real_data - real_mean
        fake_centered = fake_data - fake_mean
        
        n_samples = real_data.size(0)
        cov_real = torch.mm(real_centered.t(), real_centered) / (n_samples - 1 + 1e-8)
        cov_fake = torch.mm(fake_centered.t(), fake_centered) / (n_samples - 1 + 1e-8)
        
        loss = torch.mean((cov_real - cov_fake) ** 2)
        return loss

class SemanticRegularizationLoss(nn.Module):
    """
    Chứa 2 Loss cực mạnh cho Generator:
    1. L1 Perturbation Loss: Ép GAN sinh nhiễu thưa thớt (Minimal edits).
    2. Soft Policy Saturation Loss: Phạt Tanh/Softsign nếu nó đẩy sát mức giới hạn của Mask.
    """
    def __init__(self, policy_path):
        super(SemanticRegularizationLoss, self).__init__()
        
        with open(policy_path, 'r', encoding='utf-8') as f:
            policy = json.load(f)
            
        total_features = policy['metadata']['total_features']
        
        # Khởi tạo tensor cục bộ (sẽ chuyển thành buffer)
        max_vars = torch.ones(total_features)
        
        # Nạp giới hạn variance từ file mapping
        for zone_name, zone_data in policy['zones'].items():
            max_var = zone_data['allowed_variance']
            for idx in zone_data['features']:
                max_vars[idx] = max_var
                
        # [FIX QUAN TRỌNG]: Đăng ký thành Buffer để Tensor tự động move to(GPU)
        # và được lưu trong state_dict của mô hình.
        self.register_buffer('max_variances', max_vars)
        self.register_buffer('safe_variances', max_vars + 1e-8)

    def forward(self, perturbation):
        """
        perturbation: Là scaled_perturbation (đã nhân mask) từ Generator.
        """
        # 1. L1 Loss (Sparsity / Feature Drift)
        # Ép phần lớn các cột nhiễu về 0. Đồng thời đóng vai trò như Feature Drift Loss.
        l1_loss = torch.mean(torch.abs(perturbation))
        
        # 2. Soft Saturation Loss (Anti-Saturation)
        # Tính tỷ lệ sử dụng ngân sách: ratio = |nhiễu| / max_variance
        # Bình phương ratio để phạt cực nặng vùng CRITICAL nếu dám nhúc nhích.
        ratio = torch.abs(perturbation) / self.safe_variances
        saturation_loss = torch.mean(ratio ** 2)
        
        return l1_loss, saturation_loss