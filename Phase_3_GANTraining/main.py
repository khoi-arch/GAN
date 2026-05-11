import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import numpy as np
import random

# Import kiến trúc và Loss từ các file cùng thư mục
from models import IDSGenerator, Discriminator
from losses import WGANLoss, CovarianceMatchingLoss, SemanticRegularizationLoss, compute_gradient_penalty

warnings.filterwarnings('ignore')

def set_global_seed(seed=42):
    """Giữ vững tính Deterministic cho quá trình Huấn luyện"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data(data_dir):
    print("[1/4] Đang nạp Tensors...")
    malware_tensor = torch.load(os.path.join(data_dir, "tensor_malware.pt"))
    benign_tensor = torch.load(os.path.join(data_dir, "tensor_benign.pt"))
    return malware_tensor, benign_tensor

def train_gan_experiment(data_dir="../data_artifacts", epochs=200, batch_size=128):
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Đang chạy trên thiết bị: {device}")
    
    # 1. Load Data
    X_malware, X_benign = load_data(data_dir)
    input_dim = X_malware.shape[1]
    noise_dim = 32
    
    # DataLoader
    malware_dataset = TensorDataset(X_malware)
    malware_loader = DataLoader(malware_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    benign_dataset = TensorDataset(X_benign)
    benign_loader = DataLoader(benign_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Khởi tạo Models
    print("[2/4] Khởi tạo Generator (Masked) & Discriminator (WGAN Critic)...")
    policy_path = os.path.join(data_dir, "adversarial_policy.json")
    
    generator = IDSGenerator(input_dim=input_dim, noise_dim=noise_dim, policy_path=policy_path).to(device)
    discriminator = Discriminator(input_dim=input_dim).to(device)
    
    # Optimizers: WGAN-GP khuyên dùng Adam với learning rate nhỏ và beta1=0.0
    opt_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    # 3. Khởi tạo Losses
    print("[3/4] Khởi tạo toàn bộ Hệ thống Loss...")
    wgan_loss = WGANLoss()
    loss_cov = CovarianceMatchingLoss().to(device)
    loss_semantic = SemanticRegularizationLoss(policy_path, device=device)
    
    # Trọng số của các hàm Loss (Hyperparameters cực kỳ quan trọng)
    lambda_gp = 10.0      # Gradient Penalty (Cố định của WGAN-GP)
    lambda_cov = 5.0      # Giữ cấu trúc Manifold
    lambda_l1 = 2.0       # Ép Sparsity (Càng cao GAN càng lười sửa)
    lambda_sat = 10.0     # Phạt nặng Tanh Saturation
    n_critic = 5          # Train Critic 5 lần / Generator 1 lần

    print("[4/4] Bắt đầu quá trình Huấn luyện Cường độ cao...")
    
    generator.train()
    discriminator.train()
    
    benign_iter = iter(benign_loader)
    
    for epoch in range(epochs):
        for i, (real_malware,) in enumerate(malware_loader):
            real_malware = real_malware.to(device)
            current_batch_size = real_malware.size(0)
            
            # Lấy data Benign (Vòng lặp vô tận)
            try:
                real_benign = next(benign_iter)[0].to(device)
            except StopIteration:
                benign_iter = iter(benign_loader)
                real_benign = next(benign_iter)[0].to(device)
                
            # ==========================================
            # HUẤN LUYỆN DISCRIMINATOR (CRITIC)
            # ==========================================
            for _ in range(n_critic):
                opt_D.zero_grad()
                
                # Generator sinh Fake Malware
                z = torch.randn(current_batch_size, noise_dim).to(device)
                fake_malware, _ = generator(real_malware, z)
                
                # Chấm điểm (Không cần detach vì Critic sẽ backward riêng)
                score_real = discriminator(real_benign)
                score_fake = discriminator(fake_malware.detach())
                
                # WGAN Critic Loss
                c_loss = wgan_loss.get_critic_loss(score_real, score_fake)
                
                # Gradient Penalty
                gp = compute_gradient_penalty(discriminator, real_benign, fake_malware.detach(), device)
                
                # Tổng D Loss
                d_loss = c_loss + lambda_gp * gp
                d_loss.backward()
                opt_D.step()

            # ==========================================
            # HUẤN LUYỆN GENERATOR
            # ==========================================
            opt_G.zero_grad()
            
            # Sinh lại Fake Malware để truyền Gradient qua Generator
            z = torch.randn(current_batch_size, noise_dim).to(device)
            fake_malware, perturbation = generator(real_malware, z)
            
            # 1. Adversarial Loss (Lừa Critic chấm điểm cao)
            score_fake = discriminator(fake_malware)
            g_adv_loss = wgan_loss.get_generator_loss(score_fake)
            
            # 2. Covariance Loss (Giữ tương quan đặc trưng)
            g_cov_loss = loss_cov(real_malware, fake_malware)
            
            # 3. Semantic Regularization (L1 & Saturation)
            l1_loss, sat_loss = loss_semantic(perturbation)
            
            # TỔNG LỰC HÀM LOSS CHO GENERATOR
            g_total_loss = (g_adv_loss) + \
                           (lambda_cov * g_cov_loss) + \
                           (lambda_l1 * l1_loss) + \
                           (lambda_sat * sat_loss)
            
            g_total_loss.backward()
            opt_G.step()
            
        # In log tiến độ mỗi 10 epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs}] | "
                  f"D_Loss: {d_loss.item():.4f} | "
                  f"G_Adv: {g_adv_loss.item():.4f} | "
                  f"Cov: {g_cov_loss.item():.4f} | "
                  f"L1: {l1_loss.item():.4f} | "
                  f"Sat: {sat_loss.item():.4f}")

    print("✅ Hoàn tất huấn luyện. Generator đã nắm vững Bản đồ Tác chiến!")
    
    # Save Model Weights
    torch.save(generator.state_dict(), os.path.join(data_dir, "generator_weights.pth"))
    
    # Sinh ra một mẻ Fake Malware cuối cùng để lưu lại
    generator.eval()
    with torch.no_grad():
        z_final = torch.randn(X_malware.shape[0], noise_dim).to(device)
        final_fake_malware, _ = generator(torch.FloatTensor(X_malware).to(device), z_final)
        torch.save(final_fake_malware.cpu(), os.path.join(data_dir, "tensor_fake_malware.pt"))
    print(f"   -> Đã lưu Fake Malware Tensor tại: {os.path.join(data_dir, 'tensor_fake_malware.pt')}")

if __name__ == "__main__":
    train_gan_experiment()