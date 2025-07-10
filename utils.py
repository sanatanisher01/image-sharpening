import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_ssim(img1, img2):
    """Calculate SSIM between two image tensors"""
    # Convert to numpy and ensure proper format
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    ssim_scores = []
    for i in range(img1_np.shape[0]):  # Batch dimension
        # Convert from CHW to HWC
        im1 = np.transpose(img1_np[i], (1, 2, 0))
        im2 = np.transpose(img2_np[i], (1, 2, 0))
        
        # Get image dimensions
        h, w = im1.shape[:2]
        win_size = min(7, min(h, w))
        if win_size % 2 == 0:
            win_size -= 1
        
        # Calculate SSIM with appropriate parameters
        score = ssim(im1, im2, channel_axis=2, data_range=1.0, win_size=win_size)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)

def benchmark_fps(model, input_size=(1, 3, 1080, 1920), device='cuda', num_runs=100):
    """Benchmark model FPS"""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    if device == 'cuda':
        start_time.record()
    else:
        import time
        start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    else:
        elapsed_time = time.time() - start
    
    fps = num_runs / elapsed_time
    return fps

def load_and_preprocess_image(image_path, target_size=None):
    """Load and preprocess image for inference"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size:
        img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def save_image_tensor(tensor, path):
    """Save tensor as image"""
    # Convert tensor to numpy
    img_np = tensor.squeeze(0).detach().cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path, img_np)