import torch
import os
import numpy as np
from models import StudentModel
from utils import calculate_ssim, benchmark_fps, load_and_preprocess_image, save_image_tensor
from dataset import get_dataloaders
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = StudentModel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
    
    def evaluate_on_dataset(self, test_dir):
        """Evaluate model on test dataset"""
        _, test_loader = get_dataloaders('data/train', test_dir, batch_size=1)
        
        ssim_scores = []
        fps_scores = []
        
        print("Evaluating on test dataset...")
        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
                
                # Measure inference time
                import time
                start_time = time.time()
                output = self.model(lr_imgs)
                end_time = time.time()
                
                fps = 1.0 / (end_time - start_time)
                fps_scores.append(fps)
                
                # Calculate SSIM
                ssim_score = calculate_ssim(output, hr_imgs)
                ssim_scores.append(ssim_score)
                
                if i % 20 == 0:
                    print(f"Processed {i+1} images, Current SSIM: {ssim_score:.4f}, FPS: {fps:.1f}")
        
        avg_ssim = np.mean(ssim_scores)
        avg_fps = np.mean(fps_scores)
        
        print(f"\nFinal Results:")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"SSIM > 0.9: {np.sum(np.array(ssim_scores) > 0.9)} / {len(ssim_scores)} images")
        
        return avg_ssim, avg_fps, ssim_scores
    
    def benchmark_performance(self):
        """Benchmark model performance at different resolutions"""
        resolutions = [
            (480, 640),   # 480p
            (720, 1280),  # 720p
            (1080, 1920), # 1080p
        ]
        
        print("Benchmarking performance at different resolutions...")
        for h, w in resolutions:
            fps = benchmark_fps(self.model, (1, 3, h, w), self.device)
            print(f"Resolution {w}x{h}: {fps:.1f} FPS")
    
    def visual_comparison(self, test_dir, output_dir='results', num_samples=5):
        """Generate visual comparisons"""
        os.makedirs(output_dir, exist_ok=True)
        
        _, test_loader = get_dataloaders('data/train', test_dir, batch_size=1)
        
        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                if i >= num_samples:
                    break
                
                lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
                output = self.model(lr_imgs)
                
                # Save images
                save_image_tensor(lr_imgs, os.path.join(output_dir, f'sample_{i}_input.png'))
                save_image_tensor(output, os.path.join(output_dir, f'sample_{i}_output.png'))
                save_image_tensor(hr_imgs, os.path.join(output_dir, f'sample_{i}_target.png'))
                
                # Calculate SSIM for this sample
                ssim_score = calculate_ssim(output, hr_imgs)
                print(f"Sample {i+1} SSIM: {ssim_score:.4f}")
    
    def generate_report(self, test_dir):
        """Generate comprehensive evaluation report"""
        print("="*50)
        print("IMAGE SHARPENING MODEL EVALUATION REPORT")
        print("="*50)
        
        # Dataset evaluation
        avg_ssim, avg_fps, ssim_scores = self.evaluate_on_dataset(test_dir)
        
        # Performance benchmarking
        self.benchmark_performance()
        
        # Visual samples
        self.visual_comparison(test_dir)
        
        # Summary statistics
        print(f"\nSUMMARY:")
        print(f"- Average SSIM: {avg_ssim:.4f}")
        print(f"- Target SSIM (>0.9): {'✓ ACHIEVED' if avg_ssim > 0.9 else '✗ NOT ACHIEVED'}")
        print(f"- Average FPS: {avg_fps:.1f}")
        print(f"- Target FPS (30-60): {'✓ ACHIEVED' if 30 <= avg_fps <= 60 else '✗ NOT ACHIEVED'}")
        print(f"- Images with SSIM > 0.9: {np.sum(np.array(ssim_scores) > 0.9)} / {len(ssim_scores)}")
        
        return {
            'avg_ssim': float(avg_ssim),
            'avg_fps': float(avg_fps),
            'ssim_scores': [float(s) for s in ssim_scores],
            'target_ssim_achieved': bool(avg_ssim > 0.9),
            'target_fps_achieved': bool(30 <= avg_fps <= 60)
        }

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator('checkpoints/best_student_model.pth', device)
    results = evaluator.generate_report('data/val')