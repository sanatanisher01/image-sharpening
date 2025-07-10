import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from models import TeacherModel, StudentModel
from dataset import get_dataloaders
from utils import calculate_ssim, AverageMeter
import os

class KnowledgeDistillationTrainer:
    def __init__(self, teacher_path, device='cuda'):
        self.device = device
        self.teacher = TeacherModel(teacher_path).to(device)
        self.student = StudentModel().to(device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        # Metrics
        self.best_ssim = 0.0
        
    def distillation_loss(self, student_output, teacher_output, target, alpha=0.7):
        """Knowledge distillation loss combining L1 and MSE"""
        # Task loss (reconstruction)
        task_loss = self.l1_loss(student_output, target)
        
        # Feature matching loss (distillation)
        distill_loss = self.mse_loss(student_output, teacher_output)
        
        return alpha * task_loss + (1 - alpha) * distill_loss
    
    def train_epoch(self, train_loader, epoch):
        self.student.train()
        losses = AverageMeter()
        ssim_scores = AverageMeter()
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
            
            # Teacher forward pass
            with torch.no_grad():
                teacher_output = self.teacher(lr_imgs)
            
            # Student forward pass
            student_output = self.student(lr_imgs)
            
            # Calculate loss
            loss = self.distillation_loss(student_output, teacher_output, hr_imgs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            losses.update(loss.item(), lr_imgs.size(0))
            ssim = calculate_ssim(student_output, hr_imgs)
            ssim_scores.update(ssim, lr_imgs.size(0))
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {losses.avg:.4f}, SSIM: {ssim_scores.avg:.4f}')
        
        return losses.avg, ssim_scores.avg
    
    def validate(self, val_loader):
        self.student.eval()
        ssim_scores = AverageMeter()
        fps_scores = AverageMeter()
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
                
                # Measure FPS
                start_time = time.time()
                student_output = self.student(lr_imgs)
                end_time = time.time()
                
                inference_time = max(end_time - start_time, 1e-6)  # Prevent division by zero
                fps = 1.0 / inference_time
                fps_scores.update(fps, lr_imgs.size(0))
                
                # Calculate SSIM
                ssim = calculate_ssim(student_output, hr_imgs)
                ssim_scores.update(ssim, lr_imgs.size(0))
        
        return ssim_scores.avg, fps_scores.avg
    
    def train(self, train_dir, val_dir, epochs=100, save_dir='checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter('logs')
        
        train_loader, val_loader = get_dataloaders(train_dir, val_dir)
        
        for epoch in range(epochs):
            # Training
            train_loss, train_ssim = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_ssim, val_fps = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('SSIM/Train', train_ssim, epoch)
            writer.add_scalar('SSIM/Val', val_ssim, epoch)
            writer.add_scalar('FPS/Val', val_fps, epoch)
            
            print(f'Epoch {epoch}: Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}, FPS: {val_fps:.1f}')
            
            # Save best model
            if val_ssim > self.best_ssim:
                self.best_ssim = val_ssim
                torch.save(self.student.state_dict(), os.path.join(save_dir, 'best_student_model.pth'))
                print(f'New best model saved with SSIM: {val_ssim:.4f}')
        
        writer.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = KnowledgeDistillationTrainer('BSRGAN.pth', device)
    trainer.train('data/train', 'data/val')