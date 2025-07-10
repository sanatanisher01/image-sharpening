#!/usr/bin/env python3
"""
Complete pipeline for Image Sharpening using Knowledge Distillation
Runs training, evaluation, and generates comprehensive results
"""

import os
import torch
import argparse
from train import KnowledgeDistillationTrainer
from evaluate import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description='Image Sharpening Pipeline')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both',
                       help='Run training, evaluation, or both')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--device', default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    if args.mode in ['train', 'both']:
        print("="*50)
        print("STARTING TRAINING PHASE")
        print("="*50)
        
        trainer = KnowledgeDistillationTrainer('BSRGAN.pth', device)
        trainer.train('data/train', 'data/val', epochs=args.epochs)
        
        print("Training completed!")
    
    if args.mode in ['eval', 'both']:
        print("="*50)
        print("STARTING EVALUATION PHASE")
        print("="*50)
        
        if not os.path.exists('checkpoints/best_student_model.pth'):
            print("Error: No trained model found. Please run training first.")
            return
        
        evaluator = ModelEvaluator('checkpoints/best_student_model.pth', device)
        results = evaluator.generate_report('data/val')
        
        # Save results
        import json
        with open('results/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Evaluation completed! Results saved to results/evaluation_results.json")
        
        # Print final summary
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        print(f"✓ Average SSIM: {results['avg_ssim']:.4f}")
        print(f"✓ Average FPS: {results['avg_fps']:.1f}")
        print(f"✓ SSIM Target (>0.9): {'ACHIEVED' if results['target_ssim_achieved'] else 'NOT ACHIEVED'}")
        print(f"✓ FPS Target (30-60): {'ACHIEVED' if results['target_fps_achieved'] else 'NOT ACHIEVED'}")
        
        if results['target_ssim_achieved'] and results['target_fps_achieved']:
            print("\n🎉 ALL TARGETS ACHIEVED! Model ready for deployment.")
        else:
            print("\n⚠️  Some targets not met. Consider additional training or model optimization.")

if __name__ == '__main__':
    main()