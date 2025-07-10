# Image Sharpening using Knowledge Distillation

A real-time image sharpening system for video conferencing using knowledge distillation from BSRGAN teacher model to an ultra-lightweight student model.

## Features

- **Teacher-Student Architecture**: Uses BSRGAN as teacher model and lightweight CNN as student
- **Real-time Performance**: Achieves 30-60 FPS on 1920x1080 resolution
- **High Quality**: Targets SSIM > 0.9 for enhanced image sharpness
- **Video Conferencing Optimized**: Handles low bandwidth and compression artifacts

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Train the Model**
```bash
python train.py
```

3. **Evaluate Performance**
```bash
python evaluate.py
```

4. **Real-time Inference**
```bash
python inference.py
```

## Model Architecture

### Teacher Model (BSRGAN)
- Pre-trained high-performance image super-resolution model
- Provides high-quality reference outputs for knowledge distillation

### Student Model (Lightweight CNN)
- Ultra-lightweight architecture with only 32-64 feature channels
- 4 residual blocks for efficient feature extraction
- Optimized for real-time inference

## Training Process

1. **Data Preparation**: Images are degraded using bicubic downscaling and noise addition
2. **Knowledge Distillation**: Student learns from both ground truth and teacher outputs
3. **Loss Function**: Combines L1 reconstruction loss and KL divergence distillation loss
4. **Optimization**: Adam optimizer with learning rate scheduling

## Performance Targets

- **SSIM Score**: > 0.9 (90% structural similarity)
- **Frame Rate**: 30-60 FPS at 1920x1080 resolution
- **Model Size**: Ultra-lightweight for real-time deployment
- **Quality**: Enhanced sharpness for video conferencing

## File Structure

```
├── models.py          # Teacher and Student model definitions
├── dataset.py         # Data loading and preprocessing
├── train.py          # Training script with knowledge distillation
├── evaluate.py       # Comprehensive evaluation and benchmarking
├── inference.py      # Real-time inference for images and video
├── utils.py          # Utility functions (SSIM, FPS benchmarking)
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Usage Examples

### Single Image Sharpening
```python
from inference import RealTimeSharpener

sharpener = RealTimeSharpener('checkpoints/best_student_model.pth')
sharpener.sharpen_image('input.jpg', 'output.jpg')
```

### Video Stream Processing
```python
# Process webcam feed
sharpener.process_video_stream(input_source=0)

# Process video file
sharpener.process_video_stream(input_source='video.mp4', output_path='sharpened_video.avi')
```

## Evaluation Metrics

- **SSIM (Structural Similarity Index)**: Measures perceptual image quality
- **FPS (Frames Per Second)**: Real-time performance metric
- **Visual Quality Assessment**: Subjective evaluation on diverse image categories

## Expected Results

- SSIM scores consistently above 0.9
- Real-time performance at 30-60 FPS
- Significant improvement in video conferencing image quality
- Robust performance across different image categories (text, nature, people, etc.)