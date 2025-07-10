import torch
import cv2
import numpy as np
import time
from models import StudentModel
from utils import load_and_preprocess_image, save_image_tensor

class RealTimeSharpener:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = StudentModel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
    def sharpen_image(self, image_path, output_path=None):
        """Sharpen a single image"""
        # Load and preprocess
        img_tensor = load_and_preprocess_image(image_path).to(self.device)
        
        # Inference
        with torch.no_grad():
            start_time = time.time()
            sharpened = self.model(img_tensor)
            end_time = time.time()
        
        inference_time = end_time - start_time
        fps = 1.0 / inference_time
        
        # Save result
        if output_path:
            save_image_tensor(sharpened, output_path)
        
        print(f"Image sharpened in {inference_time:.4f}s ({fps:.1f} FPS)")
        return sharpened, fps
    
    def process_video_stream(self, input_source=0, output_path=None):
        """Process video stream in real-time"""
        cap = cv2.VideoCapture(input_source)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        fps_counter = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                sharpened_tensor = self.model(frame_tensor)
            end_time = time.time()
            
            # Convert back to OpenCV format
            sharpened_np = sharpened_tensor.squeeze(0).detach().cpu().numpy()
            sharpened_np = np.transpose(sharpened_np, (1, 2, 0))
            sharpened_np = (sharpened_np * 255).astype(np.uint8)
            sharpened_frame = cv2.cvtColor(sharpened_np, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            inference_time = max(end_time - start_time, 1e-6)
            fps = 1.0 / inference_time
            fps_counter.append(fps)
            
            # Display FPS on frame
            cv2.putText(sharpened_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show result
            cv2.imshow('Sharpened Video', sharpened_frame)
            
            # Save frame if output specified
            if output_path:
                out.write(sharpened_frame)
            
            # Exit on any key press
            key = cv2.waitKey(30) & 0xFF  # Increased wait time
            if key != 255:  # Any key pressed
                print(f"Key pressed: {key}")
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        avg_fps = np.mean(fps_counter)
        print(f"Average FPS: {avg_fps:.1f}")
        return avg_fps

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sharpener = RealTimeSharpener('checkpoints/best_student_model.pth', device)
    
    # Example usage
    print("Real-time Image Sharpener")
    print("1. Process single image")
    print("2. Process video stream")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '1':
        image_path = input("Enter image path: ")
        output_path = input("Enter output path (optional): ") or None
        sharpener.sharpen_image(image_path, output_path)
    
    elif choice == '2':
        print("Starting video stream processing...")
        print("Press 'q' to quit")
        sharpener.process_video_stream()

if __name__ == '__main__':
    main()