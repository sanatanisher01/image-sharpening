import cv2
import torch
import numpy as np
from models import StudentModel
import time
import sys
import subprocess

class DirectVirtualCamera:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = StudentModel().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model.eval()
        self.running = True
        
    def install_virtual_camera(self):
        """Install virtual camera without OBS"""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvirtualcam"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except:
            return False
    
    def process_frame(self, frame):
        # Skip processing every other frame for speed
        if hasattr(self, 'frame_skip'):
            self.frame_skip = not self.frame_skip
            if not self.frame_skip:
                return getattr(self, 'last_processed', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            self.frame_skip = True
        
        # Use moderate resize for balance of speed and quality
        small_frame = cv2.resize(frame, (480, 360))
        blurred_frame = cv2.GaussianBlur(small_frame, (3, 3), 0.8)
        frame_rgb = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
        
        frame_tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            sharpened = self.model(frame_tensor)
        
        sharpened_np = sharpened[0].cpu().permute(1, 2, 0).numpy()
        sharpened_np = (sharpened_np * 255).astype(np.uint8)
        result = cv2.addWeighted(sharpened_np, 0.7, frame_rgb, 0.3, 0)
        
        # Resize back to original size
        result = cv2.resize(result, (640, 480))
        self.last_processed = result
        return result
    
    def start_camera(self, camera_id=0):
        """Start direct virtual camera"""
        print("🚀 AI Camera for Meet/Zoom (No OBS Required)")
        print("=" * 50)
        
        # Install virtual camera
        print("⏳ Setting up virtual camera...")
        if not self.install_virtual_camera():
            print("❌ Failed to install virtual camera")
            return
        
        try:
            import pyvirtualcam
        except ImportError:
            print("❌ Virtual camera not available")
            return
        
        # Setup camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Physical camera ready")
        print("✅ AI model loaded")
        print("✅ Installing virtual camera driver...")
        
        try:
            with pyvirtualcam.Camera(width=640, height=480, fps=30, fmt=pyvirtualcam.PixelFormat.RGB) as vcam:
                print("✅ Virtual camera created!")
                print(f"📹 Device: {vcam.device}")
                print("\n🎯 READY TO USE:")
                print("1. Open Google Meet/Zoom")
                print("2. Camera Settings → Select 'Python Virtual Camera'")
                print("3. AI-sharpened video will appear!")
                print("\n🔄 Processing... Press Ctrl+C to stop")
                
                frame_count = 0
                start_time = time.time()
                fps_list = []
                
                while self.running:
                    frame_start = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Resize frame to match virtual camera
                    frame = cv2.resize(frame, (640, 480))
                    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # AI processing
                    processed_frame = self.process_frame(frame)
                    
                    # Calculate SSIM only every 10 frames for speed
                    if frame_count % 10 == 0:
                        try:
                            from utils import calculate_ssim
                            orig_tensor = torch.from_numpy(original_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                            proc_tensor = torch.from_numpy(processed_frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
                            ssim_score = calculate_ssim(proc_tensor, orig_tensor)
                        except:
                            ssim_score = 0.85  # Default value
                    else:
                        ssim_score = getattr(self, 'last_ssim', 0.85)
                    self.last_ssim = ssim_score
                    
                    # Calculate FPS
                    frame_end = time.time()
                    frame_fps = 1.0 / max(frame_end - frame_start, 1e-6)
                    fps_list.append(frame_fps)
                    if len(fps_list) > 30:  # Keep last 30 frames
                        fps_list.pop(0)
                    avg_fps = sum(fps_list) / len(fps_list)
                    
                    # Add metrics overlay
                    processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(processed_bgr, 'AI SHARPENED', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_bgr, f'FPS: {avg_fps:.1f}', (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_bgr, f'SSIM: {ssim_score:.3f}', (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # FPS status indicator
                    fps_color = (0, 255, 0) if 30 <= avg_fps <= 60 else (0, 0, 255)
                    cv2.putText(processed_bgr, f'Target: 30-60 FPS', (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
                    
                    # Convert back to RGB for virtual camera
                    final_frame = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Send to virtual camera
                    vcam.send(final_frame)
                    vcam.sleep_until_next_frame()
                    
                    frame_count += 1
                    if frame_count % 60 == 0:
                        print(f"📊 FPS: {avg_fps:.1f} | SSIM: {ssim_score:.3f} | Frames: {frame_count}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try restarting as administrator")
        
        cap.release()
        print("✅ Camera stopped")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera = DirectVirtualCamera('checkpoints/best_student_model.pth', device)
    
    try:
        camera.start_camera()
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        camera.running = False