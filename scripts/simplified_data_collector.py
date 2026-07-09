#!/usr/bin/env python3
"""
Simple Insect Data Collection Script for Raspberry Pi
Place this in: ~/Desktop/insect_counting/scripts/
Run with: source ../venv/bin/activate && python3 collect_data.py
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
from datetime import datetime

# Configuration
SAVE_DIR = "../dataset/raw_images"
# HQ Camera supports higher resolution - using 1920x1080 for better quality
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

class SimpleDataCollector:
    def __init__(self):
        # Create save directory
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        print("Initializing camera...")
        
        # Initialize HQ Camera with optimized settings
        self.picam2 = Picamera2()
        
        # HQ Camera configuration for insect detection
        config = self.picam2.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
            controls={
                "AfMode": 2,  # Continuous autofocus
                "AfSpeed": 1,  # Fast autofocus
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        time.sleep(2)  # Camera warm-up
        
        self.count = 0
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Camera ready!")
        print("\n" + "="*50)
        print("CONTROLS:")
        print("  SPACE BAR - Capture single image")
        print("  A - Auto capture (1 per second)")
        print("  S - Stop auto capture")
        print("  Q - Quit")
        print("="*50 + "\n")
    
    def capture_single(self):
        """Capture one image"""
        frame = self.picam2.capture_array()
        
        filename = f"insect_{self.session}_{self.count:04d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, frame_bgr)
        
        self.count += 1
        print(f"✓ Saved: {filename} (Total: {self.count})")
        
        # Show brief feedback
        feedback = frame_bgr.copy()
        cv2.putText(feedback, "CAPTURED!", (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow('Collection', feedback)
        cv2.waitKey(200)  # Show for 200ms
    
    def auto_capture(self):
        """Auto capture mode - 1 image per second"""
        print("\n>>> AUTO CAPTURE MODE STARTED <<<")
        print("Press 'S' to stop\n")
        
        last_time = time.time()
        
        while True:
            frame = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add overlay
            display = frame_bgr.copy()
            cv2.putText(display, "AUTO MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Count: {self.count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press S to stop", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Collection', display)
            
            # Capture every 1 second
            if time.time() - last_time >= 1.0:
                filename = f"insect_{self.session}_{self.count:04d}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                cv2.imwrite(filepath, frame_bgr)
                self.count += 1
                print(f"✓ Auto captured: {filename}")
                last_time = time.time()
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('q'):
                print("\n>>> AUTO CAPTURE STOPPED <<<\n")
                break
    
    def run(self):
        """Main loop"""
        try:
            while True:
                # Get frame
                frame = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Create display
                display = frame_bgr.copy()
                
                # Add info text
                cv2.putText(display, f"Images: {self.count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add crosshair
                h, w = display.shape[:2]
                cv2.line(display, (w//2-20, h//2), (w//2+20, h//2), (0,255,0), 1)
                cv2.line(display, (w//2, h//2-20), (w//2, h//2+20), (0,255,0), 1)
                
                # Add controls hint
                cv2.putText(display, "SPACE:Capture | A:Auto | Q:Quit", 
                           (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                # Show frame
                cv2.imshow('Collection', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - capture
                    self.capture_single()
                    
                elif key == ord('a'):  # Auto mode
                    self.auto_capture()
                    
                elif key == ord('q'):  # Quit
                    break
        
        finally:
            print(f"\n{'='*50}")
            print(f"Session complete!")
            print(f"Total images collected: {self.count}")
            print(f"Saved to: {SAVE_DIR}")
            print(f"{'='*50}\n")
            
            cv2.destroyAllWindows()
            self.picam2.stop()

if __name__ == "__main__":
    collector = SimpleDataCollector()
    collector.run()
