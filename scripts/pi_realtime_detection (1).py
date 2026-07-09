#!/usr/bin/env python3
"""
Real-Time Fruit Fly Detection and Counting for Raspberry Pi
Uses HQ Camera and YOLO model for live detection
Logs detections to CSV with timestamps
"""

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
import time
from datetime import datetime
import csv
from pathlib import Path
import os

class RealTimeFruitFlyDetector:
    def __init__(self, model_path="models/best.pt", output_dir="detection_logs"):
        """
        Initialize real-time detector
        
        Args:
            model_path: Path to trained YOLO model
            output_dir: Directory for logs and captured images
        """
        print("="*60)
        print("FRUIT FLY REAL-TIME DETECTION SYSTEM")
        print("="*60)
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = YOLO(model_path)
        print("✓ Model loaded successfully")
        
        # Initialize camera
        print("\nInitializing camera...")
        self.picam2 = Picamera2()
        
        # Camera configuration optimized for detection
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Camera warm-up
        print("✓ Camera initialized (640x480)")
        
        # Setup output directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.captures_dir = self.output_dir / "captured_detections"
        self.captures_dir.mkdir(exist_ok=True)
        
        # Session info
        self.session_start = datetime.now()
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # CSV logging
        self.csv_path = self.output_dir / f"detection_log_{session_id}.csv"
        self._initialize_csv()
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.capture_interval = 60  # Save snapshot every 60 seconds
        self.last_capture_time = time.time()
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
        self.detection_history = []
        
        print(f"✓ Logging to: {self.csv_path}")
        print("="*60)
    
    def _initialize_csv(self):
        """Create CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Frame_Number',
                'Fruit_Fly_Count',
                'Average_Confidence',
                'Max_Confidence',
                'Min_Confidence',
                'Processing_Time_ms',
                'Snapshot_Saved'
            ])
    
    def _log_detection(self, frame_num, count, confidences, proc_time, snapshot_saved=False):
        """Log detection to CSV"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if count > 0:
            avg_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            min_conf = np.min(confidences)
        else:
            avg_conf = max_conf = min_conf = 0
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                frame_num,
                count,
                f"{avg_conf:.3f}" if count > 0 else "N/A",
                f"{max_conf:.3f}" if count > 0 else "N/A",
                f"{min_conf:.3f}" if count > 0 else "N/A",
                f"{proc_time:.2f}",
                "Yes" if snapshot_saved else "No"
            ])
    
    def detect_frame(self, frame):
        """Run detection on single frame"""
        start_time = time.time()
        
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Extract results
        boxes = results[0].boxes
        count = len(boxes)
        
        # Get confidences
        confidences = []
        if count > 0:
            confidences = boxes.conf.cpu().numpy()
        
        # Draw annotations
        annotated_frame = results[0].plot()
        
        return {
            'annotated': annotated_frame,
            'count': count,
            'confidences': confidences,
            'processing_time': processing_time,
            'results': results[0]
        }
    
    def _add_overlay(self, frame, count, avg_conf, fps, session_time):
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Detection count (large, prominent)
        cv2.putText(frame, f"FRUIT FLIES: {count}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Confidence
        if count > 0:
            cv2.putText(frame, f"Confidence: {avg_conf:.1%}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Session info
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Session: {session_time}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Total detections
        cv2.putText(frame, f"Total frames: {self.frame_count}", (w - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _save_snapshot(self, frame, count):
        """Save snapshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}_count{count}.jpg"
        filepath = self.captures_dir / filename
        cv2.imwrite(str(filepath), frame)
        return True
    
    def run(self):
        """Main detection loop"""
        print("\nStarting real-time detection...")
        print("Controls:")
        print("  S - Save current snapshot manually")
        print("  Q - Quit and show statistics")
        print("  SPACE - Pause/Resume")
        print("\nDetecting...\n")
        
        paused = False
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        try:
            while True:
                # FPS calculation
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Capture frame
                frame = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if not paused:
                    # Run detection
                    result = self.detect_frame(frame_bgr)
                    
                    self.frame_count += 1
                    count = result['count']
                    
                    # Calculate average confidence
                    avg_conf = np.mean(result['confidences']) if count > 0 else 0
                    
                    # Add overlay
                    session_elapsed = str(datetime.now() - self.session_start).split('.')[0]
                    self._add_overlay(
                        result['annotated'], 
                        count, 
                        avg_conf,
                        current_fps,
                        session_elapsed
                    )
                    
                    # Auto-save snapshot every minute
                    snapshot_saved = False
                    if time.time() - self.last_capture_time >= self.capture_interval:
                        snapshot_saved = self._save_snapshot(result['annotated'], count)
                        self.last_capture_time = time.time()
                        print(f"[Auto-saved] Frame {self.frame_count}: {count} flies")
                    
                    # Log detection
                    self._log_detection(
                        self.frame_count,
                        count,
                        result['confidences'],
                        result['processing_time'],
                        snapshot_saved
                    )
                    
                    # Update statistics
                    if count > 0:
                        self.total_detections += count
                    
                    display_frame = result['annotated']
                else:
                    # Show paused overlay
                    display_frame = frame_bgr.copy()
                    cv2.putText(display_frame, "PAUSED", (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                # Display
                cv2.imshow('Fruit Fly Detection - Live', display_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                elif key == ord(' '):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"\n[{status}]\n")
                
                elif key == ord('s'):
                    if not paused:
                        self._save_snapshot(display_frame, count)
                        print(f"[Manual save] Frame {self.frame_count}: {count} flies")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and show statistics"""
        cv2.destroyAllWindows()
        self.picam2.stop()
        
        # Calculate statistics
        session_duration = datetime.now() - self.session_start
        
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        print(f"Session duration: {session_duration}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total fly detections: {self.total_detections}")
        
        if self.frame_count > 0:
            avg_per_frame = self.total_detections / self.frame_count
            print(f"Average flies per frame: {avg_per_frame:.2f}")
        
        print(f"\nData saved to:")
        print(f"  CSV log: {self.csv_path}")
        print(f"  Snapshots: {self.captures_dir}")
        print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check if model exists
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"✗ Error: Model not found at {model_path}")
        print("Please copy best.pt to models/ folder")
        sys.exit(1)
    
    # Create and run detector
    detector = RealTimeFruitFlyDetector(model_path)
    detector.run()
