#!/usr/bin/env python3
"""
Fruit Fly Detection - PC Simulator
Simulates the Raspberry Pi detection system on PC
Tests model on images with automatic timing and CSV logging
"""

import cv2
import os
from datetime import datetime
import csv
from pathlib import Path
from ultralytics import YOLO
import time

class FruitFlySimulator:
    def __init__(self, model_path, images_folder, output_folder="simulation_results"):
        """
        Initialize the simulator
        
        Args:
            model_path: Path to best.pt model file
            images_folder: Folder containing test images
            output_folder: Where to save results
        """
        # Load model
        print("Loading model...")
        self.model = YOLO(model_path)
        print(f"✓ Model loaded: {model_path}")
        
        # Setup folders
        self.images_folder = Path(images_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        self.annotated_folder = self.output_folder / "annotated_images"
        self.annotated_folder.mkdir(exist_ok=True)
        
        # Get image files
        self.image_files = sorted([
            f for f in self.images_folder.glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        print(f"✓ Found {len(self.image_files)} images")
        
        # CSV setup
        self.csv_path = self.output_folder / f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._initialize_csv()
        
        # Statistics
        self.total_detections = 0
        self.current_index = 0
        self.detection_results = []
        
        # Display settings
        self.display_time = 15  # seconds per image
        self.confidence_threshold = 0.5
        
    def _initialize_csv(self):
        """Create CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Image_Name', 
                'Fruit_Fly_Count',
                'Confidence_Avg',
                'Processing_Time_ms'
            ])
        print(f"✓ CSV log created: {self.csv_path}")
    
    def _log_detection(self, image_name, count, avg_confidence, processing_time):
        """Log detection to CSV"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                image_name,
                count,
                f"{avg_confidence:.3f}" if count > 0 else "N/A",
                f"{processing_time:.2f}"
            ])
        
        # Store for statistics
        self.detection_results.append({
            'timestamp': timestamp,
            'image': image_name,
            'count': count
        })
    
    def detect_on_image(self, image_path):
        """Run detection on single image"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Could not read {image_path}")
            return None
        
        # Run detection
        start_time = time.time()
        results = self.model(img, conf=self.confidence_threshold, verbose=False)
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Extract results
        boxes = results[0].boxes
        count = len(boxes)
        
        # Calculate average confidence
        avg_confidence = 0
        if count > 0:
            confidences = boxes.conf.cpu().numpy()
            avg_confidence = confidences.mean()
        
        # Draw annotations
        annotated_img = results[0].plot()
        
        # Add overlay information
        self._add_overlay(annotated_img, image_path.name, count, avg_confidence, processing_time)
        
        # Log to CSV
        self._log_detection(image_path.name, count, avg_confidence, processing_time)
        
        # Save annotated image
        save_path = self.annotated_folder / f"annotated_{image_path.name}"
        cv2.imwrite(str(save_path), annotated_img)
        
        return {
            'image': annotated_img,
            'count': count,
            'confidence': avg_confidence,
            'processing_time': processing_time
        }
    
    def _add_overlay(self, img, filename, count, confidence, proc_time):
        """Add information overlay to image"""
        h, w = img.shape[:2]
        
        # Semi-transparent background for text
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Add text information
        y_offset = 30
        cv2.putText(img, f"Image: {filename}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(img, f"Fruit Flies Detected: {count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset += 30
        if count > 0:
            cv2.putText(img, f"Avg Confidence: {confidence:.2%}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress indicator
        progress_text = f"Image {self.current_index + 1}/{len(self.image_files)}"
        cv2.putText(img, progress_text, (w - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def _print_statistics(self):
        """Print session statistics"""
        print("\n" + "="*60)
        print("SESSION STATISTICS")
        print("="*60)
        
        total_images = len(self.detection_results)
        total_flies = sum(r['count'] for r in self.detection_results)
        avg_per_image = total_flies / total_images if total_images > 0 else 0
        
        images_with_flies = sum(1 for r in self.detection_results if r['count'] > 0)
        detection_rate = (images_with_flies / total_images * 100) if total_images > 0 else 0
        
        print(f"Total images processed: {total_images}")
        print(f"Total fruit flies detected: {total_flies}")
        print(f"Average flies per image: {avg_per_image:.2f}")
        print(f"Images with flies: {images_with_flies} ({detection_rate:.1f}%)")
        print(f"Images without flies: {total_images - images_with_flies}")
        print(f"\nResults saved to: {self.output_folder}")
        print(f"CSV log: {self.csv_path}")
        print(f"Annotated images: {self.annotated_folder}")
        print("="*60 + "\n")
    
    def run_auto_mode(self):
        """Run automatic slideshow mode"""
        print("\n" + "="*60)
        print("AUTOMATIC MODE - PC SIMULATION")
        print("="*60)
        print(f"Display time: {self.display_time} seconds per image")
        print("Controls:")
        print("  SPACE - Next image immediately")
        print("  P     - Pause/Resume auto-advance")
        print("  Q     - Quit")
        print("="*60 + "\n")
        
        paused = False
        
        for idx, image_path in enumerate(self.image_files):
            self.current_index = idx
            
            print(f"\n[{idx+1}/{len(self.image_files)}] Processing: {image_path.name}")
            
            # Detect
            result = self.detect_on_image(image_path)
            if result is None:
                continue
            
            print(f"  → Detected: {result['count']} flies")
            print(f"  → Confidence: {result['confidence']:.2%}")
            print(f"  → Processing: {result['processing_time']:.2f}ms")
            
            # Display
            cv2.imshow('Fruit Fly Detection Simulator', result['image'])
            
            # Wait with keyboard control
            start_wait = time.time()
            while True:
                elapsed = time.time() - start_wait
                
                # Auto-advance after display_time (unless paused)
                if not paused and elapsed >= self.display_time:
                    break
                
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('q'):  # Quit
                    print("\nQuitting...")
                    cv2.destroyAllWindows()
                    self._print_statistics()
                    return
                
                elif key == ord(' '):  # Next immediately
                    break
                
                elif key == ord('p'):  # Pause/Resume
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"  [{status}]")
        
        cv2.destroyAllWindows()
        self._print_statistics()
    
    def run_manual_mode(self):
        """Run manual control mode"""
        print("\n" + "="*60)
        print("MANUAL MODE - PC SIMULATION")
        print("="*60)
        print("Controls:")
        print("  D or →  - Next image")
        print("  A or ←  - Previous image")
        print("  Q       - Quit")
        print("="*60 + "\n")
        
        while 0 <= self.current_index < len(self.image_files):
            image_path = self.image_files[self.current_index]
            
            print(f"\n[{self.current_index+1}/{len(self.image_files)}] {image_path.name}")
            
            # Detect
            result = self.detect_on_image(image_path)
            if result is None:
                self.current_index += 1
                continue
            
            print(f"  → Flies: {result['count']}")
            
            # Display
            cv2.imshow('Fruit Fly Detection Simulator', result['image'])
            
            # Wait for key
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    self._print_statistics()
                    return
                
                elif key == ord('d') or key == 83:  # D or Right arrow
                    self.current_index += 1
                    break
                
                elif key == ord('a') or key == 81:  # A or Left arrow
                    self.current_index = max(0, self.current_index - 1)
                    break
        
        cv2.destroyAllWindows()
        self._print_statistics()


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("FRUIT FLY DETECTION - PC SIMULATOR")
    print("="*60)
    
    # Configuration
    MODEL_PATH = "fruitfly_model/v1/weights/best.pt"
    IMAGES_FOLDER = "test_images"  # Put your 30 test images here
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n✗ Error: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        sys.exit(1)
    
    # Check if images folder exists
    if not os.path.exists(IMAGES_FOLDER):
        print(f"\n✗ Error: Images folder not found: {IMAGES_FOLDER}")
        print("Creating folder... Please add your test images there")
        os.makedirs(IMAGES_FOLDER)
        sys.exit(1)
    
    # Create simulator
    simulator = FruitFlySimulator(MODEL_PATH, IMAGES_FOLDER)
    
    if len(simulator.image_files) == 0:
        print(f"\n✗ Error: No images found in {IMAGES_FOLDER}")
        print("Please add .jpg, .png, or .bmp images")
        sys.exit(1)
    
    # Choose mode
    print("\nSelect mode:")
    print("1. Automatic (15 sec per image)")
    print("2. Manual (keyboard control)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        simulator.run_auto_mode()
    elif choice == "2":
        simulator.run_manual_mode()
    else:
        print("Invalid choice")
        sys.exit(1)
