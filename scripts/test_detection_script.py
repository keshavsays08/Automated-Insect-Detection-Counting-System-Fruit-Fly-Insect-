#!/usr/bin/env python3
"""
Test your trained fruit fly detection model
"""

from ultralytics import YOLO
import cv2
from picamera2 import Picamera2
import time

# Configuration
MODEL_PATH = "models/fruitfly_detector/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5

class FruitFlyTester:
    def __init__(self):
        print("Loading model...")
        self.model = YOLO(MODEL_PATH)
        
        print("Initializing camera...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)
        
        print("\n" + "="*60)
        print("FRUIT FLY DETECTOR - TESTING MODE")
        print("="*60)
        print("Press 'Q' to quit")
        print("="*60 + "\n")
    
    def run(self):
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Run detection
                results = self.model(frame_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Count detections
                num_flies = len(results[0].boxes)
                
                # Add count overlay
                cv2.putText(annotated_frame, f"Fruit Flies: {num_flies}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow('Fruit Fly Detection Test', annotated_frame)
                
                # Print detections
                if num_flies > 0:
                    for i, box in enumerate(results[0].boxes):
                        conf = box.conf[0].item()
                        print(f"Fly {i+1}: Confidence {conf:.2f}")
                
                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cv2.destroyAllWindows()
            self.picam2.stop()
            print("\nTest complete!")

if __name__ == "__main__":
    tester = FruitFlyTester()
    tester.run()
