"""
CLI utility: detect faces in an image and save cropped faces.
Uses OpenCV Haar Cascade for face detection.
Usage:
    python detect_and_crop.py --image "path/to/image.jpg" --outdir "output_folder"
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
import sys

def detect_faces_opencv(pil_img: Image.Image):
    """Detect faces using OpenCV Haar Cascade"""
    try:
        # Convert PIL image to OpenCV format
        cv_image = np.array(pil_img)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Load Haar cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Convert coordinates to (x1, y1, x2, y2) format
        return [[int(x), int(y), int(x + w), int(y + h)] for (x, y, w, h) in faces]
    
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Detect faces and save crops")
    parser.add_argument("--image", type=str, required=True, help="Path to image (jpg/png)")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory for crops")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found -> {image_path}")
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    crops_dir = outdir / f"{image_path.stem}_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)
        
    boxes = detect_faces_opencv(pil_img)

    # Create annotated image
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    
    # Save annotated image
    ann_path = crops_dir / "annotated.jpg"
    annotated.save(ann_path, quality=95)

    # Save face crops
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = pil_img.crop((x1, y1, x2, y2))
        crop.save(crops_dir / f"face_{i+1:03d}.jpg", quality=95)

    print(f"✅ Done! Detected {len(boxes)} face(s).")
    print(f"📁 Annotated image: {ann_path}")
    print(f"📁 Crops folder: {crops_dir}")
    
    if boxes:
        for i in range(len(boxes)):
            print(f"   - face_{i+1:03d}.jpg")

if __name__ == "__main__":
    main()