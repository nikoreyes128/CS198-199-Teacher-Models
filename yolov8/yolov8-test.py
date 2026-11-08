import torch
import json
import os
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load the vanilla COCO-pretrained YOLOv8s model
model = YOLO('yolov8s.pt')

# 1. Print Parameters and FLOPs (YOLO natively calculates this on load)
model.info(detailed=False)

# Paths pointing directly to the valid folder based on Roboflow COCO export
annotation_file = '../dataset/ExDark-coco/valid/_annotations.coco.json' 
image_dir = '../dataset/ExDark-coco/valid/' 

# Format -> YOLO COCO ID: ExDark JSON ID
coco_to_exdark = {
    1: 1,   # bicycle -> Bicycle
    8: 2,   # boat -> Boat
    39: 3,  # bottle -> Bottle
    5: 4,   # bus -> Bus
    15: 5,  # cat -> Cat
    41: 6,  # cup -> Cup
    3: 7,   # motorcycle -> Motorbike
    0: 8,   # person -> People
    60: 9,  # dining table -> Table
    2: 10,  # car -> car
    56: 11, # chair -> chair
    16: 12  # dog -> dog
}

coco_gt = COCO(annotation_file)
results = []

total_inference_time_ms = 0.0
total_images = len(coco_gt.getImgIds())

print("Running YOLOv8 inference and mapping classes...")

# Loop through every image in the dataset
for img_id in coco_gt.getImgIds():
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    
    # Run Inference
    prediction = model.predict(img_path, verbose=False)[0]
    
    # Accumulate the raw inference time (excluding pre/post-processing)
    total_inference_time_ms += prediction.speed['inference']
    
    # Convert YOLO outputs to COCO format [x, y, w, h] and map the class IDs
    for box in prediction.boxes:
        coco_class_id = int(box.cls.item())
        
        # Only process the box if it is one of the 12 ExDark classes
        if coco_class_id in coco_to_exdark:
            exdark_class_id = coco_to_exdark[coco_class_id]
            
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            width = xmax - xmin
            height = ymax - ymin
            
            results.append({
                "image_id": img_id,
                "category_id": exdark_class_id,
                "bbox": [xmin, ymin, width, height],
                "score": box.conf.item()
            })

# Calculate overall FPS
# Convert ms to seconds, then divide total images by total seconds
total_inference_time_sec = total_inference_time_ms / 1000.0
fps = total_images / total_inference_time_sec if total_inference_time_sec > 0 else 0

print("\n--- Hardware Metrics ---")
print(f"Total Inference Time: {total_inference_time_sec:.2f} seconds")
print(f"Average FPS: {fps:.2f} frames/sec\n")

# Save results to a temporary JSON
output_json = "yolo_mapped_predictions.json"
with open(output_json, "w") as f:
    json.dump(results, f)

print("Inference complete. Calculating Phase 0 Baseline metrics...")

# Evaluate against Ground Truth
coco_dt = coco_gt.loadRes(output_json)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()