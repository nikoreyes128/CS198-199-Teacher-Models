import torch
import json
import os
import time
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image
from fvcore.nn import FlopCountAnalysis

# 1. Load the processor and the vanilla COCO-pretrained model
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

# Move model to GPU if available for accurate timing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Hardware Profiling (Parameters and FLOPs)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Parameters: {params / 1e6:.2f} Million")

# Create a dummy image tensor (Batch Size 1, 3 Channels, 640x640) to profile the FLOPs
dummy_input = torch.randn(1, 3, 640, 640).to(device)
# fvcore calculates the exact FLOPs required for one forward pass
flops = FlopCountAnalysis(model, dummy_input).total()
print(f"Computational Cost: {flops / 1e9:.2f} GFLOPs\n")

# 3. Set your paths pointing directly to the valid folder
annotation_file = '../dataset/ExDark-coco/valid/_annotations.coco.json' 
image_dir = '../dataset/ExDark-coco/valid/' 

# 4. Map standard COCO class strings to the specific ExDark JSON IDs
coco_name_to_exdark_id = {
    'bicycle': 1,
    'boat': 2,
    'bottle': 3,
    'bus': 4,
    'cat': 5,
    'cup': 6,
    'motorcycle': 7,   # ExDark calls this 'Motorbike'
    'person': 8,       # ExDark calls this 'People'
    'dining table': 9, # ExDark calls this 'Table'
    'car': 10,
    'chair': 11,
    'dog': 12
}

# 5. Load the Ground Truth (GT) annotations using pycocotools
coco_gt = COCO(annotation_file)
results = []

total_inference_time_sec = 0.0
total_images = len(coco_gt.getImgIds())

print("Running inference on ExDark images...")

# 6. Loop through every image in the dataset
for img_id in coco_gt.getImgIds():
    img_info = coco_gt.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load {img_info['file_name']}. Skipping.")
        continue
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # 7. Wrap the inference execution in a timer
    # Use torch.cuda.synchronize() for accurate GPU timing if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_inference_time_sec += (end_time - start_time)
    
    # Scale bounding boxes back to the original image size
    target_sizes = torch.tensor([image.size[::-1]])
    processed_results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.01)[0]
    
    # 8. Convert HF outputs to COCO format and map the classes
    for score, label, box in zip(processed_results["scores"], processed_results["labels"], processed_results["boxes"]):
        # Get the predicted string name from the Hugging Face config
        predicted_name = model.config.id2label[label.item()].lower()
        
        # Only process the box if it belongs to one of the 12 ExDark classes
        if predicted_name in coco_name_to_exdark_id:
            exdark_class_id = coco_name_to_exdark_id[predicted_name]
            
            xmin, ymin, xmax, ymax = box.tolist()
            width = xmax - xmin
            height = ymax - ymin
            
            results.append({
                "image_id": img_id,
                "category_id": exdark_class_id,
                "bbox": [xmin, ymin, width, height],
                "score": score.item()
            })

# 9. Calculate overall FPS
fps = total_images / total_inference_time_sec if total_inference_time_sec > 0 else 0

print("\n--- Hardware Metrics ---")
print(f"Total Inference Time: {total_inference_time_sec:.2f} seconds")
print(f"Average FPS: {fps:.2f} frames/sec\n")

# 10. Save the results temporarily to calculate the metrics
output_json = "rtdetr_predictions.json"
with open(output_json, "w") as f:
    json.dump(results, f)

print("Inference complete. Calculating COCO metrics...")

# 11. Load predictions and evaluate against Ground Truth
coco_dt = coco_gt.loadRes(output_json)
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()