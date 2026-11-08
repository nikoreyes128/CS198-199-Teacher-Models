import json

with open('../dataset/ExDark-coco/valid/_annotations.coco.json', 'r') as f:
    data = json.load(f)

print("--- COCO CATEGORIES IN JSON ---")
for cat in data['categories']:
    print(f"Name: {cat['name']} -> ID: {cat['id']}")

print("\n--- SAMPLE IMAGE ENTRY ---")
print(data['images'][0])

print("\n--- SAMPLE ANNOTATION ENTRY ---")
print(data['annotations'][0])