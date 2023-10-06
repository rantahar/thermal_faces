import sys
import torch
import json
import numpy as np
from subsection_utils import plot_boxes_on_image, non_max_suppression
from reduce_model import FaceDetector
import cv2
import time


threshold = 5
region_sizes = [32,48,64]
step_fraction = 0.2
input_image = "data/train_data/Pilot1_MagicFlute_Part1_26000.npy"
assert len(sys.argv) > 1
path_to_model = sys.argv[1]

# Load the model from disk
model_dict = torch.load(
    path_to_model,
    map_location=torch.device('cpu')
)
units = model_dict.get("units")
region_size = model_dict.get("region_size")

model = FaceDetector(region_size, region_size, units)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

json_file = input_image.replace(".npy",".json")

# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
with open(json_file, 'r') as f:
    json_data = json.load(f)

# Load temperature values and process
image = np.load(input_image).astype(np.float32)

boxes = []
height, width = image.shape
for size in region_sizes:
    start_time = time.time()
    step_size = int(size*step_fraction)
    for y in range(0, height - size + 1, step_size):
        for x in range(0, width - size + 1, step_size):
            subregion = np.array(image[y:y+size, x:x+size])
            subregion = cv2.resize(subregion, (region_size, region_size))
            tensor = torch.tensor(subregion).unsqueeze(0)
            score = model(tensor).squeeze(0).item()
            if score > threshold:
                boxes.append((x, y, size, size, score))

    print(f"Size {size} sections processed in: {time.time() - start_time:.2f}s")


boxes = non_max_suppression(boxes, 0.1)

plot_boxes_on_image(image, boxes)



