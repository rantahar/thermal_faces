import json
import numpy as np
from subsection_utils import extract_subregions, plot_boxes_on_image

region_size = 48
step_fraction = 0.2
input_image = "data/train_data/Pilot2_MagicFlute_Part1_18500.npy"
json_file = input_image.replace(".npy",".json")

# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
with open(json_file, 'r') as f:
    json_data = json.load(f)

regions = extract_subregions(image, json_data["labels"], region_size, region_size, step_fraction)

boxes = [(r[2], r[3], region_size, region_size) for r in regions if r[1] > 0.5]
plot_boxes_on_image(image, boxes)
