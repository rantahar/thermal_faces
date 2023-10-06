import json
import numpy as np
from subsection_utils import extract_rescaled_subregions, plot_boxes_on_image

region_sizes = [32, 48, 64]
step_fraction = 0.2
input_image = "data/train_data/Pilot2_MagicFlute_Part1_18500.npy"
json_file = input_image.replace(".npy",".json")

# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
with open(json_file, 'r') as f:
    json_data = json.load(f)

labels = [l for l in json_data["labels"] if l['l'] == 1]
regions = extract_rescaled_subregions(image, labels, region_sizes, step_fraction)


boxes = [(r[2], r[3], r[4], r[4], 1) for r in regions if r[1] > 0.5]
plot_boxes_on_image(image, boxes, labels = False)
