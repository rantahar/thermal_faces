import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from subsection_utils import extract_subregions, plot_boxes_on_image
from reduce_model import FaceDetector

threshold = 0.99
units = 16
region_size = 64
step_fraction = 0.2
input_image = "data/train_data/Pilot2_MagicFlute_Part1_10500.npy"
if len(sys.argv) > 1:
    path_to_model = sys.argv[1]
else:
    path_to_model = f"saved/reduction_model_6_{units}_1000"

# Load the model from disk
model_dict = torch.load(
    path_to_model,
    map_location=torch.device('cpu')
)
units = model_dict.get("units", units)
region_size = model_dict.get("region_size", region_size)

model = FaceDetector(region_size, region_size, units)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
regions = extract_subregions(image, None, region_size, region_size, step_fraction)

arrays = [r[0] for r in regions]
with torch.no_grad():
    tensor = torch.tensor(arrays)
    predictions = model(tensor).squeeze(0).numpy()

print(predictions.max())
threshold = threshold * predictions.max()

boxes = []
for i in range(len(regions)):
    array, _, x, y = regions[i]
    output = predictions[i]

    if output > threshold:
        print((x, y, region_size, region_size), output)
        boxes.append((x, y, region_size, region_size))
    
print(image.shape)
plot_boxes_on_image(image, boxes)
        

