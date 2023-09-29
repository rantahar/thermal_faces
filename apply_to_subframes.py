import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from subsection_utils import extract_subregions, display_image
from reduce_model import FaceDetector

units = 16
section_size = 64
step_fraction = 0.5
input_image = "data/train_data/Pilot2_MagicFlute_Part1_10500.npy"
if len(sys.argv) > 1:
    units = int(sys.argv[1])
    path_to_model = sys.argv[2]
else:
    units = 16
    path_to_model = f"saved/reduction_model_6_{units}_1000"


def visualize_predictions(temperatures, boxes, threshold):
    plt.imshow(temperatures, cmap='hot')
    plt.axis('off')
    plt.title('Temperature Image')
    plt.show()



# Load the model from disk
model = FaceDetector(section_size, section_size, units)
model.load_state_dict(
    torch.load(
        path_to_model,
        map_location=torch.device('cpu')
    )['model_state_dict']
)
model.eval()

# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
regions = extract_subregions(image, None, section_size, section_size, step_fraction)

for array, _, x, y in regions:
    array = torch.tensor(array)
    display_image(array)

    # Run the model on the image tensor
    with torch.no_grad():
        image_tensor = torch.tensor(array).unsqueeze(0)
        output = model(image_tensor).squeeze(0).numpy()
    
    print(x, y, output)

