import sys
import torch
import numpy as np
import cv2
from unet_model import FaceDetector

units = 32
input_image = "data/train_data/Pilot2_MagicFlute_Part1_8500.npy"
if len(sys.argv) > 1:
    units = int(sys.argv[1])
    path_to_model = sys.argv[2]
else:
    units = 32
    path_to_model = f"saved/model_5_{units}_1000"


def visualize_predictions(temperatures, output, threshold):
    # Create a copy of the temperature frame
    image = cv2.normalize(
        temperatures, 
        None, 
        alpha=0, 
        beta=255, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_8U
    )
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Find the regions where the model prediction is above the threshold
    out_max = output.max()
    #print(np.where(output_np>(threshold*output_np.max())))
    above_threshold_indices = np.where(output > (out_max*threshold))
    image[above_threshold_indices] = [255,0,0]

    # Display the output frame
    cv2.imshow("Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Load the model from disk
model = FaceDetector(units)
model.load_state_dict(
    torch.load(
        path_to_model,
        map_location=torch.device('cpu')
    )['model_state_dict']
)
model.eval()

# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
image_tensor = torch.tensor(image)

# Run the model on the image tensor
with torch.no_grad():
    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor).squeeze(0)

# Convert the output tensor to a numpy array
output_np = output.numpy()

print(output_np.max(), output_np.min(), np.where(output_np==output_np.max()))

# Display the input image and the predicted output
visualize_predictions(image, output_np, 0.3)
