from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from urllib.request import urlopen
import numpy as np

# Assuming results contains the prediction from YOLO
results = model("https://ultralytics.com/images/bus.jpg")

# Load the image
image_url = results.render() if hasattr(results, 'render') else "https://ultralytics.com/images/bus.jpg"
image_file = urlopen(image_url)
image = np.asarray(bytearray(image_file.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(image)


bounding_boxes = results[0].boxes.boxes.numpy()
print(bounding_boxes)

# Draw bounding boxes
for (x1, y1, x2, y2, conf, cls) in bounding_boxes:
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()
plt.show()