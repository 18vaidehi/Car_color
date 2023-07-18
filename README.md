# Object Detection with YOLOv5 and Color Identification Readme

This repository contains code that utilizes the YOLOv5 model for object detection on an input image and identifies the dominant color of the detected vehicles.

## Prerequisites
- Python 3.6 or higher
- PyTorch
- OpenCV (cv2)
- NumPy
- PIL (Pillow)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ultralytics/yolov5.git
   ```

2. Change the working directory to the cloned repository:
   ```
   cd yolov5
   ```

3. Download the pre-trained YOLOv5 model weights:
   ```
   wget https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt
   ```

4. Install the required Python packages:
   ```
   pip install torch torchvision
   pip install opencv-python
   pip install numpy
   pip install pillow
   ```

## Usage

1. Import the necessary libraries and load the YOLOv5 model:

   ```python
   import torch
   from torchvision import transforms
   from PIL import Image
   import cv2
   import numpy as np
   import math
   model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
   ```

2. Load the image you want to process:

   ```python
   image_path = 'path/to/the/image.jpg'
   image = cv2.imread(image_path)

   # Convert the image to RGB
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   ```

3. Preprocess the image and run it through the YOLOv5 model:

   ```python
   transform = transforms.Compose([
       transforms.Resize((640, 640)),
       transforms.ToTensor(),
   ])
   img_size = 640
   image_resized = cv2.resize(image_rgb, (img_size, img_size))
   image = torch.from_numpy(image_resized.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)

   results = model(image)
   pred = results[0]
   ```

4. Get the bounding boxes, class labels, and scores of the detected vehicles:

   ```python
   boxes = pred[:, :4].cpu().numpy()
   class_id = pred[:, -1].cpu().numpy().astype(int)
   score = pred[:, 4].cpu().numpy()
   ```

5. Define a function to identify the dominant color:

   ```python
   def get_color_name(b, g, r):
       # Define color dictionary
       color_dict = {
           # Add color mappings here
       }

       color_name = 'Unknown'
       min_dist = float('inf')
       for name, color in color_dict.items():
           dist = math.sqrt((color[0] - b) ** 2 + (color[1] - g) ** 2 + (color[2] - r) ** 2)
           if dist < min_dist:
               min_dist = dist
               color_name = name

       return color_name
   ```

6. Iterate through the detected vehicles, get their color, and print the results:

   ```python
   for box, label, score in zip(boxes, class_id, score):
       x1, y1, x2, y2 = box.tolist()
       vehicle_image = image[:, :, int(y1):int(y2), int(x1):int(x2)].cpu().numpy().transpose(0, 2, 3, 1)[0]

       # Handle cases where the image may not have RGB channels
       if vehicle_image is not None and len(vehicle_image) > 0:
           if len(vehicle_image.shape) == 3 and vehicle_image.shape[2] == 3:
               vehicle_image_bgr = cv2.cvtColor(vehicle_image, cv2.COLOR_RGB2BGR)
           else:
               vehicle_image_bgr = cv2.cvtColor(vehicle_image, cv2.COLOR_GRAY2BGR)
       else:
           vehicle_image_bgr = np.zeros((10, 10, 3), dtype=np.uint8)

       # Calculate the mean BGR color of the vehicle image
       bgr_mean = cv2.mean(vehicle_image_bgr)[:3]
       b = int(bgr_mean[0])
       g = int(bgr_mean[1])
       r = int(bgr_mean[2])

       # Get the dominant color name
       color_name = get_color_name(b, g, r)

       # Print the bounding box, class label, score, and color
       print("Bounding Box: {}".format(box))
       print("Class: {}".format(class_id))
       print("Score: {}".format(score))
       print("Color: {}".format(color_name))
   ```

Remember to replace `'path/to/the/image.jpg'` with the actual path to the image you want to process. Additionally, add more color mappings to the `color_dict` dictionary if needed.

## License

The code in this repository is under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- YOLOv5 by Ultralytics: https://github.com/ultralytics/yolov5
- PyTorch: https://pytorch.org/
- OpenCV: https://opencv.org/
- NumPy: https://numpy.org/
- PIL (Pillow): https://python-pillow.org/

Please note that this code assumes that you have already set up the environment with the required packages and the YOLOv5 model weights. Enjoy object detection and color identification with YOLOv5!
