"""
Combine all dataset to a big folder
with images, labels, binary_labels
"""

import shutil
import os, cv2, time
import numpy as np

local_path = "./dataset/public"
output_path = "./ganzin_dataset/"

output_image_path = os.path.join(output_path, "images")
output_label_path = os.path.join(output_path, "labels")

os.makedirs(output_path, exist_ok=True)
os.makedirs(output_image_path, exist_ok=True)
os.makedirs(output_label_path, exist_ok=True)

# Sub list
# subjects = ["S1"]
subjects = ["S1", "S2", "S3", "S4"]

# Get the case
start_index = 0
sum = 0
for sub in subjects:
    sub_path = os.path.join(local_path, sub)
    sub_folder = os.listdir(sub_path)
    for mini_folder in sub_folder:
        # print(mini_folder)
        mini_folder_path = os.path.join(sub_path, mini_folder)
        mini_folder_images = os.listdir(mini_folder_path)
        image_indices = sorted([int(name.split(".")[0]) for name in mini_folder_images if name.endswith(".jpg")])
        max_length = image_indices[-1]
        for idx in range(max_length+1):
            input_path = os.path.join(mini_folder_path, f"{idx}.jpg")
            label_path = os.path.join(mini_folder_path, f"{idx}.png")
            if os.path.isfile(label_path):
                sum += 1
                dst_input_path = os.path.join(output_image_path, f"{start_index}.jpg")
                dst_label_path = os.path.join(output_label_path, f"{start_index}.png")
                shutil.copy(input_path, dst_input_path)
                shutil.copy(label_path, dst_label_path)
                start_index += 1

# Process label
output_binary_label_path = os.path.join(output_path, "binary_labels")
os.makedirs(output_binary_label_path, exist_ok=True)
label_images_list = os.listdir(output_label_path)
for label_name in label_images_list:
    label_path = os.path.join(output_label_path, label_name)
    label_img = cv2.imread(label_path)
    gray_label = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    binary_label = np.zeros_like(gray_label, dtype=np.uint8)
    binary_label[gray_label != 0.0] = 1
    cv2.imwrite(os.path.join(output_binary_label_path, label_name), binary_label)