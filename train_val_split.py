
import numpy as np
import os, cv2, time
import random, shutil

local_dir = "./ganzin_dataset/"
output_dir = "ganzin_dataset_final"
os.makedirs(output_dir, exist_ok=True)

images_dir = os.path.join(local_dir, "images")
label_dir = os.path.join(local_dir, "labels")
binary_label_dir = os.path.join(local_dir, "binary_labels")

phases = ["train", "val"]
for phase in phases:
    output_images_dir = os.path.join(output_dir, phase, "images")
    output_label_dir = os.path.join(output_dir, phase, "labels")
    output_binary_label_dir = os.path.join(output_dir, phase, "binary_labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_binary_label_dir, exist_ok=True)

total_images = len(os.listdir(images_dir))
split_ratio = 0.95

total_indices = list(range(0, total_images))
num_train_folder = round(split_ratio * total_images)

train_index  = random.sample(total_indices, num_train_folder)
val_index = list(set(total_indices) - set(train_index))

for i, idx in enumerate(train_index):
    img_path = os.path.join(images_dir, f"{idx}.jpg")
    label_path = os.path.join(label_dir, f"{idx}.png")
    binary_label_path = os.path.join(binary_label_dir, f"{idx}.png")

    dst_img_path = os.path.join(output_dir, "train", "images", f"{i}.jpg")
    dst_label_path = os.path.join(output_dir, "train", "labels", f"{i}.png")
    dst_binary_label_path = os.path.join(output_dir, "train", "binary_labels", f"{i}.png")

    shutil.copy(img_path, dst_img_path)
    shutil.copy(label_path, dst_label_path)
    shutil.copy(binary_label_path, dst_binary_label_path)

for i, idx in enumerate(val_index):
    img_path = os.path.join(images_dir, f"{idx}.jpg")
    label_path = os.path.join(label_dir, f"{idx}.png")
    binary_label_path = os.path.join(binary_label_dir, f"{idx}.png")

    dst_img_path = os.path.join(output_dir, "val", "images", f"{i}.jpg")
    dst_label_path = os.path.join(output_dir, "val", "labels", f"{i}.png")
    dst_binary_label_path = os.path.join(output_dir, "val", "binary_labels", f"{i}.png")

    shutil.copy(img_path, dst_img_path)
    shutil.copy(label_path, dst_label_path)
    shutil.copy(binary_label_path, dst_binary_label_path)