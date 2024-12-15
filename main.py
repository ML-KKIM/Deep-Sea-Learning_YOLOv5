# -*- coding: utf-8 -*-

import os
import random
import shutil
import json
from PIL import Image

# Set paths
project_path = r'E:\GradSchool\CS7643DL\2024 Fall\Final project\the-nature-conservancy-fisheries-monitoring'
data_path = os.path.join(project_path, 'Data')
image_path = os.path.join(data_path, 'image', 'raw')
labels_path = os.path.join(data_path, 'labels')
yolov5_path = r'C:\Users\DK\yolov5' 

# Create directories if they don't exist
train_dir = os.path.join(image_path, '..', 'train')
val_dir = os.path.join(image_path, '..', 'val')

labels_train_dir = os.path.join(labels_path, 'train')
labels_val_dir = os.path.join(labels_path, 'val')

for dir_path in [train_dir, val_dir, labels_train_dir, labels_val_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Split dataset
def split_dataset(dataset_path, train_dir, val_dir, split_ratio=0.8):
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        files = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
        random.shuffle(files)

        split_index = int(len(files) * split_ratio)
        train_files = files[:split_index]
        val_files = files[split_index:]

        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(train_dir, class_name, file))

        for file in val_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(val_dir, class_name, file))

# Convert JSON annotations to YOLOv5 format
def convert_annotations(anon_path, labels_base_path, dataset_path):
    #re doing class
    #YFT = 7
    #Shark = 6
    #Other = 5
    #LAG = 3
    #DOL = 2
    #BET = 1
    #ALB = 0
    classes = ['ALB', 'BET', 'DOL', 'LAG',  'NoF', 'OTHER', 'SHARK', 'YFT']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def convert_bbox(size, box):
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x_center = (box['x'] + box['width'] / 2.0) * dw
        y_center = (box['y'] + box['height'] / 2.0) * dh
        width = box['width'] * dw
        height = box['height'] * dh
        return (x_center, y_center, width, height)

    for json_file in os.listdir(anon_path):
        if not json_file.endswith('.json'):
            continue

        cls_lower = json_file.split('_')[0]
        cls = cls_lower.upper()

        if cls not in class_to_idx:
            print(f"Warning: Class '{cls}' from '{json_file}' not in classes list.")
            continue

        json_path = os.path.join(anon_path, json_file)

        with open(json_path, 'r') as f:
            data = json.load(f)

        for item in data:
            filename = item['filename']
            annotations = item.get('annotations', [])

            split = 'train' if random.random() < 0.8 else 'val'

            image_path = os.path.join(dataset_path, split, cls, filename)
            label_dir = os.path.join(labels_base_path, split, cls)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

            if not os.path.exists(image_path):
                print(f"Warning: Image '{image_path}' does not exist.")
                continue

            with Image.open(image_path) as img:
                img_width, img_height = img.size

            yolo_annotations = []
            for ann in annotations:
                bbox = {
                    'x': ann['x'],
                    'y': ann['y'],
                    'width': ann['width'],
                    'height': ann['height']
                }
                x_center, y_center, width, height = convert_bbox((img_width, img_height), bbox)
                yolo_annotations.append(f"{class_to_idx[cls]} {x_center} {y_center} {width} {height}")

            os.makedirs(label_dir, exist_ok=True)
            with open(label_path, 'w') as lf:
                lf.write('\n'.join(yolo_annotations))

# Generate YAML for YOLOv5 training
def generate_yaml(train_dir, val_dir, yaml_path):
    yaml_content = f"""
    train: {train_dir}
    val: {val_dir}

    # Number of classes
    nc: 8

    # Class names
    names: ['ALB', 'BET', 'DOL', 'LAG',  'NoF', 'OTHER', 'SHARK', 'YFT']
    """

    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

# Train YOLOv5 model
def train_yolov5(yaml_path, epochs):
    os.chdir(yolov5_path)
    os.system(f'python train.py --img 640 --batch 16 --epochs {epochs} --data "{yaml_path}" --weights yolov5s.pt --device 0 --exist-ok --save-period 1')


# Testing with YOLOv5 model
def test_yolov5(best_weights_path, test_images_path, output_path):
    os.chdir(yolov5_path)  # Change directory to YOLOv5 root
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
    
    # Run YOLOv5 detection script
    os.system(f'python detect.py --weights "{best_weights_path}" --source "{test_images_path}" --img 640 --conf 0.25 --device 0 --save-txt --save-conf --project "{output_path}" --name test_results')
'''
# Step 1: Split the dataset
split_dataset(image_path, train_dir, val_dir, split_ratio=0.8)

# Step 2: Convert JSON annotations to YOLOv5 format
anon_path = os.path.join(data_path, 'anon')
convert_annotations(anon_path, labels_path, data_path)
'''


'''

# Step 3: Generate YAML for YOLOv5 training
yaml_path = os.path.join(data_path, 'data_yaml.yaml')
generate_yaml(train_dir, val_dir, yaml_path)


# Step 4: Train YOLOv5 model
epochs = 300
train_yolov5(yaml_path, epochs)


'''

# Step 5: Evaluate on image files
best_weights_path = r'C:/Users/DK/yolov5/runs/train/exp8/weights/best.pt'
#test_stg1
test_images_path = r'E:/GradSchool/CS7643DL/2024 Fall/Final project/the-nature-conservancy-fisheries-monitoring/test_stg1/test_stg1'

#test_stg2
#test_images_path = r'E:/GradSchool/CS7643DL/2024 Fall/Final project/the-nature-conservancy-fisheries-monitoring/test_stg2/test_stg2'

output_path = 'C:/Users/DK/yolov5/runs/detect'

test_yolov5(best_weights_path, test_images_path, output_path)





os.system('pause')


