import json
import os

# Define the mapping of COCO class IDs to YOLO class IDs
coco_to_yolo_mapping = {
    1: 0,  # COCO class ID 1 corresponds to YOLO class ID 0, adjust this mapping as needed
    2: 1,  # COCO class ID 2 corresponds to YOLO class ID 1, adjust this mapping as needed
    3: 2, # Add more mappings for other COCO classes to YOLO classes
}

# Load the JSON annotation file
with open('data/areca/annotations/areca_train.json') as json_file:
    annotations = json.load(json_file)

# Iterate through the annotations
for image_info in annotations['images']:
    image_id = image_info['id']
    file_name = image_info['file_name']
    image_width = image_info['width']
    image_height = image_info['height']
    image_labels = []

    # Find annotations for the current image
    for annotation in annotations['annotations']:
        if annotation['image_id'] == image_id:
            # Extract the label and bounding box information
            coco_label = annotation['category_id']
            coco_bbox = annotation['bbox']

            # Normalize the bounding box coordinates
            normalized_bbox = [
                (coco_bbox[0]) / image_width,  # x_min
                (coco_bbox[1]) / image_height,  # y_min
                (coco_bbox[0] + coco_bbox[2]) / image_width,  # x_max
                (coco_bbox[1] + coco_bbox[3]) / image_height  # y_max
            ]

            # Map COCO label to YOLO label
            if coco_label in coco_to_yolo_mapping:
                yolo_label = coco_to_yolo_mapping[coco_label]
                # Format the label information in YOLO format (class_id x_center y_center width height)
                label_text = f"{yolo_label} {normalized_bbox[0] + ((normalized_bbox[2] - normalized_bbox[0]) / 2)} " \
                             f"{normalized_bbox[1] + ((normalized_bbox[3] - normalized_bbox[1]) / 2)} " \
                             f"{normalized_bbox[2] - normalized_bbox[0]} {normalized_bbox[3] - normalized_bbox[1]}"
                image_labels.append(label_text)

    # Write the label text file in YOLO format
    label_file_name = os.path.splitext(file_name)[0] + '.txt'
    with open(label_file_name, 'w') as label_file:
        for label_text in image_labels:
            label_file.write(label_text + '\n')
