import cv2
import os
# Folder path containing the images and annotations
folder_path = 'data/areca_99/images/'

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.JPG'):
        # Load the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Load the corresponding annotation file
        annotation_path = os.path.join(folder_path, filename.replace('.JPG', '.txt'))
        with open(annotation_path, 'r') as file:
            # Read the normalized bounding box coordinates from the annotation file
            normalized_boxes = []
            for line in file:
                box_data = line.strip().split()
                x_center, y_center, box_width, box_height = map(float, box_data[1:])
                normalized_boxes.append((x_center, y_center, box_width, box_height))

        # Convert normalized coordinates to pixel values
        height, width, _ = image.shape  
        bounding_boxes = []
        for box in normalized_boxes:
            x_center, y_center, box_width, box_height = box
            x = int((x_center - (box_width / 2)) * width)
            y = int((y_center - (box_height / 2)) * height)
            w = int(box_width * width)
            h = int(box_height * height)
            bounding_boxes.append((x, y, w, h))

        # Draw the bounding boxes on the image
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            print(x,y,w,h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with bounding boxes
        window_name = 'Image with Bounding Boxes'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, image.shape[1], image.shape[0])
        cv2.imshow(window_name, image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
