import cv2
import glob
import csv

def crop_image(image_path, x, y, width, height):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Crop the image using the provided dimensions
    cropped_image = image[y:y+height, x:x+width]

    return cropped_image

def adjust_labels(labels, x, y, width, height, image_width, image_height):
    adjusted_labels = []

    for label in labels:
        class_id, x_center, y_center, box_width, box_height = map(float, label.split())

        # Convert normalized coordinates to pixel values
        pixel_x_center = int((x_center * image_width))
        pixel_y_center = int((y_center * image_height))
        pixel_box_width = int(box_width * image_width)
        pixel_box_height = int(box_height * image_height)

        new_x_center = (pixel_x_center - x)/ width
        new_y_center = (pixel_y_center -y) / height
        new_box_width = pixel_box_width / width
        new_box_height = pixel_box_height / height

        adjusted_label = f"{int(class_id)} {new_x_center} {new_y_center} {new_box_width} {new_box_height}"
        adjusted_labels.append(adjusted_label)

    return adjusted_labels

# Provide the directory path containing the images
image_directory = 'data/areca_99/images/train/'

# Provide the directory path containing the YOLO format text files
label_directory = 'data/areca_99/labels/'

# Get a list of all the image files in the directory
image_paths = glob.glob(image_directory + '*.JPG')
image_paths = image_paths

# Iterate over each image and corresponding label file
for image_path in image_paths:
    # Extract the file name from the path
    file_name = image_path.split('/')[-1]

    # Get the corresponding label file path
    label_file_path = label_directory + file_name.replace('.JPG', '.txt')
    print(label_file_path)

    # Read the labels from the label file
    with open(label_file_path, 'r') as file:
        labels = file.read().splitlines()

    # Get the image dimensions
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Crop the image to the desired dimensions
    cropped_image = crop_image(image_path, x=619, y=0, width=3937, height=3349)

    # Adjust the labels according to the new image dimensions
    # a=image_directory.split('/')[:4]
    # n = 'image_names.csv' 
    path = 'data/areca_99/images/image_names.csv'
    with open(path,'r') as file:
        reader = csv.DictReader(file)

        for row in reader: 
            name = row['Image Name']
            width = int(row['width'])
            height = int(row['height'])
            x = int(row['x'])
            y = int(row['y'])
            adjusted_labels = adjust_labels(labels, x=x, y=y, width=width, height=height, image_width=image_width, image_height=image_height)

    # Save the cropped image
    cv2.imwrite(f'{file_name}', cropped_image)

    # Save the adjusted labels to a new label file
    with open(f'{label_file_path}', 'w') as file:
        file.write('\n'.join(adjusted_labels))
