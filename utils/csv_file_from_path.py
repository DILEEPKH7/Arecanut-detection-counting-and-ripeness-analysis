
import os
import csv

folder_path = 'C:/Users/khdil/Desktop/My_files/Github/YOLOv6/data/areca_99/images/train/'  # Replace with the path to your folder

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter for image files (you can modify this condition based on your specific requirements)
image_files = [file for file in files if file.endswith('.JPG') or file.endswith('.png')]

# Write the image names to the CSV file
csv_file_path = 'image_names.csv'  # The name of the output CSV file

with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image Name'])  # Write the header
    for image_file in image_files:
        writer.writerow([image_file])  # Write each image name on a new row

print("CSV file created successfully!")
