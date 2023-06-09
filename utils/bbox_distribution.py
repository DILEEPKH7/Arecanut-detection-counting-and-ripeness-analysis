import glob
import matplotlib.pyplot as plt

img_width = 1024
img_height = 1024
def parse_yolo_boxes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    widths = []
    heights = []

    for line in lines:
        _, _, width, height = map(float, line.strip().split()[1:])
        widths.append(width)
        heights.append(height)

    return widths, heights

def plot_bounding_box_distribution(widths, heights, file_name):
    plt.hist(widths, bins=50, alpha=0.5, color='b', label='Width')
    plt.hist(heights, bins=50, alpha=0.5, color='r', label='Height')

# Provide the directory path containing the YOLO format text files
directory_path = 'data/areca_tiling_1024/labels/train/'

# Get a list of all the text files in the directory
file_paths = glob.glob(directory_path + '*.txt')

widths_all = []
heights_all = []

# Iterate over each text file and combine the bounding box sizes
for file_path in file_paths:
    widths, heights = parse_yolo_boxes(file_path)
    widths_all.extend(widths)
    heights_all.extend(heights)

widths_all = [width * img_width for width in widths_all]
heights_all = [height * img_width for height in heights_all]
# Plot the bounding box size distribution
plt.figure(figsize=(10, 5))
plot_bounding_box_distribution(widths_all, heights_all, "Combined Files")
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.title('Bounding Box Size Distribution')
plt.legend()
plt.show()
