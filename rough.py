import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import time

# Function to process an image
def process_image(image_path):
    # Simulate some processing time
    time.sleep(0.5)
    return f"Processed image: {image_path}"

# List of image paths
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]

# Number of worker processes
num_threads = multiprocessing.cpu_count()

# # Create a pool of worker processes
# with Pool(num_threads) as pool:
#     # Create a progress bar
#     with tqdm(total=len(image_paths), desc="Processing images") as pbar:
#         # Map the process_image function to the image paths
#         results = pool.imap(process_image, image_paths)
        
#         # Iterate over the results with progress bar update
#         for result in results:
#             pbar.update(1)  # Update the progress bar
#             pbar.set_postfix({"Result": result})  # Update the displayed result

# # The progress bar closes automatically when exiting the 'with' block
