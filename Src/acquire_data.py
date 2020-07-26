"""
Filename: acquire_data.py

Function:  Downloads the Massachusetts Roads Dataset or the Massachusetts Buildings Dataset. By changing "link_file" to point at a custom list of links, you can download any other dataset too.

Author: Jerin Paul (https://github.com/Paulymorphous)
Website: https://www.livetheaiexperience.com/
"""

import urllib.request
import os
from tqdm import tqdm
import click
import time

def download_images(link_file_images, output_directory, image_type):
	"""
	Reads a file with links to the images, and downloads them to the specified location.

	Parameters
   	----------
	>link_file_images (str): path to the file with images.
	>output_directory (str): path to target directory.
	>image_type (str): Whether the images are target masks or satellite images.
	"""

	print("\nDownloading", image_type)
	
	counter = 0

	with open(link_file_images, 'r') as link_file:
		image_links = link_file.readlines()

	for image_link in tqdm(image_links, total = len(image_links)):
		
		image_path = output_directory + image_type + "/" + os.path.basename(image_link)
		
		urllib.request.urlretrieve(image_link, image_path)
		
		counter += 1
	
	print("{} images downloaded to {}\n".format(counter, output_directory+image_type))


if __name__ == '__main__':
	
	dataset_name = "MassachusettsRoads"
	link_file_images = "../Data/_Links/{}}/Images.txt".format(dataset_name)
	link_file_targets = "../Data/_Links/{}}/Targets.txt".format(dataset_name)
	output_directory = "../Data/{}/".format(dataset_name)

	if not os.path.exists(output_directory):
		os.mkdir(output_directory)
		os.mkdir(output_directory + "Images/")
		os.mkdir(output_directory + "Targets/")

	start_time = time.time()
	download_images(link_file_images, output_directory, "Images")
	download_images(link_file_targets, output_directory, "Targets")
	print("TOTAL TIME: {} minutes".format(round((time.time() - start_time)/60, 2)))