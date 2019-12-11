"""
download_images.py
This python scripts downloads images from a folder list of links stored in a text file.
This scripts simply reads the links line by line and downloads them  to the desired location.
"""

import urllib.request
import os
from tqdm import tqdm

def download_images(type, mode, url_file_path, output_file_root):
	"""
	Downloads the images and stores at the desired location.
	"""
	counter = 0

	with open(url_file_path, 'r') as link_file:
		image_links = link_file.readlines()

	for link in tqdm(image_links):
		output_path = output_file_root + str(counter) + '.tiff'
		urllib.request.urlretrieve(link, output_path)
		counter += 1
	
if __name__ == '__main__':
	
	type = '/Images/' 
	mode = 'Train'
	
	url_file_path = '../Data/Roads/_Links/Images.txt'
	output_file_root = '../Data/Roads/' + mode + type 
	
	download_images(type, mode, url_file_path, output_file_root)