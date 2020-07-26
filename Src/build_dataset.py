"""
Filename: build_dataset.py

Function: Crops the images into small 256x256 images and divides the dataset into training and testing set.

Author: Jerin Paul (https://github.com/Paulymorphous)
Website: https://www.livetheaiexperience.com/
"""

import numpy as np
import cv2
from tqdm import tqdm
import os
import math
import time

def train_test_split(images_path, masks_path, test_split=0.3):
    """
    Splits the dataset into train and test sets and stores then in an ImageDataAugmentation friendly format.   

    Please note:
    > All the images which has less than 1% annotation, in terms of area is removed. In other words, Images that are 99% empty are removed.

    Parameters
   	----------
	>images_path (str): Path to the directory containing all the images.
	>masks_path (str): Path to the directory containing all the masks.
	>test_split (float): Ratio of the size of the test set to the entire dataset. Default value: 0.3
    """

    image_filenames = [filename for filename in os.walk(images_path)][0][2]
    test_set_size = int(test_split*len(image_filenames)) 
    
    root_path = os.path.dirname(os.path.dirname(images_path)) + "/"
    train_dir = root_path + "Train/"
    test_dir = root_path + "Test/"
    
    if not os.path.exists(train_dir):
        print("CREATING:", train_dir)
        os.makedirs(train_dir+"Images/samples/")
        os.makedirs(train_dir+"Masks/samples/")
        
    if not os.path.exists(test_dir):
        print("CREATING:", test_dir)
        os.makedirs(test_dir+"Images/samples/")
        os.makedirs(test_dir+"Masks/samples/")
        
    train_image_dir = train_dir+"Images/samples/"
    train_mask_dir = train_dir+"Masks/samples/"
    test_image_dir = test_dir+"Images/samples/"
    test_mask_dir = test_dir+"Masks/samples/"
    
    for n, filename in enumerate(image_filenames):
        if n < test_set_size:
            os.rename(images_path + filename, test_image_dir + filename)
            os.rename(masks_path + filename, test_mask_dir + filename)
        else:
            os.rename(images_path + filename, train_image_dir + filename)
            os.rename(masks_path + filename, train_mask_dir + filename)
            
    print("Train-Test-Split COMPLETED.\nNUMBER OF IMAGES IN TRAIN SET:{}\nNUMBER OF IMAGES IN TEST SET: {}".format(len(image_filenames)-test_set_size, test_set_size))
    print("\nTrain Directory:", train_dir)
    print("Test Directory:", test_dir)

def crop_and_save(images_path, masks_path, new_images_path, new_masks_path, img_width, img_height):
    """
    Imports Images and creates multiple crops and then stores them in the specified folder. Cropping is important in the project to protect spatial information, which otherwise would be lost if we resize the images.
    Please note:
    > All the images which has less than 1% annotation, in terms of area is removed. In other words, Images that are 99% empty are removed.

    Parameters
   	----------
	>images_path (str): Path to the directory containing all the images.
	>masks_path (str): Path to the directory containing all the masks.
	>new_images_path (str): Path to the Directory where the cropped images will be stored.
	>new_masks_path (str): Path to the Directory where the cropped masks will be stored.
	>img_width (int): width of the cropped image.
    >img_height (int): height of the cropped image.
    """
    
    print("Building Dataset.")
    
    num_skipped = 0
    start_time = time.time()
    files = next(os.walk(images_path))[2]
    print('Total number of files =',len(files))
    
    for image_file in tqdm(files, total = len(files)):
       
        image_path = images_path + image_file
        image = cv2.imread(image_path)
        
        mask_path = masks_path + image_file[:-1]
        mask = cv2.imread(mask_path, 0)
        
        num_splits = math.floor((image.shape[0]*image.shape[1])/(img_width*img_height))
        counter = 0
        
        for r in range(0, image.shape[0], img_height):
            for c in range(0, image.shape[1], img_width):
                counter += 1
                blank_image = np.zeros((img_height ,img_width, 3), dtype = int)
                blank_mask = np.zeros((img_height ,img_width), dtype = int)
                
                new_image_path = new_images_path + str(counter) + '_' + image_file
                new_mask_path = new_masks_path + str(counter) + '_' + image_file
                
                new_image = np.array(image[r:r+img_height, c:c+img_width,:])
                new_mask = np.array(mask[r:r+img_height, c:c+img_width])
            
                
                blank_image[:new_image.shape[0], :new_image.shape[1], :] += new_image
                blank_mask[:new_image.shape[0], :new_image.shape[1]] += new_mask
                
                blank_mask[blank_mask>1] = 255
                
                # Skip any Image that is more than 99% empty.
                if np.any(blank_mask):
                    num_black_pixels, num_white_pixels = np.unique(blank_mask, return_counts=True)[1]
                    
                    if num_white_pixels/num_black_pixels < 0.01:
                        num_skipped+=1
                        continue

                    cv2.imwrite(new_image_path, blank_image)
                    cv2.imwrite(new_mask_path, blank_mask)
                
    
    print("EXPORT COMPLETE: {} seconds.\nImages exported to {}\nMasks exported to{}".format(round((time.time()-start_time), 2), new_images_path, new_masks_path))
    print("\n{} Images were skipped.".format(num_skipped))


if __name__ == "__main__":
    root_data_path = "../Data/BuildingsDataSet/"
    test_to_train_ratio = 0.3 
    img_width = img_height = 256
    num_channels = 3

    # Path Information
    images_path = root_data_path + "sat/"
    masks_path = root_data_path + "map/"
    new_images_path = root_data_path + "Images/"
    new_masks_path = root_data_path + "Masks/"

    for path in [new_images_path, new_masks_path]:
        if not os.path.exists(path):
            os.mkdir(path)
            print("DIRECTORY CREATED: {}".format(path))
        else:
             print("DIRECTORY ALREADY EXISTS: {}".format(path))

    crop_and_save(images_path, masks_path, new_image_path, new_mask_path, img_width, img_height)
    train_test_split(new_images_path, new_masks_path, test_to_train_ratio)