# Semantic Segmentation of Roads in aerial imagery. 

Semantic segmentation is the process of classifying each pixel of an image into distinct classes using deep learning. This aids in identifying regions in an image where certain objects reside. 

This aim of this project is to identify and segment roads in aerial imagery. Detecting roads can be an important factor in predicting further development of cities, and this concept plays a major role in GeoArchitect (A project which I started). Segmentation of roads is important to map-based applications and is used for finding distances or shortest routes between two places.

Read more about this project here.

## Contents:
1. Dataset.
2. Manipulating the data.
3. About the model.


## 1. Dataset

For this challenge, I used the [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/). This dataset contains aerial images, along with the target masks. You can use download_images.py to download all the images mentioned in this site. If you have internet connections that may fluctuate then downloading the data using a torrent client would be a smart way to take. You can download the images from academic torrents, and you can find the dataset [here](http://academictorrents.com/details/3b17f08ed5027ea24db04f460b7894d913f86c21).

The dataset contains 1171 images and respectiv masks. Both the masks and the images are 1500x1500 in the resolution are present in the .tiff format. Have a look at the following sample.

![Samples](https://github.com/Paulymorphous/Road-Segmentation/blob/master/Images/Sample.jpg)

## 2. Manipulating the data

The pre-processing steps involved: 
1. Removed images where more than 25% of the map was missing.
2. Cropped 256x256 images out of the images. Hence, increasing the total number of images to more than 22,000.
3. Binarized the mask so that the pixel value is always between 0 and 1.

## 3. About the model.

To solve this problem, I used an Unet, it is a fully convolutional network, with 3 cross-connections. Adam optimiser with a learning rate of 0.00001 was used, along with dice loss (because of the unbalanced nature of the dataset.) 
The model trained for 61 epochs before earlstopper kicked in and killed the training process. A validation dice loss of 0.7548 was achieved.

The model can be found in Models/road_mapper_final.h5.



