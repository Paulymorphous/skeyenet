# Imports 
import numpy as np

from tensorflow.keras.models import load_model
from loss_functions import dice_coef as dice_coef_loss, iou_coef, soft_dice_loss
from tensorflow.keras.preprocessing.image import load_img
from skimage import transform
from PIL import Image

# Global Variables
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 256, 256, 3
ORIG_HEIGHT, ORIG_WIDTH = 0, 0

# Gives a tensor of size (1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
def image_makeup(img_filepath):
    np_img = load_img(img_filepath)
    global ORIG_HEIGHT, ORIG_WIDTH
    ORIG_HEIGHT, ORIG_WIDTH = np_img.size
    np_img = np.array(np_img).astype('float32') 
    np_img = transform.resize(np_img, (256, 256, 3))
    np_img = np.expand_dims(np_img, axis=0)
    return np_img

def clean_up_predictions(preds) -> list:
    threshold = 0.05
    preds = 255 * (preds > threshold).astype('uint8')
    imgs = []
    for i in range(len(preds)):
        image = np.squeeze(preds[i][:, :, 0])
        image = Image.fromarray(image)
        image = image.resize((ORIG_HEIGHT, ORIG_WIDTH))
        imgs.append(image)

def predict(img_path) -> list:
    model = load_model("./Models/road_mapper_final.h5", custom_objects = {
        "soft_dice_loss" : soft_dice_loss,
        "iou_coef" : iou_coef,
        "dice_coef_loss" : dice_coef_loss,
        "dice_loss" : dice_coef_loss,
    })

    preds = model.predict(image_makeup(img_path))
    imgs_list = clean_up_predictions(preds)
    return imgs_list 




