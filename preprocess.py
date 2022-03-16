import numpy as np
from numpy import asarray
#from skimage.io import imread
import os
from PIL import Image, ImageOps
import time


def preprocess(img_dir):
    img_lst = os.listdir(img_dir)

    for img_name in img_lst:

        #Resize image
        img = Image.open(os.path.join(img_dir, img_name))
        img = img.resize((256,256), Image.ANTIALIAS)
        #Greyscale image
        img = ImageOps.grayscale(img)
       
        #Transform image to np.array
        pixels = asarray(img)
        pixels = pixels.astype('float32')
        
        #DEBUG: check how many channels there are, and which mode. "L" Greyscale, "P" Palet, "RGB" RGB.
        #print(img.getbands())
        #print(img.getchannel(0))
        
        #centering
        mean = pixels.mean()
        pixels = pixels - mean

        # normalize to the range 0-1
        pixels = (pixels - pixels.min())/(pixels.max() - pixels.min())

        # Save Image
        PIL_img = Image.fromarray(np.uint8(pixels))
        PIL_img.save('PP_Data/' + img_name)

start_time = time.time()
#Kims Path: 'C:/Users/Monkk/OneDrive/Dokumenter/AAU/CS/CS2/01.P8/Data/ChinaSet_AllFiles/CXR_png'
image_directory = 'C:/Users/Dennis/Downloads/xrayTB/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png'

preprocess(image_directory)

print("--- %s seconds ---" % (time.time() - start_time))


