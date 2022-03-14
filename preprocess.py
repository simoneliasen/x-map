import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.image as mpimg
import os
from PIL import Image
import glob


def preprocess(directory):
    img_lst = os.listdir(directory)

    for i in img_lst[0:3]:

        #Resize image
        img = Image.open(os.path.join(directory, i))
        print(img.size)
        img = img.resize((256,256), Image.ANTIALIAS)
        print(img.size)
        img.show()




#img = mpimg.imread(f)
#imgplot = plt.imshow(img)
#plt.show()

preprocess('C:/Users/Monkk/OneDrive/Dokumenter/AAU/CS/CS2/01.P8/Data/ChinaSet_AllFiles/CXR_png')


        
#img = mpimg.imread('Data/CHNCXR_0001_0.png')
#imgplot = plt.imshow(img)
#plt.show()



