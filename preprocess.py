import numpy as np
from numpy import asarray
#from skimage.io import imread
import os
#from PIL import Image, ImageOps
import time
import shutil
import cv2

def label_files(img_dir):
    mypath = 'PP_data'
    for f in os.listdir(img_dir):
        # move if TB positive
        if f.endswith('1.png'):
            shutil.move(os.path.join(img_dir, f), os.path.join(mypath,"TB_Positive"))
        # move if TB negative
        elif f.endswith('0.png'):
            shutil.move(os.path.join(img_dir, f), os.path.join(mypath, "TB_Negative"))
        # notify if something else
        else:
            print('Could not categorize file with name %s' % f)
    


def preprocess(img_dir):
    img_lst = os.listdir(img_dir)

    for img_name in img_lst:

        #Resize image
        img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
        before_img = asarray(img)
        #pixels = pixels.astype('float32')
        print("before normalization")        
        print(before_img)
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #cv2.imshow('hej', norm_img)
        #cv2.waitKey(0)
        cv2.imwrite('PP_Data/' + img_name, 255*norm_img)
        #Transform image to np.array
        pixels = asarray(norm_img)
        print("after normalization")
        print(pixels)
        
        

start_time = time.time()
#Kims Path: 'C:/Users/Monkk/OneDrive/Dokumenter/AAU/CS/CS2/01.P8/Data/ChinaSet_AllFiles/CXR_png'
#Dennis Stationær: D:/Downloads/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png
#Dennis laptop: C:/Users/Dennis/Downloads/xrayTB/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png'

img_dir ='D:/Downloads/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png'

# preprocess(image_directory)
label_files(img_dir)

print("--- %s seconds ---" % (time.time() - start_time))


