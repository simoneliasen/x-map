from lib2to3.pytree import convert
from os import listdir
import os
from os.path import isfile, join
from PIL import Image
import PIL
import math
import numpy as np

imgpath = r"C:\Users\PC\Desktop\TB_Positve"
#imgs = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
#maskpath = r"..\PP_data2\TB_negative"
#masks = [f for f in listdir(maskpath) if isfile(join(maskpath, f))]

masked_img_path = r"..\mask_test\masked_img"

def generate_masked_imgs():
    #går ud fra at hver img har 1 mask.
    for i in range(len(masks)):
        #x-ray billede.
        img = imgs[i]
        path = f"{imgpath}\\{img}"
        im = Image.open(path).convert('RGBA')
        #im.show()

        #mask
        mask = masks[i]
        path = f"{maskpath}\\{mask}"
        im_mask = Image.open(path).convert('L')
        im_mask = PIL.ImageOps.invert(im_mask) #lungerne skal være sorte.
        #im_mask.show()

        #sort baggrundsbillede
        mask_width = im.size[0]
        mask_height = im.size[1]
        black_mask = Image.new('RGB', (mask_width, mask_height), 0)

        #masked img
        im.paste(black_mask, (0, 0), im_mask)
        #im.show()
        im.save(f"{masked_img_path}\\{img}")

#generate_masked_imgs()






#BACKUP METODER: Skal justeres lidt ift. hvad der ønskes.
def removeImgsWithoutMasks():
    print(len(imgs))
    maskNames = []
    for mask in masks:
        maskNames.append(mask.replace("_mask", ""))

    inter = set(imgs) & set(maskNames)
    print(len(inter))

    for img in imgs:
        if img not in inter:
            os.remove(f"{imgpath}\\{img}")

def convert2rgb():
    for img in imgs[1:]:
        path = f"{imgpath}\\{img}"
        im = Image.open(path)
        new = im.convert("RGB")
        new.save(path)

    for mask in masks[1:]:
        path = f"{maskpath}\\{mask}"
        im = Image.open(path)
        new = im.convert("RGB")
        new.save(path)

def add_mask2name():
    for mask in masks[-3:]:
        path = f"{maskpath}\\{mask}"
        tmp = mask.replace("_mask", "")
        new_name = tmp.replace(".png", "_mask.png")
        new_path = f"{maskpath}\\{new_name}"
        os.rename(path, new_path)

def add_padding(size, img):
    #sort baggrundsbillede
    black_mask = Image.new('RGB', (size, size), 0)

    #men hvad er  upper left corner?
    img_height, img_width = img.size

    top = (size - img_height) / 2
    left = (size - img_width) / 2

    #masked img
    black_mask.paste(img, (int(top), int(left)))
    #im.show()
    return black_mask

def downscale(size):
    pospath = r"C:\Users\PC\Desktop\datasets\easy\PP_datav3\TB_Positive"
    #pospath = r"C:\Users\PC\Desktop\hejmor"
    pos = [f for f in listdir(pospath) if isfile(join(pospath, f))]

    negpath = r"C:\Users\PC\Desktop\datasets\easy\PP_datav3\TB_Negative"
    neg = [f for f in listdir(negpath) if isfile(join(negpath, f))]

    newpospath = r"C:\Users\PC\Desktop\datasets\easy\resized\TB_Positive"
    newnegpath = r"C:\Users\PC\Desktop\datasets\easy\resized\TB_Negative"

    print(len(pos))
    print(len(neg))
    for img in pos:
        try:
            path = f"{pospath}/{img}"
            im = Image.open(path)
            width, height = im.size
            aspect_ratio = width / height
            new_width = 0.0
            new_height = 0.0

            if width > height:
                new_width = size
                new_height = size / aspect_ratio
            else:
                new_height = size
                new_width = size * aspect_ratio
            
            new_size = (int(new_width), int(new_height))
            new = im.resize(new_size)
            padded = add_padding(size, new)
            new_path = f"{newpospath}/{img}"
            padded.save(new_path)
        except Exception as e:
            print(e)
            continue

    for img in neg:
        try:
            path = f"{negpath}/{img}"
            im = Image.open(path)
            width, height = im.size
            aspect_ratio = width / height
            new_width = 0.0
            new_height = 0.0

            if width > height:
                new_width = size
                new_height = size / aspect_ratio
            else:
                new_height = size
                new_width = size * aspect_ratio
            
            new_size = (int(new_width), int(new_height))
            new = im.resize(new_size)
            padded = add_padding(size, new)
            new_path = f"{newnegpath}/{img}"
            padded.save(new_path)
        except Exception as e:
            print(e)
            continue

#gjorde ingenting:
def convert2jpg():
    for mask in masks[1:]:
        path = f"{maskpath}\\{mask}"
        im = Image.open(path)
        new_path = r"C:\Users\PC\Desktop\Pytorch-UNet\data\masks_jpg"
        mask = mask.replace(".png", ".jpg")
        new_path = f"{new_path}\\{mask}"
        im.save(new_path)

    for img in imgs[1:]:
        path = f"{imgpath}\\{img}"
        im = Image.open(path)
        new_path = r"C:\Users\PC\Desktop\Pytorch-UNet\data\imgs_jpg"
        img = img.replace(".png", ".jpg")
        new_path = f"{new_path}\\{img}"
        im.save(new_path)

def convert_mask2p():
    for mask in masks:
        ext = mask.split(".")[1]
        path = f"{maskpath}\\{mask}"
        im = Image.open(path)
        new_path = r"C:\Users\PC\Desktop\Pytorch-UNet\data\masks_gif"
        mask = mask.replace(ext, ".gif")
        new_path = f"{new_path}\\{mask}"
        im = im.convert('P')
        im.save(new_path)

#for size in [299]:
    #downscale(size) #300 = inception


def split_test_train():
    pospath = r"C:\Users\PC\Desktop\datasets\easy\resized\TB_Positive"
    pos = [f for f in listdir(pospath) if isfile(join(pospath, f))]

    negpath = r"C:\Users\PC\Desktop\datasets\easy\resized\TB_Negative"
    neg = [f for f in listdir(negpath) if isfile(join(negpath, f))]

    new_pos_test_path = r"C:\Users\PC\Desktop\datasets\easy\test\TB_Positive"
    new_pos_train_path = r"C:\Users\PC\Desktop\datasets\easy\train\TB_Positive"

    new_neg_test_path = r"C:\Users\PC\Desktop\datasets\easy\test\TB_Negative"
    new_neg_train_path = r"C:\Users\PC\Desktop\datasets\easy\train\TB_Negative"

    print(len(pos))
    print(len(neg))

    #pos:
    num_train = len(pos)
    indices = list(range(num_train))
    split = int(np.floor(num_train * 0.8))
    np.random.shuffle(indices)
    test_idx, train_idx = indices[split:], indices[:split]
    
    for index in test_idx:
        file = pos[index]
        path = pospath + f"\\{file}"
        new_path = new_pos_test_path + f"\\{file}"
        os.rename(path, new_path)

    for index in train_idx:
        file = pos[index]
        path = pospath + f"\\{file}"
        new_path = new_pos_train_path + f"\\{file}"
        os.rename(path, new_path)
        

    #neg
    num_train = len(neg)
    indices = list(range(num_train))
    split = int(np.floor(num_train * 0.8))
    np.random.shuffle(indices)
    test_idx, train_idx = indices[split:], indices[:split]
    
    for index in test_idx:
        file = neg[index]
        path = negpath + f"\\{file}"
        new_path = new_neg_test_path + f"\\{file}"
        os.rename(path, new_path)

    for index in train_idx:
        file = neg[index]
        path = negpath + f"\\{file}"
        new_path = new_neg_train_path + f"\\{file}"
        os.rename(path, new_path)

def remove_subset_from_dir():
    remove_pos_path = r"C:\Users\PC\Desktop\datasets\hard\data_224\TB_Positive"
    

    remove_neg_path = r"C:\Users\PC\Desktop\datasets\hard\data_224\TB_Negative"
    

    new_neg_path = r"C:\Users\PC\Desktop\datasets\easy\resized - Kopi\TB_Negative"
    new_neg = [f for f in listdir(new_neg_path) if isfile(join(new_neg_path, f))]
    
    new_pos_path = r"C:\Users\PC\Desktop\datasets\easy\resized - Kopi\TB_Positive"
    new_pos = [f for f in listdir(new_pos_path) if isfile(join(new_pos_path, f))]

    for file_name in new_pos:
        try:
            path = f"{remove_pos_path}\\{file_name}"
            os.remove(path)

        except Exception:
            continue

    for file_name in new_neg:
        try:
            path = f"{remove_neg_path}\\{file_name}"
            os.remove(path)

        except Exception:
            continue

def move_random_x_left():
    #578 test filer fra easy
    #864 skal i test (hard)

    #dvs. der skal 864 - 578 = 286 ekstra findes fra covid osv.
    #det er dog udelukkende negative

    neg_path = r"C:\Users\PC\Desktop\datasets\hard\data_224\TB_Negative"
    neg = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]

    new_neg_path = r"C:\Users\PC\Desktop\datasets\hard\test\TB_Negative"

    num = len(neg)
    indices = list(range(num))
    split = 286
    np.random.shuffle(indices)
    test_idx = indices[:split]

    for index in test_idx:
        file = neg[index]
        path = neg_path + f"\\{file}"
        new_path = new_neg_path + f"\\{file}"
        os.rename(path, new_path)

def move_all_except_subset():
    neg_path = r"C:\Users\PC\Desktop\datasets\hard\data_224 - Kopi - Kopi\TB_Negative"
    neg = [f for f in listdir(neg_path) if isfile(join(neg_path, f))]
    
    pos_path = r"C:\Users\PC\Desktop\datasets\hard\data_224 - Kopi - Kopi\TB_Positive"
    pos = [f for f in listdir(pos_path) if isfile(join(pos_path, f))]

    new_neg_path = r"C:\Users\PC\Desktop\datasets\hard\train\TB_Negative"
    new_pos_path = r"C:\Users\PC\Desktop\datasets\hard\train\TB_Positive"


    test_neg_path = r"C:\Users\PC\Desktop\datasets\hard\test\TB_Negative"
    test_neg = [f for f in listdir(test_neg_path) if isfile(join(test_neg_path, f))]
    
    test_pos_path = r"C:\Users\PC\Desktop\datasets\hard\test\TB_Positive"
    test_pos = [f for f in listdir(test_pos_path) if isfile(join(test_pos_path, f))]

    for file in neg:
        if file not in test_neg:
            path = neg_path + f"\\{file}"
            new_path = new_neg_path + f"\\{file}"
            os.rename(path, new_path)

    for file in pos:
        if file not in test_pos:
            path = pos_path + f"\\{file}"
            new_path = new_pos_path + f"\\{file}"
            os.rename(path, new_path)
#downscale(224)

#split_test_train()

#remove_files_from_dir()

#move_random_x_left()

move_all_except_subset()