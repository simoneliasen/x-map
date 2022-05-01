from lib2to3.pytree import convert
from os import listdir
import os
from os.path import isfile, join
from PIL import Image
import PIL
import math

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
    pospath = r"/content/drive/MyDrive/6. PP_data/PP_data_v2/TB_Positive"
    #pospath = r"C:\Users\PC\Desktop\hejmor"
    pos = [f for f in listdir(pospath) if isfile(join(pospath, f))]

    negpath = r"/content/drive/MyDrive/6. PP_data/PP_data_v2/TB_Negative"
    neg = [f for f in listdir(negpath) if isfile(join(negpath, f))]

    newpospath = r"/content/drive/MyDrive/6. PP_data/PP_data_v2_resized_cropped/TB_Positive"
    newnegpath = r"/content/drive/MyDrive/6. PP_data/PP_data_v2_resized_cropped/TB_Negative"

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

    #for img in neg:
        try:
            path = f"{negpath}/{img}"
            im = Image.open(path)
            width, height = im.size
            aspect_ratio = width / height
            new_width = 0.0
            new_height = 0.0

            if width > height:
                new_width = size * aspect_ratio
                new_height = size
            else:
                new_height = size * aspect_ratio
                new_width = size
            
            new_size = (int(new_width), int(new_height))
            new = im.resize(new_size)
            new_path = f"{newnegpath}/{img}"
            new.save(new_path)
        except:
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

downscale(256)

