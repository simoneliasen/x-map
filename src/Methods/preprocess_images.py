from lib2to3.pytree import convert
from os import listdir
import os
from os.path import isfile, join
from PIL import Image
import PIL
import math

imgpath = r"..\mask_test\imgs"
imgs = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
maskpath = r"..\mask_test\masks"
masks = [f for f in listdir(maskpath) if isfile(join(maskpath, f))]

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

generate_masked_imgs()






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

def downscale():
    #for img in imgs[1:]:
    #    path = f"{imgpath}\\{img}"
    #    im = Image.open(path)
    #    width, height = im.size
    #    new_width = round(width / 10)
    #    new_height = round(height / 10)
    #    new_size = (new_width, new_height)
    #    new = im.resize(new_size)
    #    new.save(path)

    for mask in masks[1:]:
        path = f"{maskpath}\\{mask}"
        im = Image.open(path)
        width, height = im.size
        new_width = round(width / 10)
        new_height = round(height / 10)
        new_size = (new_width, new_height)
        new = im.resize(new_size)
        new.save(path)

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


