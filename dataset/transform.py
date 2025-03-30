import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask

def crop_visu(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = 448
    y = 448
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask

def crops(imgs, mask, size, ignore_value=255):
    w, h = imgs[0].size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0

    imgs = [ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) for img in imgs]
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = imgs[0].size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    imgs = [img.crop((x, y, x + size, y + size)) for img in imgs]
    mask = mask.crop((x, y, x + size, y + size))

    return imgs, mask

def crops_visu(imgs, mask, size, ignore_value=255):
    w, h = imgs[0].size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0

    imgs = [ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) for img in imgs]
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = imgs[0].size
    x = 448
    y = 448

    imgs = [img.crop((x, y, x + size, y + size)) for img in imgs]
    mask = mask.crop((x, y, x + size, y + size))

    return imgs, mask

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def hflips(imgs, mask, p=0.5):
    if random.random() < p:
        imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return imgs, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def normalizes(imgs, mask=None):
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transformed_imgs = [transform_pipeline(img) for img in imgs]
    # stacked_tensors = torch.stack(transformed_imgs, dim=0)
    # mean_tensor = stacked_tensors.mean(dim=0)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return transformed_imgs, mask
    return transformed_imgs


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

def resizes(imgs, mask, ratio_range):
    w, h = imgs[0].size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    imgs = [img.resize((ow, oh), Image.BILINEAR) for img in imgs]

    mask = mask.resize((ow, oh), Image.NEAREST)
    return imgs, mask

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
