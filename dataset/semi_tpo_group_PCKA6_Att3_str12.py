from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None, tpo_split=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.tpo_split = tpo_split

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]

            if mode == 'train_u':
                self.ids = self.ids * 12
        else:
            with open('splits/%s/val_894.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

        self.id_to_trainid = {0: 0, 38: 1, 75: 2}

        self.jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
        self.RandomGrayscale = transforms.RandomGrayscale(p=0.2)


    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def generate_non_overlapping_masks(self, width, height, grid_rows, grid_cols):
        cell_width, cell_height = width // grid_cols, height // grid_rows
        masks = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                x1, y1 = j * cell_width, i * cell_height
                x2, y2 = (j + 1) * cell_width, (i + 1) * cell_height
                masks.append((x1, y1, x2, y2))
        random.shuffle(masks)  # Shuffle to ensure randomness
        return masks

    def apply_random_masks_to_images(self, images, masks, width, height):
        masked_image_paths = []
        for index, imgs in enumerate(images):

            # Get a random mask and apply it to the image
            mask_rect = masks[index]
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(mask_rect, fill=255)

            masked_image = Image.new('RGB', (width, height))
            masked_image.paste(imgs, (0, 0), mask)

            # Save the masked image
            # masked_image_path = f'./masked_image_{index + 1}.png'
            # masked_image.save(masked_image_path)
            masked_image_paths.append(masked_image)

        return masked_image_paths

    def reconstruct_from_masked_parts(self, masked_image_paths, width, height):
            # Assuming all images are of the same size
            # sample_image = Image.open(masked_image_paths[0])
            # width, height = masked_image_paths[0].size

            # Initialize an image for reconstruction (with a white background or any other color of your choice)
            base_image = Image.new('RGB', (width, height), 'white')  # Change to 'RGB' and choose a background color

            # Overlay the masked images to reconstruct the original image
            for masked_path in masked_image_paths:
                # Convert to 'RGB' if the image is not already in that mode

                # Create a mask for non-black (or non-background) pixels
                mask = Image.eval(masked_path, lambda x: 255 if x != 0 else 0).convert('1')

                # Use paste() to combine the images
                base_image.paste(masked_path, (0, 0), mask)

            return base_image

    def __getitem__(self, item):
        if self.mode == 'train_u':
            id = self.ids[item]
            # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            # mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
            tmp_img = os.path.join(self.root, self.tpo_split, 'img')
            tmp_lab = os.path.join(self.root, self.tpo_split, 'lab')
            img_names = os.listdir(os.path.join(tmp_img, id))
            lab_name = os.listdir(os.path.join(tmp_lab, id))

            # img_names = sorted(img_names, key=lambda x: int(''.join(filter(str.isdigit, x))))
            img_names = random.sample(img_names, 6)

            # random.shuffle(img_names)
            imgs = []
            for img_name in img_names:
                img_tmp = Image.open(os.path.join(tmp_img, id, img_name)).convert('RGB')
                imgs.append(img_tmp)


            mask = cv2.imread(os.path.join(tmp_lab, id, lab_name[0]), cv2.IMREAD_GRAYSCALE)
            mask = self.id2trainId(mask)
            mask = Image.fromarray(mask)

            # if self.mode == 'val':
            #     imgs, mask = normalizes(imgs, mask)
            #     return imgs, mask, id
            #
            imgs, mask = resizes(imgs, mask, (0.5, 2.0))
            ignore_value = 254 if self.mode == 'train_u' else 255
            imgs, mask = crops(imgs, mask, self.size, ignore_value)
            imgs, mask = hflips(imgs, mask, p=0.5)
            #
            # if self.mode == 'train_l':
            #     return normalizes(imgs, mask)

            # reconstructed_image = imgs.pop()

            img_w, img_s1, img_s2 = deepcopy(imgs), deepcopy(imgs[1]), deepcopy(imgs[2])

            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
            img_s1 = blur(img_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

            # if random.random() < 0.8:
            #     img_s1 = [self.jitter(img) for img in img_s1]
            # img_s1 = [blur(self.RandomGrayscale(img), p=0.5) for img in img_s1]
            # cutmix_box1 = obtain_cutmix_box(img_s1[0].size[0], p=0.5)

            if random.random() < 0.8:
                img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
            img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
            img_s2 = blur(img_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

            # if random.random() < 0.8:
            #     img_s2 = [self.jitter(img) for img in img_s2]
            # img_s2 = [blur(self.RandomGrayscale(img), p=0.5) for img in img_s2]
            # cutmix_box2 = obtain_cutmix_box(img_s2[0].size[0], p=0.5)

            ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

            img_s1, ignore_mask = normalize(img_s1, ignore_mask)
            img_s2 = normalize(img_s2)

            mask = torch.from_numpy(np.array(mask)).long()
            ignore_mask[mask == 254] = 255

            return normalizes(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

        if self.mode == 'val':
            id = self.ids[item]
            # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            # mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
            img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]), cv2.IMREAD_GRAYSCALE)
            mask = self.id2trainId(mask)
            mask = Image.fromarray(mask)

            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train_l':

            id = self.ids[item]
            # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
            # mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
            tmp_img = os.path.join(self.root, self.tpo_split, 'img')
            tmp_lab = os.path.join(self.root, self.tpo_split, 'lab')
            img_names = os.listdir(os.path.join(tmp_img, id))
            lab_name = os.listdir(os.path.join(tmp_lab, id))

            # img_names = sorted(img_names, key=lambda x: int(''.join(filter(str.isdigit, x))))
            img_names = random.sample(img_names, 6)

            # random.shuffle(img_names)
            imgs = []
            for img_name in img_names:
                img_tmp = Image.open(os.path.join(tmp_img, id, img_name)).convert('RGB')
                imgs.append(img_tmp)


            mask = cv2.imread(os.path.join(tmp_lab, id, lab_name[0]), cv2.IMREAD_GRAYSCALE)
            mask = self.id2trainId(mask)
            mask = Image.fromarray(mask)

            imgs, mask = resizes(imgs, mask, (0.5, 2.0))
            ignore_value = 254 if self.mode == 'train_u' else 255
            imgs, mask = crops(imgs, mask, self.size, ignore_value)
            imgs, mask = hflips(imgs, mask, p=0.5)

            return normalizes(imgs, mask)





    def __len__(self):
        return len(self.ids)
