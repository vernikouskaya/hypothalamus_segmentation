import os
import random
import numpy as np
from skimage import io, color
import cv2


images_path = './MRI' # series of MRI images
masks_path = './pred' # series of predicted masks
img_overlay_path = './overlayMRI'

images = []
masks = []
images_name = []

for im in os.listdir(images_path):
    images.append(os.path.join(images_path,im))
    images_name.append(im.split('.')[0])

for msk in os.listdir(masks_path):
    masks.append(os.path.join(masks_path,msk))

alpha = 1
i = 1

while i<=len(images):

    original_image = io.imread(images[i-1])
    original_mask = io.imread(masks[i-1])


    img_color = original_image
    img_half = cv2.resize(img_color, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    #mask_half = cv2.resize(original_mask[:,:,0], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    mask_color = color.label2rgb(original_mask, colors=['red'], alpha=1.0, bg_label=0)
    #mask_color = color.label2rgb(mask_half, colors=['red'], alpha=1.0, bg_label=0)
    #mask_color = color.label2rgb(original_mask[:,:,0], bg_label=0)
    # DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
    #               'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
    img_hsv = color.rgb2hsv(img_half)
    color_mask_hsv = color.rgb2hsv(mask_color)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    new_image_path = "%s/%s.png" % (img_overlay_path, images_name[i-1])
    io.imsave(new_image_path, img_masked)

    i = i+1