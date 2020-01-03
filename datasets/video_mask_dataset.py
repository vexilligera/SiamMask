from datasets.siam_mask_dataset import DataSets
from utils.bbox_helper import *
import random
import cv2
import numpy as np

class VideoMaskDataset(DataSets):
    def __init__(self, cfg, anchor_cfg, num_epoch=1):
        super(VideoMaskDataset, self).__init__(cfg, anchor_cfg, num_epoch)
        
    def __getitem__(self, index, debug=False):
        index = self.pick[index]
        dataset, index = self.find_dataset(index)

        gray = self.gray and self.gray > random.random()
        neg = self.neg and self.neg > random.random()
            
        template, search = dataset.get_positive_pair(index, video_data=True)
        if neg:
            template = dataset.get_random_target(index)

        def center_crop(img, size):
            shape = img.shape[1]
            if shape == size: return img
            c = shape // 2
            l = c - size // 2
            r = c + size // 2 + 1
            return img[l:r, l:r]

        template_image, scale_z = self.imread(template[0])

        if self.template_small:
            template_image = center_crop(template_image, self.template_size)

        search_images, scale_xs = tuple(zip(*[self.imread(i[0]) for i in search]))

        if dataset.has_mask and not neg:
            search_masks = [(cv2.imread(i[2], 0) > 0).astype(np.float32) for i in search]
        else:
            search_masks = [np.zeros(i.shape[:2], dtype=np.float32) for i in search_images]

        if self.crop_size > 0:
            search_images = [center_crop(i, self.crop_size) for i in search_images]
            search_masks = [center_crop(i, self.crop_size) for i in search_masks]

        def toBBox(image, shape):
            imh, imw = image.shape[:2]
            if len(shape) == 4:
                w, h = shape[2]-shape[0], shape[3]-shape[1]
            else:
                w, h = shape
            context_amount = 0.5
            exemplar_size = self.template_size  # 127
            wc_z = w + context_amount * (w+h)
            hc_z = h + context_amount * (w+h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = exemplar_size / s_z
            w = w*scale_z
            h = h*scale_z
            cx, cy = imw//2, imh//2
            bbox = center2corner(Center(cx, cy, w, h))
            return bbox

        template_box = toBBox(template_image, template[1])
        search_boxes = [toBBox(i[0], i[1][1]) for i in zip(search_images, search)]

        template, tbox, _ = self.template_aug(template_image, template_box, self.template_size, gray=gray)
        rand_list = self.search_aug.gen_random_list()
        aug = [self.search_aug(search_images[i], search_boxes[i], self.search_size, gray=gray,
                mask=search_masks[i], rand_list=rand_list) for i in range(len(search_images))]
        search, bboxes, masks = list(zip(*aug))

        def draw(image, box, name):
            image = image.copy()
            x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.imwrite(name, image)

        if debug:
            draw(template_image, template_box, "debug/{:06d}_ot.jpg".format(index))
            draw(template, tbox, "debug/{:06d}_t.jpg".format(index))
            for i in range(len(search_images)):
                draw(search_images[i], search_boxes[i], "debug/{:06d}_os_{:04d}.jpg".format(index, i))
                draw(search[i], bboxes[i], "debug/{:06d}_s_{:04d}.jpg".format(index, i))

        _ = [self.anchor_target(self.anchors, bbox, self.size, neg) for bbox in bboxes]
        clses, deltas, delta_weights = list(zip(*_))

        if dataset.has_mask and not neg:
            mask_weight = [cls.max(axis=0, keepdims=True) for cls in clses]
        else:
            mask_weight = [np.zeros([1, cls.shape[1], cls.shape[2]], dtype=np.float32) for cls in clses]

        template = np.transpose(template, (2, 0, 1)).astype(np.float32)
        search = np.stack(search, axis=0)
        search = np.transpose(search, (0, 3, 1, 2)).astype(np.float32)

        masks = np.concatenate([(np.expand_dims(mask, axis=0) > 0.5) * 2 - 1 for mask in masks], axis=0).astype(np.float32)
        clses = np.stack(clses, axis=0)
        deltas = np.stack(deltas, axis=0)
        delta_weights = np.stack(delta_weights, axis=0)

        return template, search, clses, deltas, delta_weights, np.stack(bboxes, axis=0).astype(np.float32), \
                masks, np.concatenate(mask_weight, axis=0).astype(np.float32)
