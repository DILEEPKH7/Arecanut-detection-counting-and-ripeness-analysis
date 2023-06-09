#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import cv2
import numpy as np
import torch
import os
import os.path as osp
import random
import json
import time
import hashlib


from pathlib import Path
from io import UnsupportedOperation
from multiprocessing.pool import Pool
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

from events import LOGGER

# parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

# Get orientation exif tag and it is required to check the image files
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

# Task 1 - Augument the data using augmentation hyperparameters.
# Task 2 - check images when initializing datasets - Means check whether the image can be opened or not. If yes, make sure that the image is
# not oriented. If the image format is JPG or JPEG, make sure that the images are not corrupted. If corrupted, try to correct them. If the image
# is not opening doing the above the steps, return the image as corrupted.
# Task 3 - check labels when initializing datasets. Check whether the images are missing, or label format, normalised or not
# Task 4 - rectangular training. Not priority for now. Check it later
# Task 5 - Create a JSON file, that contains image path, shape and labels information + hash values.

# Subtask - Check whether we need hash values or not

class TrainValDataset(Dataset):
    '''Load images and labels for training and validation'''
    def __init__(
        self,
        img_dir,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None, # hyperparameters of augmentation
        rect=False, # rectangular training is not be applied
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0, # check 
        rank=-1,
        data_dict=None,
        task="train"
    ):
        assert task.lower() in ("train", "val", "test", "speed"), f'Not supported task: {task}'
        t1 = time.time()
        self.__dict__.update(locals()) # better way of assining input values to self instead of assigning each individual to self
        self.main_process = self.rank in (-1,0) # self.main_process == True if self.rank in (-1,0) else False
        self.task = self.task.capitalize()
        #self.class_names = data_dict["names"]
        self.img_paths, self.labels = self.get_imgs_labels(self.img_dir)
        t2 = time.time()
        if self.main_process:
            LOGGER.info(f"%.1fs for dataset initialization." % (t2 - t1))

    def __len__(self):
        """Get the length of dataset"""
        return len(self.img_paths)
    
    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes
    
    def __getitem__(self, index):
        img, (h0, w0), (h, w) = self.load_image(index)
        shape = (self.img_size)
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

        shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # for COCO mAP rescaling
        labels = self.labels[index].copy()
        if labels.size:
            w *= ratio
            h *= ratio
            # new boxes
            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = (
                w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
            )  # top left x
            boxes[:, 1] = (
                h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
            )  # top left y
            boxes[:, 2] = (
                w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
            )  # bottom right x
            boxes[:, 3] = (
                h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
            )  # bottom right y
            labels[:, 1:] = boxes   
        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes

        if self.augment:
            img, labels = self.general_augment(img, labels)

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_paths[index], shapes

    def load_image(self, index, force_load_size=None):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        path = self.img_paths[index]
        try:
            im = cv2.imread(path)
            assert im is not None, f"opencv cannot read image correctly or {path} not exists"
        except:
            im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
            assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

        h0, w0 = im.shape[:2]  # origin shape
        if force_load_size:
            r = force_load_size / max(h0, w0)
        else:
            r = self.img_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(
                im,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR,
            )
        return im, (h0, w0), im.shape[:2]
    
    def get_imgs_labels(self, img_dir):

        assert osp.exists(img_dir), f'{img_dir} is an invalid directory path!'
        valid_img_record = osp.join(
            osp.dirname(img_dir),"." + osp.basename(img_dir) + ".json"
        )

        NUM_THREADS = min(8, os.cpu_count())

        img_paths = glob.glob(osp.join(img_dir, "**/*"), recursive=True) # create image paths
        img_paths = sorted(
            p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p)
        )

        assert img_paths, f'No images found in {img_dir}.'

        img_hash = self.get_hash(img_paths) # hash values for each img_path
        if osp.exists(valid_img_record):
            with open(valid_img_record, "r") as f:
                cache_info = json.load(f)
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:
                    img_info = cache_info["information"] # gives image path, shape and labels.
                else:
                    self.check_images = True
        else:
            self.check_images = True

        if self.check_images and self.main_process:
            img_info={}
            nc, msgs = 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}:  Checking formats of images with {NUM_THREADS} process(es): "        
            )
            with Pool(NUM_THREADS) as pool:
                pbar = tqdm(
                    pool.imap(TrainValDataset.check_image, img_paths),
                    total=len(img_paths),
                )
                for img_path, shape_per_img, nc_per_img, msg in pbar:
                    if nc_per_img == 0:  # not corrupted
                        img_info[img_path] = {"shape": shape_per_img}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nc} image(s) corrupted"
            pbar.close()
            if msgs:
                LOGGER.info("\n".join(msgs))

            cache_info = {"information": img_info, "image_hash": img_hash}
            # save valid image paths.
            with open(valid_img_record, "w") as f:
                json.dump(cache_info, f)
        
        base_dir = osp.basename(img_dir)
        if base_dir !="":
            label_dir = osp.join(
                osp.dirname(osp.dirname(img_dir)),"labels", osp.basename(img_dir)
            )
            assert osp.exists(label_dir), f'{label_dir} is is an invalid directory path!'
        else:
            sub_dirs=[]
            label_dir = img_dir
            for rootdir, dirs, files in os.walk(label_dir):
                for subdir in dirs:
                    sub_dirs.append(subdir)
            assert "labels" in sub_dirs, f"Could not find a labels directory!"


        def _new_rel_path_with_ext(base_path: str, full_path: str, new_ext: str):
            rel_path = osp.relpath(full_path,base_path)
            return osp.join(osp.dirname(rel_path), osp.splitext(osp.basename(rel_path))[0] + new_ext)
        
        img_paths=list(img_info.keys())
        label_paths = sorted(
            osp.join(label_dir, _new_rel_path_with_ext(img_dir, p, ".txt"))
            for p in img_paths
        )
        assert label_paths, f"No labels found in {label_dir}."
        label_hash = self.get_hash(label_paths)
        if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
            self.check_labels = True
        
        if self.check_labels:
            cache_info["label_hash"] = label_hash
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )
            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    TrainValDataset.check_label_files, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths)) if self.main_process else pbar
                for (
                    img_path,
                    labels_per_file,
                    nc_per_file,
                    nm_per_file,
                    nf_per_file,
                    ne_per_file,
                    msg,
                ) in pbar:
                    if nc_per_file == 0:
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    if self.main_process:
                        pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            if self.main_process:
                pbar.close()
                with open(valid_img_record, "w") as f:
                    json.dump(cache_info, f)
            if msgs:
                LOGGER.info("\n".join(msgs))
            if nf == 0:
                LOGGER.warning(
                    f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
                )

        if self.task.lower() == "val":
            if self.data_dict.get("is_coco", False): # use original json file when evaluating on coco dataset.
                assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.class_names
                ), "Class names is required when converting labels to coco format for evaluating."
                save_dir = osp.join(osp.dirname(osp.dirname(img_dir)), "annotations")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = osp.join(
                    save_dir, "instances_" + osp.basename(img_dir) + ".json"
                )
                TrainValDataset.generate_coco_format_labels(
                    img_info, self.class_names, save_path
                )

        img_paths, labels = list(
            zip(
                *[
                    (
                        img_path,
                        np.array(info["labels"], dtype=np.float32)
                        if info["labels"]
                        else np.zeros((0, 5), dtype=np.float32),
                    )
                    for img_path, info in img_info.items()
                ]
            )
        )
        self.img_info = img_info
        LOGGER.info(
            f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. "
        )
        return img_paths, labels

    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_w, img_h = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Resutls saved in {save_path}"
            )

    @staticmethod
    def check_image(im_file):
        '''Verify an image.'''
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()  # PIL verify
            im = Image.open(im_file)  # need to reload the image after using verify()
            shape = im.size  # (width, height)
            try:
                im_exif = im._getexif()
                if im_exif and ORIENTATION in im_exif: # make sure that the images are not oriented
                    rotation = im_exif[ORIENTATION]
                    if rotation in (6, 8): # 6 - means rotated anticlockwise 90 degree, 8 - means rotated clockwise 90 
                        shape = (shape[1], shape[0])
            except:
                im_exif = None
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in (6, 8):
                    shape = (shape[1], shape[0])

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels" # making sure that image is > 10*10 pixels
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f: # rb - read the file in binary mode
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
            return im_file, None, nc, msg

    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True)
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # remove duplicates
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return img_path, None, nc, nm, nf, ne, msg

    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, return_int=False):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)
        
# if __name__ == '__main__':
#     a = TrainValDataset(        
#         img_dir='data/areca_100/images/train',
#         img_size=640,
#         batch_size=16,
#         augment=False,
#         hyp=None, # hyperparameters of augmentation
#         rect=False, # rectangular training is not be applied
#         check_images=False,
#         check_labels=False,
#         stride=32,
#         pad=0.0, # check 
#         rank=-1,
#         data_dict=None,
#         task="train")
#     print(a)