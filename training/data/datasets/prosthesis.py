# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Prosthesis Dataset for VGGT finetuning.

This dataset is adapted for monocular depth estimation on prosthesis data.
Since camera poses are not available, we use identity matrices for extrinsics
and default intrinsics. The model will be fine-tuned using only depth loss.
"""

import os.path as osp
import os
import logging
import json
import random
import numpy as np

import cv2

from data.dataset_util import *
from data.base_dataset import BaseDataset


class ProsthesisDataset(BaseDataset):
    """
    Dataset class for prosthesis depth estimation data.
    
    This dataset loads RGB images and depth maps from the prosthesis dataset.
    Since camera poses are not available, we use identity matrices for extrinsics
    and estimate intrinsics from image dimensions.
    """
    
    def __init__(
        self,
        common_conf,
        split: str = "train",
        PROSTHESIS_DIR: str = None,
        split_file: str = None,
        min_num_images: int = 1,
        len_train: int = 100000,
        len_test: int = 10000,
        focal_length: float = 1.0,
    ):
        """
        Initialize the ProsthesisDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            PROSTHESIS_DIR (str): Directory path to prosthesis data.
            split_file (str): Path to split file containing image paths.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            focal_length (float): Default focal length for camera intrinsics.
        Raises:
            ValueError: If PROSTHESIS_DIR or split_file is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if PROSTHESIS_DIR is None:
            raise ValueError("PROSTHESIS_DIR must be specified.")
        
        if split_file is None:
            raise ValueError("split_file must be specified.")

        self.PROSTHESIS_DIR = PROSTHESIS_DIR
        self.split_file = split_file
        self.focal_length = focal_length
        self.min_num_images = min_num_images

        # Load split file
        logging.info(f"Loading split file: {split_file}")
        self.data_store = self._load_split_file(split_file)
        
        # Group images by sequence
        self.sequence_data = self._group_by_sequence()
        
        self.sequence_list = list(self.sequence_data.keys())
        self.sequence_list_len = len(self.sequence_list)
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Prosthesis Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Prosthesis Data dataset length: {len(self)}")

    def _load_split_file(self, split_file):
        """Load split file and parse image paths."""
        data_store = {}
        
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                rgb_path = parts[0]
                depth_path = parts[1]
                
                # Extract sequence name from path
                # Expected format: sequence_name/sharpen_rgb/PNG/imageXXX.png
                path_parts = rgb_path.split('/')
                if len(path_parts) >= 1:
                    seq_name = path_parts[0]
                    
                    if seq_name not in data_store:
                        data_store[seq_name] = []
                    
                    data_store[seq_name].append({
                        'rgb_path': rgb_path,
                        'depth_path': depth_path,
                        'frame_id': idx
                    })
        
        return data_store

    def _group_by_sequence(self):
        """Group images by sequence and filter by minimum number of images."""
        sequence_data = {}
        
        for seq_name, frames in self.data_store.items():
            if len(frames) >= self.min_num_images:
                sequence_data[seq_name] = frames
            else:
                logging.warning(f"Skipping sequence {seq_name} with only {len(frames)} images (min: {self.min_num_images})")
        
        return sequence_data

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.sequence_data[seq_name]

        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )

        annos = [metadata[i] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for anno in annos:
            rgb_path = osp.join(self.PROSTHESIS_DIR, anno["rgb_path"])
            depth_path = osp.join(self.PROSTHESIS_DIR, anno["depth_path"])

            image = read_image_cv2(rgb_path)

            if self.load_depth:
                depth_map = read_depth(depth_path, 1.0)
                
                # Threshold depth map to remove outliers
                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            
            # Create dummy extrinsics (identity matrix for camera-from-world)
            # Since we don't have camera poses, we use identity
            extri_opencv = np.eye(4, dtype=np.float32)[:3, :] 
            
            # Create default intrinsics based on image size and focal length
            H, W = original_size
            fx = fy = self.focal_length * max(H, W)
            cx = W / 2.0
            cy = H / 2.0
            intri_opencv = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=rgb_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(rgb_path)
            original_sizes.append(original_size)

        set_name = "prosthesis"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch
