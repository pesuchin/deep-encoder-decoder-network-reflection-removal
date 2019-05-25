import os
import cv2
import numpy as np
import cupy as xp
import random
from pathlib import Path

import chainer
from chainer.dataset import dataset_mixin
from chainer import training
from chainer import iterators
from chainer.training import extensions
from .model import ReflectionRemovalNet


def augument_random_crop(transmission_img, reflection_img, transmission_crop_size=(128, 128), reflection_crop_size=(256, 256)):
    transmission_img = np.array(transmission_img, dtype=np.float32)
    reflection_img = np.array(reflection_img, dtype=np.float32)
    h, w, _ = transmission_img.shape
    h_ref, w_ref, _ = reflection_img.shape

    # randomly select crop start possition (top and left) for transmission image
    top = np.random.randint(0, h - transmission_crop_size[1])
    left = np.random.randint(0, w - transmission_crop_size[0])
    # randomly select crop start possition (top and left) for reflection image
    top_ref = np.random.randint(0, h_ref - reflection_crop_size[1])
    left_ref = np.random.randint(0, w_ref - reflection_crop_size[0])

    # calc top and right bound for transmission image
    bottom = top + transmission_crop_size[1]
    right = left + transmission_crop_size[0]
    # calc top and right bound for reflection image
    bottom_ref = top_ref + reflection_crop_size[1]
    right_ref = left_ref + reflection_crop_size[0]

    # crop image
    transmission_img = transmission_img[top:bottom, left:right, :]
    reflection_img = reflection_img[top_ref:bottom_ref, left_ref:right_ref, :]
    return transmission_img, reflection_img


def prepare_dataset_path(folder_path):
    current_dir = Path.cwd()
    transmission_img_dir = current_dir / folder_path / 'transmission_img'
    reflection_img_dir = current_dir / folder_path / 'reflection_img'
    transmission_img_paths = transmission_img_dir.glob('**/*.png')
    reflection_img_paths = reflection_img_dir.glob('**/*.png')
    return transmission_img_paths, reflection_img_paths


def _create_composed_image_as_array(transmission_img, reflection_img, alpha_range=(0.75, 0.8), gaussian_blur_var_range=(1, 5)):
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    gaussian_blur_var = np.random.randint(gaussian_blur_var_range[0],  gaussian_blur_var_range[1])
    G = cv2.GaussianBlur(reflection_img, (3, 3), gaussian_blur_var)
    image_blur = (alpha * transmission_img) + ((1 - alpha) * G)
    return image_blur


class ImagePairDataset(dataset_mixin.DatasetMixin):

    def __init__(self, transmission_img_paths, reflection_img_paths,
                 root='.', transmission_crop_size=(128, 128),
                 reflection_crop_size=(256, 256),
                 augmentation=True, gaussian_blur_var_range=(1, 5),
                 alpha_range=(0.75, 0.8)):
        self._transmission_img_paths = transmission_img_paths
        self._reflection_img_paths = reflection_img_paths
        self._root = root
        self._augmentation = augmentation
        self._transmission_crop_size = transmission_crop_size
        self._reflection_crop_size = reflection_crop_size
        self._alpha_range = alpha_range
        self._gaussian_blur_var_range = gaussian_blur_var_range

    def __len__(self):
        return len(self._transmission_img_paths)

    def get_example(self, i):
        transmission_img_path = os.path.join(self._root, self._transmission_img_paths[i])
        ref_index = np.random.randint(0, len(self._reflection_img_paths))
        reflection_img_path = os.path.join(self._root, self._reflection_img_paths[ref_index])
        transmission_img = cv2.imread(transmission_img_path)
        reflection_img = cv2.imread(reflection_img_path)
        if self._augmentation:
            transmission_img, reflection_img = augument_random_crop(
                transmission_img, reflection_img, transmission_crop_size=self._transmission_crop_size, reflection_crop_size=self._reflection_crop_size)
            reflection_img = cv2.resize(reflection_img, self._transmission_crop_size)

        image_blur = _create_composed_image_as_array(transmission_img,
                                                     reflection_img,
                                                     alpha_range=self._alpha_range,
                                                     gaussian_blur_var_range=self._gaussian_blur_var_range)
        h, w, ch = image_blur.shape
        image_blur = xp.array(image_blur, dtype=np.float32).transpose(2, 1, 0) / 255.
        transmission_img = xp.array(transmission_img, dtype=np.float32).transpose(2, 1, 0) / 255.
        return image_blur, transmission_img


if __name__ == '__main__':
    iteration = 200000
    out_dir = './result/'
    batchsize = 16
    gpu = 0

    snapshot_interval = (2000, 'iteration')
    report_interval = (100, 'iteration')
    rate_change_trigger = (20000, 'iteration')

    model = ReflectionRemovalNet(lambda_variable=0.001)
    optimizer = chainer.optimizers.Adam(alpha=0.002)
    optimizer.setup(model)

    transmission_img_paths, reflection_img_paths = prepare_dataset_path('dataset')
    train_dataset = ImagePairDataset(transmission_img_paths, reflection_img_paths, '')
    train_iter = iterators.MultiprocessIterator(train_dataset, batchsize,
                                                repeat=True, shuffle=True,
                                                n_processes=4)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out_dir)
    trainer.extend(extensions.ExponentialShift('alpha', rate=0.1),
                   trigger=rate_change_trigger)
    # trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=report_interval))
    trainer.extend(extensions.PrintReport(['iteration', 'main/loss']),
                   trigger=report_interval)

    trainer.run()
