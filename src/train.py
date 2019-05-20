import os
import cv2
import numpy as np
from PIL import Image
import six
from pathlib import Path

import chainer
from chainer.dataset import dataset_mixin
from chainer import training
from chainer import iterators
from chainer.training import extensions
from .model import ReflectionRemovalNet


def _create_composed_image_as_array(transmission_img, reflection_img, alpha=0.7):
    transmission_img, reflection_img = augument_random_crop(
        transmission_img, reflection_img, crop_size=size)

    G = cv2.cvtColor(cv2.GaussianBlur(reflection_img, (11, 11), 0), cv2.COLOR_BGR2RGB)

    image_blur = (transmission_img * alpha).astype(np.uint8) + ((1 - alpha) * G).astype(np.uint8)
    return image_blur


def augument_random_crop(transmission_img, right, crop_size=(768, 256)):
    transmission_img = np.array(transmission_img, dtype=np.float32)
    right = np.array(right, dtype=np.float32)
    h, w, _ = transmission_img.shape

    # randomly select crop start possition (top and transmission_img)
    top = np.random.randint(0, h - crop_size[1])
    transmission_img_position = np.random.randint(0, w - crop_size[0])

    # calc top and transmission_img bound
    bottom = top + crop_size[1]
    right_position = transmission_img_position + crop_size[0]

    # crop image
    transmission_img = transmission_img[top:bottom, transmission_img_position:right_position, :]
    right = right[top:bottom, transmission_img_position:right_position, :]
    return transmission_img, right


def prepare_dataset_path(folder_path):
    current_dir = Path.cwd()
    transmission_img_dir = current_dir / folder_path / 'transmission_img'
    reflection_img_dir = current_dir / folder_path / 'reflection_img'
    transmission_img_paths = transmission_img_dir.glob('**/*.png')
    reflection_img_paths = reflection_img_dir.glob('**/*.png')
    return transmission_img_paths, reflection_img_paths


class ImagePairDataset(dataset_mixin.DatasetMixin):

    def __init__(self, paths, root='.', augmentation=True):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip().split(' ') for path in paths_file]
        self._paths = paths
        self._root = root
        self.augmentation = augmentation

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path_left, path_right = self._paths[i]
        path_left = os.path.join(self._root, path_left).replace('.jpg', '.png')
        path_right = os.path.join(self._root, path_right).replace('.jpg', '.png')
        transmission_img = cv2.imread(path_left)
        reflection_img = cv2.imread(path_right)
        image_blur = _create_composed_image_as_array(transmission_img,
                                                     reflection_img,
                                                     alpha=0.7)
        if self.augmentation:
            pass
        image_blur /= 255.
        transmission_img /= 255.
        return image_blur, transmission_img


if __name__ == '__main__':
    iteration = 200000
    out_dir = './result/'
    batchsize = 16
    gpu = 0
    size = (224, 224)
    left_shift = [16, 16]

    snapshot_interval = (2000, 'iteration')
    report_interval = (100, 'iteration')
    rate_change_trigger = (20000, 'iteration')

    model = ReflectionRemovalNet(lambda_variable=0.001)
    optimizer = chainer.optimizers.Adam(alpha=0.002)
    optimizer.setup(model)

    train_paths = prepare_dataset_path('dataset')
    train_dataset = ImagePairDataset(train_paths, '')
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
