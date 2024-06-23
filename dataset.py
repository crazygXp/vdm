# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset loader and processor."""
from typing import Tuple

from clu import deterministic_data
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

# 用于指示 TensorFlow 自动调整数据加载和预处理过程中的资源分配，比如并行处理的数量或预取批次的大小。
# 提升数据输入性能，使得数据加载更加高效，尤其是在训练大型模型或在多GPU环境中，可以显著提高训练速度。
AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_dataset(config, data_rng):
    '''
    换句话说就是对原始的数据集根据不同的项目进行不同的处理初始化
    通过这个两个参数，config 中包含的配置信息以及初始的随机数生成器 data_rng 来生成和返回数据集的迭代器。
    这个函数根据 config 中指定的数据集类型和相关配置，创建了适合训练和评估的数据集。
    这个函数的设计允许灵活地根据配置信息定制数据集的创建过程，使得数据加载和预处理可以根据不同的需求进行调整。
    Args:
        config: 是一个配置对象，其中包含了数据集、批量大小、子步骤数量等配置信息。
        data_rng:是一个随机数生成器的种子，用于确保数据的随机性。

    Returns:新的数据集

    '''
    # 用于将一个整数（在这个例子中是 jax.process_index() 返回的进程索引）“折叠”进一个现有的随机数生成器（在这个例子中是 data_rng）。
    # 这样做的目的是为每个进程生成一个唯一的随机数生成器，以确保在并行或分布式计算环境中，每个进程都能独立地生成随机数，而不会相互干扰。
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    # rng1 -> train set  ; rng2 -> eval_set
    rng1, rng2 = jax.random.split(data_rng)
    if config.data.dataset == 'cifar10':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)

    elif config.data.dataset == 'cifar10_aug':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10_augment)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)

    elif config.data.dataset == 'cifar10_aug_with_channel':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10_augment_with_channel_flip)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)

    elif config.data.dataset == 'imagenet32':
      _, train_ds = create_train_dataset(
          'downsampled_imagenet/32x32',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10)

      _, eval_ds = create_eval_dataset(
          'downsampled_imagenet/32x32',
          config.training.batch_size_eval,
          'validation',
          rng2,
          _preprocess_cifar10)
    else:
      raise Exception("Unrecognized config.data.dataset")

    return iter(train_ds), iter(eval_ds)

def create_train_dataset(
        task: str,
        batch_size: int,
        # 在实际更新模型权重之前，会进行 多少次小的梯度累积和更新。
        substeps: int,
        data_rng,
        # 指定如何预处理数据集中的每个样本
        # 提供了一个方便的数据结构，使得用户可以同时访问数据集的详细描述和数据本
        preprocess_fn) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset]:
  """Create datasets for training."""
  #根据全局批量大小计算每个设备的批量大小
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  # 就是给了数据集一个对象，使用这个对象来调用各种方法和属性，以操作和访问数据集
  # 函数根据提供的 task 创建一个数据集构建器对象。这个对象包含了下载、准备和构建数据集所需的所有信息和方法。
  dataset_builder = tfds.builder(task)
  # 如果本地尚未存在指定的数据集，该方法会从源（如互联网）下载数据集。
  # 该方法还会准备数据集，这可能包括解压下载的文件、执行数据清洗或预处理步骤等。
  dataset_builder.download_and_prepare()

  # deterministic_data 可能提供了一种机制来确保数据加载过程是可重复的，这对于调试和复现实验结果非常重要。
  # 可以帮助确保在不同运行或实验中以相同的顺序和方式加载数据
  # 并且允许用户指定和访问数据的不同切割部分
  # 将builder对象中包含与训练集有关的数据分割出来并赋给train_spil
  train_split = deterministic_data.get_read_instruction_for_host(
      "train", dataset_builder.info.splits["train"].num_examples)
  #  根据本地设备数量、每个设备上的批数，设置批处理维度
  batch_dims = [jax.local_device_count(), substeps, per_device_batch_size]

  # 创建对应的训练数据集
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      num_epochs=None,
      shuffle=True,
      batch_dims=batch_dims,
      preprocess_fn=preprocess_fn,
      # 不是手动设置一个固定的预取批次数量，而是让 TensorFlow 动态地决定最优的预取量。
      # 预取是指有多少批次的数据被预先加载到内存中，以便它们可以立即被模型使用，而不需要等待数据加载完成。
      prefetch_size=tf.data.experimental.AUTOTUNE,
      rng=data_rng)

  return dataset_builder.info, train_ds


def create_eval_dataset(
        task: str,
        batch_size: int,
        subset: str,
        data_rng,
        preprocess_fn) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset]:
  #   具体同上
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)

  eval_split = deterministic_data.get_read_instruction_for_host(
      subset, dataset_builder.info.splits[subset].num_examples)
  # 根据本地设备数量、每个设备上的批数，设置批处理维度
  batch_dims = [jax.local_device_count(), per_device_batch_size]

  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      num_epochs=None,
      shuffle=True,
      batch_dims=batch_dims,
      preprocess_fn=preprocess_fn,
      prefetch_size=tf.data.experimental.AUTOTUNE,
      rng=data_rng)

  return dataset_builder.info, eval_ds


# 下面是对CIFAR-10的三种数据增强方法，用作预处理步骤函数
def _preprocess_cifar10(features):
  #   这是最基本的预处理函数，没有应用数据增强，只是将输入特征中的图像提取出来。
  """Helper to extract images from dict."""
  conditioning = tf.zeros((), dtype='uint8')
  return {"images": features["image"], "conditioning": conditioning}


def _preprocess_cifar10_augment(features):
  #   这个函数应用了两种随机的数据增强技术：
  # 随机左右翻转 (tf.image.flip_left_right)：以一定的概率对图像进行水平翻转。
  # 随机90度旋转 (tf.image.rot90)：以一定的概率对图像进行90度的旋转（可能是0次、1次、2次或3次）。
  img = features['image']
  img = tf.cast(img, 'float32')

  # random left/right flip
  _img = tf.image.flip_left_right(img)
  aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(aug, _img, img)

  # random 90 degree rotations
  u = tf.random.uniform(shape=[])
  k = tf.cast(tf.math.ceil(3. * u), tf.int32)
  _img = tf.image.rot90(img, k=k)
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  if False:
    _img = tf.transpose(img, [2, 0, 1])
    _img = tf.random.shuffle(_img)
    _img = tf.transpose(_img, [1, 2, 0])
    _aug = tf.random.uniform(shape=[]) > 0.5
    img = tf.where(_aug, _img, img)
    aug = aug | _aug
  #     aug 变量跟踪是否应用了增强（0表示未增强，1表示增强），最后将这个信息作为 conditioning 张量返回。
  aug = tf.cast(aug, 'uint8')

  return {'images': img, 'conditioning': aug}


def _preprocess_cifar10_augment_with_channel_flip(features):
  #   这个函数在 _preprocess_cifar10_augment 的基础上增加了随机颜色通道翻转：
  # 随机颜色通道洗牌：首先将图像从 HWC 格式（高度、宽度、通道）转换为 CHW 格式（通道、高度、宽度），然后随机打乱通道的顺序，最后再转换回 HWC 格式。
  img = features['image']
  img = tf.cast(img, 'float32')

  # random left/right flip
  _img = tf.image.flip_left_right(img)
  aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(aug, _img, img)

  # random 90 degree rotations
  u = tf.random.uniform(shape=[])
  k = tf.cast(tf.math.ceil(3. * u), tf.int32)
  _img = tf.image.rot90(img, k=k)
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  # random color channel flips
  _img = tf.transpose(img, [2, 0, 1])
  _img = tf.random.shuffle(_img)
  _img = tf.transpose(_img, [1, 2, 0])
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  # 与第二个函数一样，aug 变量跟踪是否对图像应用了增强，并作为 conditioning 张量返回。
  aug = tf.cast(aug, 'uint8')

  return {'images': img, 'conditioning': aug}
