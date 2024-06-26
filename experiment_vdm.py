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

import numpy as np
import jax.numpy as jnp
from jax._src.random import PRNGKey
import jax
from typing import Any, Tuple

from vdm.experiment import Experiment
import vdm.model_vdm


class Experiment_VDM(Experiment):
  """训练和评估一个VDM模型."""

  def get_model_and_params(self, rng: PRNGKey):
      # 获取模型的配置以及参数
      # PRNGKey :该参数是一个伪随机数生成器键（PRNGKey），用于模型初始化。
    config = self.config
    config = vdm.model_vdm.VDMConfig(**config.model) #将配置转换为VDM模型所需的配置格式。
    model = vdm.model_vdm.VDM(config) # 从model_vdm 文件中创建一个vdm模型实例

    inputs = {"images": jnp.zeros((2, 32, 32, 3), "uint8")} #定义了输入的格式 作为一个字典 两个32x32像素的RGB图像
    inputs["conditioning"] = jnp.zeros((2,)) #向 inputs 字典中添加了另一个键 "conditioning"，其对应的值是一个形状为 (2,) 的一维张量
    '''
    inputs = {
    "images": jnp.zeros((2, 32, 32, 3), "uint8"),  # 图像数据
    "conditioning": jnp.zeros((2,), "uint8")        # 条件数据
}
    '''
    #   # rng1 -> train set  ; rng2 -> eval_set
    rng1, rng2 = jax.random.split(rng)
    #   使用分割后的随机数生成器键和输入数据初始化模型参数。
    params = model.init({"params": rng1, "sample": rng2}, **inputs)
    return model, params

  def loss_fn(self, params, inputs, rng, is_train) -> Tuple[float, Any]:
    """ 计算模型的损失函数"""

    rng, sample_rng = jax.random.split(rng)
    # 分割随机数生成器（RNG）键，是为了保持原始 RNG 键的连续性和可复现性，同时为不同的操作生成独立的随机数序列
    # 最终得到了三个独立的随机数生成器键，它们可以分别用于不同的随机操作，同时保持操作之间的独立性
    rngs = {"sample": sample_rng} # 字典
    if is_train: #判断当前是否是训练模式 如果是 会将随机数生成器再分出一个dropout用来防治过拟合
      rng, dropout_rng = jax.random.split(rng)
      rngs["dropout"] = dropout_rng # 字典

    # 使用对偶采样方法对时间步进行抽样
    #  将模型的参数（params）和输入数据（inputs）应用到模型的计算过程中，以生成模型的输出。
    # 在模型的前向传播过程中，时间步的采样采用了 antithetic sampling 技术
    '''
    # 定义了VDM的输出
     loss_recon: chex.Array  # [B] 重构损失
    loss_klz: chex.Array  # [B] 先验损失
    loss_diff: chex.Array  # [B] 扩散损失
    var_0: float # 代表 z0 和 z1 的噪声方差的参数
    var_1: float
    '''
    outputs = self.state.apply_fn(
        variables={'params': params},
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
    )

    # 计算了一个缩放因子 rescale_to_bpd用于将模型的损失转换为比特每像素的形式
    rescale_to_bpd = 1./(np.prod(inputs["images"].shape[1:]) * np.log(2.))
    # 计算不同类型的损失并转换为 bpd
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd #先验损失
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd #重构损失
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd #扩散损失
    bpd = bpd_recon + bpd_latent + bpd_diff # 总损失
    # 创建标量字典
    scalar_dict = {
        "bpd": bpd,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
    }
    # 创建图像字典
    img_dict = {"inputs": inputs["images"]}
    # 总的 bpd 损失和度量指标字典
    # 实现了将图像数据与其对应的损失指标相关联,看到每个图像的模型性能表现
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return bpd, metrics

  def sample_fn(self, *, dummy_inputs, rng, params):
    '''
    从模型中采样 干净的x
    Args:
        dummy_inputs: 一个占位符输入，可能用于提供输入数据的形状或类型信息
        rng:一个随机数生成器键，用于控制采样过程中的随机性
        params:模型参数，用于在采样过程中应用模型。

    Returns:生成的样本 samples。

    '''
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch')) # 保证不同批次的随机性

    if self.model.config.sm_n_timesteps > 0:
      T = self.model.config.sm_n_timesteps
    else:
      T = 1000

    # 初始化条件向量
    conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype='uint8')

    # sample z_0 from the diffusion model  从扩散模型中采样z_0
    rng, sample_rng = jax.random.split(rng)
    z_init = jax.random.normal(sample_rng, dummy_inputs.shape) #从标准正态分布中采样初始去噪的潜在变量z_t
    #  z_0 是从标准正态分布中随机采样的,虽然这里的z_0是随机采样得到的，但是在之后的扩散和去噪过程中，
    #  模型会通过缩小多个损失使得模型恢复的数据逼近干净的数据
    #  扩散模型的训练过程和逆扩散机制确保了生成的数据与原始数据 x 在统计上是相关的
    #  这是通过优化模型参数、最小化重构误差以及可能的条件生成来实现的。
    def body_fn(i, z_t):
      return self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T,
          z_t=z_t,
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample, # 这里method 用的是model_vdm 中的采样方法
      ) # 逆向 apply_fn 作为逆向去噪的中间件

    z_0 = jax.lax.fori_loop(
        lower=0, upper=T, body_fun=body_fn, init_val=z_init) #逆向去噪

    samples = self.state.apply_fn(
        variables={'params': params},
        z_0=z_0,
        method=self.model.generate_x, # 这里method 用的是model_vdm 中的采样方法
    ) #从z_0重构出x

    return samples #这最后返回的就是干净的x
