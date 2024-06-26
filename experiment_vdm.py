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
    outputs = self.state.apply_fn(
        variables={'params': params},
        **inputs,
        rngs=rngs,
        deterministic=not is_train,
    )

    rescale_to_bpd = 1./(np.prod(inputs["images"].shape[1:]) * np.log(2.))
    bpd_latent = jnp.mean(outputs.loss_klz) * rescale_to_bpd
    bpd_recon = jnp.mean(outputs.loss_recon) * rescale_to_bpd
    bpd_diff = jnp.mean(outputs.loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff
    scalar_dict = {
        "bpd": bpd,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "var0": outputs.var_0,
        "var": outputs.var_1,
    }
    img_dict = {"inputs": inputs["images"]}
    metrics = {"scalars": scalar_dict, "images": img_dict}

    return bpd, metrics

  def sample_fn(self, *, dummy_inputs, rng, params):
    rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    if self.model.config.sm_n_timesteps > 0:
      T = self.model.config.sm_n_timesteps
    else:
      T = 1000

    conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype='uint8')

    # sample z_0 from the diffusion model
    rng, sample_rng = jax.random.split(rng)
    z_init = jax.random.normal(sample_rng, dummy_inputs.shape)

    def body_fn(i, z_t):
      return self.state.apply_fn(
          variables={'params': params},
          i=i,
          T=T,
          z_t=z_t,
          conditioning=conditioning,
          rng=rng,
          method=self.model.sample,
      )

    z_0 = jax.lax.fori_loop(
        lower=0, upper=T, body_fun=body_fn, init_val=z_init)

    samples = self.state.apply_fn(
        variables={'params': params},
        z_0=z_0,
        method=self.model.generate_x,
    )

    return samples
