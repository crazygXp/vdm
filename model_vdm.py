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

from typing import Callable, Optional, Iterable

import chex
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np


@flax.struct.dataclass
class VDMConfig:
  """VDM configurations. 对模型的具体配置 """
  vocab_size: int
  sample_softmax: bool
  antithetic_time_sampling: bool
  with_fourier_features: bool
  with_attention: bool

  # 噪声调度的配置
  gamma_type: str
  gamma_min: float
  gamma_max: float

  # 评价模型的配置
  sm_n_timesteps: int # 时间步长的数量
  sm_n_embd: int ## 嵌入维度
  sm_n_layer: int #下采样层的数量
  sm_pdrop: float # dropout的强度
  sm_kernel_init: Callable = jax.nn.initializers.normal(0.02) # 核 初始化


######### Latent VDM model #########

@flax.struct.dataclass
class VDMOutput:
  # 定义了VDM的输出
  loss_recon: chex.Array  # [B] 重构损失
  loss_klz: chex.Array  # [B] 先验损失
  loss_diff: chex.Array  # [B] 扩散损失
  var_0: float # 代表 z0 和 z1 的噪声方差的参数
  var_1: float


class VDM(nn.Module):
  '''根据下面的所有方法 计算出VDM的三个损失

  '''
  config: VDMConfig

  def setup(self):
    self.encdec = EncDec(self.config) #初始化一个编码解码模块
    self.score_model = ScoreUNet(self.config) #U-Net 结构的分数模型，用于预测数据的噪声成分
    # 控制扩散过程中噪声水平的变化
    if self.config.gamma_type == 'learnable_nnet':  #一个可学习的网络 就是文中提到的
      self.gamma = NoiseSchedule_NNet(self.config)
    elif self.config.gamma_type == 'fixed':#一个固定的线性计划
      # 提供了一种预定的噪声衰减或增长模式，通常是线性的，意味着噪声水平随时间线性增加或减少。
      self.gamma = NoiseSchedule_FixedLinear(self.config)
    elif self.config.gamma_type == 'learnable_scalar':#一个可学习的标量值
      # 虽然噪声水平由一个单一的可学习参数控制，但这个标量值可以在整个训练过程中调整，以优化模型的整体性能。
      self.gamma = NoiseSchedule_Scalar(self.config)
    else:
      raise Exception("Unknown self.var_model")

  def __call__(self, images, conditioning, deterministic: bool = True):
    '''根据输入图像计算模型的损失函数

    Args:
      images: 图像数据 一般是一个批量
      conditioning: 条件信息，这可以是额外的输入数据，用于条件生成任务，比如文本描述或其他条件变量，以指导模型生成过程。
      deterministic: 当设置为 True 时，模型在前向传播过程中不包含随机性，使得结果可复现。
      如果设置为 False，则模型可能会在每次调用时引入随机性，这在某些训练策略中可能有用。

    Returns:
       loss_recon: chex.Array  # [B] 重构损失
       loss_klz: chex.Array  # [B] 先验损失
       loss_diff: chex.Array  # [B] 扩散损失
        var_0: float # 激活后的两个噪声
        var_1: float

    '''
    g_0, g_1 = self.gamma(0.), self.gamma(1.)
    #在时间步 0和1 处的噪声水平
    # 这个gamma在整个代码中反映的是噪声水平随时间的变化，只不过根据不同的噪声计划采用的值不一样，
    # 当采用的是Scalar时用的就是一个标量值，而对于NNet这个给方法采用的就是信噪比
    # 在 NoiseSchedule_Scalar 中，它是直接控制噪声水平的标量值；而在 NoiseSchedule_NNet 中，它可能是信噪比的对数，用于更复杂的噪声控制策略。
    var_0, var_1 = nn.sigmoid(g_0), nn.sigmoid(g_1)
    x = images
    # 批量的个数 也就是一批图像有多少个
    n_batch = images.shape[0]

    # encode 编码器 正向加噪过程可以看作是一种数据到潜在变量的转换过程，
    # 这个过程中的数据到潜在变量的映射，实际上承担了编码器的角色 f在这里对应的就是编码后的数据
    f = self.encdec.encode(x)

    # 1. 重构损失
    # add noise and reconstruct
    # eps0是从标准正态分布中采样的噪声，对应的就是从干净数据到增加了第一次的噪声从而到达了z0
    eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
    # z_0 是通过在原始数据 f 中添加噪声 eps_0 来生成的，这里的 var_0 是一个变量，可能表示原始数据中噪声的比例或方差。
    z_0 = jnp.sqrt(1. - var_0) * f + jnp.sqrt(var_0) * eps_0
    # 对z_0进行了重缩放
    z_0_rescaled = f + jnp.exp(0.5 * g_0) * eps_0  # = z_0/sqrt(1-var)
    # logprob方法返回给定输入 x 和潜在变量 z_0_rescaled 下的对数概率
    # 文中也有提到重构近似的是正向的第一步加噪，所以根据第一步噪声以及z0计算对数概率
    loss_recon = - self.encdec.logprob(x, z_0_rescaled, g_0)

    # 2. 先验损失
    # KL z1 with N(0,1) prior 文中对于先验损失的描述就是计算两个分布之间的KL散度
    # 也就是计算模拟出的z1和最后的真实正态噪声分布之间的KL
    mean1_sqr = (1. - var_1) * jnp.square(f)
    # 编码器输出的潜在变量的均值的平方
    loss_klz = 0.5 * jnp.sum(mean1_sqr + var_1 - jnp.log(var_1) - 1., axis=(1, 2, 3))
    # 这里的var_1 编码器输出的潜在变量的标准差的平方,也就是潜在变量的方差
    # 计算两个正态分布之间的KL散度公式可以化简为 论文笔记 其中对应的就是两个分布的参数
    # 计算的是先验损失计算模型生成的潜在变量 z1 的分布与先验分布（标准正态分布 ）之间的 散度
    #  同时处理的是图像数据所以要求三个轴的和,在计算先验损失（prior loss）时，这种求和操作是有意义的
    #  ，因为它允许我们得到整个图像的损失的总和，而不是只考虑单个像素或通道的损失。

    # 3. 扩散损失 文中对与扩散损失考虑了两种 1.离散时间 2.连续时间
    # 采样时间步长 生成连续的、循环的时间步
    # 这里就是指定sample作为键生成一个随机数生成器，为了在不同的个操作中保持随机数的一致可复现
    rng1 = self.make_rng("sample")
    if self.config.antithetic_time_sampling: #判断是否启用了特定的对偶时间采样方法
      t0 = jax.random.uniform(rng1) #随机采样起点
      t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)
      #从 t0 开始，生成一个等差数列，其中每个后续时间步 t 都是前一个时间步加上一个固定增量。增量是 1/n_batch，这里 n_batch 是批次大小。
    #   最后一个参数 1 作为取余的数字，因为要保证t0 + jnp.arange(0., 1., step=1. / n_batch)仍然在0-1之间
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,)) #如果没有特定时间采样就是随机采样

    # 时间步长的离散化
    T = self.config.sm_n_timesteps # 从配置中获取离散时间步的数量
    if T > 0:
      t = jnp.ceil(t * T) / T #将上一步得到的连续时间步 离散化

    # 采样潜在变量  z_t
    #gamma = NoiseSchedule_NNet(self.config)
    g_t = self.gamma(t) #在时间步 t 处的对数信噪比
    var_t = nn.sigmoid(g_t)[:, None, None, None] # 噪声方差
    eps = jax.random.normal(self.make_rng("sample"), shape=f.shape) # 从标准正态分布中采样的随机噪声。
    z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps #计算最终的状态
    #计算预测噪声——噪声预测模型
    eps_hat = self.score_model(z_t, g_t, conditioning, deterministic)
    # 计算预测噪声和真实噪声之间的均方误差 也就是文中两个公式中的ε的差
    loss_diff_mse = jnp.sum(jnp.square(eps - eps_hat), axis=[1, 2, 3])

    if T == 0:
      # loss for infinite depth T, i.e. 连续时间 在连续时间公式中需要的是 loss_diff_mse以及信噪比的导数即g_t_grad
      #jax.jvp（func，primals，tangents） 计算函数相对于其参数的雅可比矩阵（参数的导数矩阵）与一个向量的乘积
      # 所以jax.jvp 计算的是 self.gamma(t) 的值以及 self.gamma 关于 t 的梯度。
      # 由于 tangents 是一个全1数组，雅可比向量积简化为 self.gamma 的梯度，这是因为全1数组与任何矩阵的乘积都等于该矩阵的列向量和。
      # 使用全1数组是因为它可以简化计算过程，直接得到梯度，而不需要显式地构建雅可比矩阵。
      _, g_t_grad = jax.jvp(self.gamma, (t,), (jnp.ones_like(t),))
      # 对应文中的公式17
      loss_diff = .5 * g_t_grad * loss_diff_mse
    else:
      # loss for finite depth T, i.e. 离散时间
      s = t - (1./T)
      g_s = self.gamma(s)
      # jnp.expm1是一个JAX库中的特殊函数，用于计算x.exp-1所以这里表示的就是exp(g_t - g_s) - 1
      # 对应文中的公式 14
      loss_diff = .5 * T * jnp.expm1(g_t - g_s) * loss_diff_mse

    # 结束扩散损失的计算

    return VDMOutput(
        loss_recon=loss_recon,
        loss_klz=loss_klz,
        loss_diff=loss_diff,
        var_0=var_0,
        var_1=var_1,
    )

  def sample(self, i, T, z_t, conditioning, rng):
    '''
    实现了逆向去噪的过程
    Args:
      i: 当前步
      T: 总步数
      z_t: 最终潜在变量
      conditioning: 条件
      rng: 随机数生成器

    Returns: 前一步的潜在变量

    '''
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)

    t = (T - i) / T
    s = (T - i - 1) / T
    # 对应时间步的信噪比函数的值
    g_s, g_t = self.gamma(s), self.gamma(t)
    # 噪声预测模型 计算出上一步到这一步的噪声
    eps_hat = self.score_model(
        z_t,
        g_t * jnp.ones((z_t.shape[0],), g_t.dtype),
        conditioning,
        deterministic=True)
    a = nn.sigmoid(-g_s) #经过s和t时间步的信噪比的概率形式
    b = nn.sigmoid(-g_t)
    c = - jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t)) #时间步 t 对应的噪声标准差
    # 上一个状态
    z_s = jnp.sqrt(nn.sigmoid(-g_s) / nn.sigmoid(-g_t)) * (z_t - sigma_t * c * eps_hat) + \
        jnp.sqrt((1. - a) * c) * eps

    return z_s

  # 重构，实现从z_0回到x
  def generate_x(self, z_0):
    g_0 = self.gamma(0.)

    var_0 = nn.sigmoid(g_0)
    z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)
    # 使用编码器-解码器模型的解码部分（self.encdec.decode）
    # 来从调整尺度后的潜在变量 z_0_rescaled 和信噪比 g_0 生成数据 x 的 对数概率
    # Logits 是未经过 softmax 函数处理的原始输出
    logits = self.encdec.decode(z_0_rescaled, g_0)

    # get output samples
    if self.config.sample_softmax:
      out_rng = self.make_rng("sample")
      samples = jax.random.categorical(out_rng, logits)
    else:
      # 选择 logits 中最大值对应的类别，这相当于进行 argmax 操作来获取最可能的类别标签。
      samples = jnp.argmax(logits, axis=-1)
    # 重构后的样本
    return samples

######### Encoder and decoder #########
# 用于具体逆向去噪和加噪过程中的方法

class EncDec(nn.Module):
  """Encoder and decoder. """
  config: VDMConfig

  def __call__(self, x, g_0):
    # For initialization purposes
    h = self.encode(x)
    # 返回的是一个对数概率
    return self.decode(h, g_0)

  def encode(self, x):
    # 这段代码将x从离散值（0, 1, ...）转换到区间 (-1, 1)。
    # 这里的四舍五入是为了确保输入是离散的
    # （尽管通常情况下，x 是一个离散变量，比如 uint8 类型）。

    x = x.round()
    return 2 * ((x+.5) / self.config.vocab_size) - 1

  def decode(self, z, g_0):
    ''' 计算在给定不同潜在变量 z 的情况下，每个干净数据点 x的对数概率分布

    通过输入的当前参数代表的当前信噪比以及对应的潜在变量，具体来说
    对于干净数据的分布就是用当前潜在变量和干净数据可能取值之间的差异描述的，
    只不过通过信噪比动态的调控这个分布的样子，是宽还是窄，对应干净数据的信心
    注意:decode 方法：它用于生成一个概率分布，这个分布表示在给定潜在变量 z 的情况下，
    所有可能的干净数据点 x 各自出现的概率。这个过程就像是画出了一个概率的“地图”，
    上面标注了每个 x 值的概率，但并没有特定指向实际观测到的数据点。
    Args:
      z: 潜在变量
      g_0: z_0的信噪比

    Returns: 对数概率分布

    '''
    config = self.config

    #如果 ( x ) 的各维度之间没有依赖关系，那么对数几率是精确的
    # 在图像处理任务中，config.vocab_size 通常指的是词汇表的大小，它表示可以表示的不同值或类别的数量。
    # 在像素级别的图像数据中，每个像素值通常是从0到255的整数（对于8位图像）
    # 因此，config.vocab_size 将设置为256，以覆盖所有可能的像素强度值。
    x_vals = jnp.arange(0, config.vocab_size)[:, None]
    # x_vals 变量原本包含了从0到 config.vocab_size - 1 的整数序列，
    # 这个序列表示了单个通道（比如红色通道）可能的所有像素值
    # 为了处理三通道的图像数据，我们需要将这个序列扩展到三个维度，以代表每个像素点的三个颜色通道。
    # x_vals 数组的形状 [256, 3] 表示了三个通道上每个可能的离散像素值的索引 用于后续计算
    x_vals = jnp.repeat(x_vals, 3, 1)
    # 映射到合适的范围,最后得到的这个数组用于计算每个潜在变量 z 对应于每个可能的 x 值的概率
    x_vals = self.encode(x_vals).transpose([1, 0])[None, None, None, :, :]
    # 通过解码步骤，模型计算出在潜在空间中每个 z 对应于 x_vals 中每个值的对数概率。
    # 这些对数概率可以被转换成概率，从而给出在给定 z 的条件下，每个可能的 x 值出现的可能性。
    inv_stdev = jnp.exp(-0.5 * g_0[..., None])
    # 每个潜在变量 z 对应的高斯分布的逆标准差,用于调整数据点的不确定性
    #  z[..., None] - x_vals 的差异实际上是在评估噪声数据与干净数据可能值范围之间的差异。表示每个噪声数据点与干净数据可能值之间偏差的数组。
    # 将这个数组与逆标准差相乘,为了根据当前步骤的信噪比调整这些偏差的大小。
    logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)
    '''
   信噪比较低（噪声较大）：

这时，inv_stdev（逆标准差）较大，表示潜在变量 z 的不确定性较高。
差异 (z[..., None] - x_vals) 与较大的 inv_stdev 相乘后，结果被放大，但由于是对数概率的计算，这导致 logits 变小。
小的 logits 值意味着给定 z 条件下，每个 x_vals 的概率较低，概率分布变得平坦，表示我们对任何特定干净数据点的预测不够信任。
信噪比较高（噪声较小）：

这时，inv_stdev 较小，表示潜在变量 z 的不确定性较低。
差异 (z[..., None] - x_vals) 与较小的 inv_stdev 相乘后，结果被缩小，导致 logits 变大。
大的 logits 值意味着给定 z 条件下，某些 x_vals 的概率较高，概率分布出现尖锐的峰值，表示我们对干净数据点的预测更有信心。
通过这种方式，模型能够根据潜在变量 z 中的噪声水平，动态调整对干净数据点 x 的概率估计。
这种调整是通过信噪比来控制的，信噪比作为模型对数据点预测信任度的一个指标。
当信噪比高时，模型对预测的干净数据点非常信任，表现为概率分布的高峰值；
当信噪比低时，模型对预测的干净数据点信任度降低，表现为概率分布的平坦化。
这种动态调整是通过对数概率计算中的逆标准差 inv_stdev 的调整实现的。
    '''
    logprobs = jax.nn.log_softmax(logits)
    return logprobs

  def logprob(self, x, z, g_0):
    '''计算出正确干净数据的分布

    换句话说，logprob 方法是用来评估模型对于实际观测数据的拟合度，即模型认为观测到的数据点有多“可能”。
    logprob 方法：它使用的是 decode 方法生成的分布，但目的不同。
    logprob 方法通过 one-hot 编码的方式，从 decode 方法提供的概率“地图”中找出实际观测到的数据点 x 所对应的概率，
    然后计算这个概率的对数。这个过程就像是在“地图”上找到了与观测数据匹配的那个点，并计算了这个点的高度（概率的对数）。
    '''
    x = x.round().astype('int32')
    # 创建一个 one-hot 编码的张量 x_onehot。self.config.vocab_size 指定了词汇表的大小，即 one-hot 编码的深度。
    x_onehot = jax.nn.one_hot(x, self.config.vocab_size)
    logprobs = self.decode(z, g_0)
    # 与 one-hot 编码的 x 相乘是为了确保只有正确的像素值的概率被用于计算最终的损失或对数概率
    # 这里是4个轴，因为是图像数据，对应的就是（B,H,W,C）批数，高，宽，通道数（三通）
    # 这里的四个维度求和实际上就是在计算期望（expected value），即对所有可能的干净数据点 x 的概率进行加权求和。
    logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3, 4))
    return logprob


######### Score model #########
# 噪声预测模型

class ScoreUNet(nn.Module):
  config: VDMConfig
  # 对于这个噪声预测模型来说，它是通过当前的时间步嵌入、条件向量、原始特征、傅里叶特征共同特征融合以后再经过卷积完成噪声的预测
  @nn.compact
  def __call__(self, z, g_t, conditioning, deterministic=True):
    '''

    Args:
      z:
      g_t:
      conditioning: 代表了用于条件生成的额外信息。这些信息被用来影响和指导模型的生成过程，使其能够生成满足特定条件的数据
      deterministic:
    Returns:
      预测的噪声
    '''
    config = self.config

    # 基于“g_t”和“conditioning”计算条件向量
    n_embd = self.config.sm_n_embd # 嵌入维度

    lb = config.gamma_min #信噪比的最小值
    ub = config.gamma_max #信噪比的最大值
    t = (g_t - lb) / (ub - lb)  # ---> [0,1] 将信噪比g_t映射到0-1之间
    # t 成为了一个归一化的信噪比指标，它可以用来表示数据在整个扩散过程中的“位置”,t 是一个在 [0, 1] 范围内的值，可以作为扩散过程的相对时间指示器

    # 确保变量 t 满足一个标量、零维数组，或者一维数组
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    if jnp.isscalar(t):
      t = jnp.ones((z.shape[0],), z.dtype) * t #确保 t 被扩展成与 z 的第一个维度相同的大小。每个批次中的每个图像都会有一个对应的时间步 t 值。
    elif len(t.shape) == 0:
      # z.shape[0] 通常表示批次大小（batch size），即一次处理的图像数量
      t = jnp.tile(t[None], z.shape[0])
    #     t[None] 将 t 转换为一个一维数组，其中只有一个元素。
    #     jnp.tile 函数将这个只有一个元素的数组复制 z.shape[0] 次，创建一个新的一维数组，其长度与 z 的第一个维度相同

    temb = get_timestep_embedding(t, n_embd) #将时间步信息编入时间嵌入向量 ,可以捕获时间序列中的动态特性
    '''
    拼接时间步信息与条件向量,
    这里的conditioning  通常指的就是提供给模型的额外指导信息，这些信息可以被视为一种特征，用于影响和引导生成过程
    conditioning[:, None] 将这个数组变为一个二维数组，其中第二维的大小为1。
    然后，使用 jnp.concatenate 沿着第一个维度（axis=1）将时间嵌入 temb 和 conditioning 拼接起来，
    形成一个更丰富的条件向量 cond。
    '''
    cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
    # 拼接后的条件向量 cond 随后通过一个或多个全连接层和激活函数进行处理，
    # 以便进一步提取特征并生成最终的条件表示，这个过程在代码的后续部分完成。
    # 这里一共有四层,一个dense全连接一个swish激活 再跟两个一摸一样
    # 对于网络的输出是将cond映射到具有n_embd * 4个输出特征的层 并通过一个激活层 定义为dense 0
    # 这就是深度网络或多层感知机（MLP），它可以进一步提取和组合特征，增加模型的表征能力。
    # 通过这个MLP就可以捕捉到数据的多尺度特征和条件特定的属性，为后续的生成任务提供了必要的上下文信息
    '''
    我是不是也可以替换这个目标，如果我处理的是TSP问题，将这里的条件设为最小化旅行长度，
    通过与时间步进行融合是不是可以认为，指导生成的目标就是在每个时间步最小化旅行长度
    从理论上讲，答案是肯定的。在变分扩散模型中，conditioning 通常用于提供额外的上下文或条件信息，以指导生成过程

    '''
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # 将傅立叶特征与输入进行连接
    if config.with_fourier_features:
      z_f = Base2FourierFeatures(start=6, stop=8, step=1)(z)
      # 将傅里叶特征和正常特征进行拼接
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input 输入的线性投影
    #  当原始特征和傅里叶特征被拼接后，卷积层可以帮助整合这两种特征，使网络能够学习到它们之间的相互关系和组合方式。
    # 提取新特征: 卷积层通过其卷积核可以提取输入特征图中的局部特征，生成新的特征表示，这些新特征可能捕捉到输入数据中的更高层次的模式。
    h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
    hs = [h] #初始化特征列表 将当前的特征图 h 作为第一个元素存储起来

    '''
    下采样（downsampling）、中间处理（middle）、上采样（upsampling）
    这些组件有助于模型学习复杂的特征并保持信息流
    '''
    # Downsampling 下采样
    for i_block in range(self.config.sm_n_layer):
      # 按照config要求创建残差块  out_ch作为输出通道数的参数设置为与输入具有相同的通道数
      # 并且根据当前循环的索引 动态的生成残差快的名字
      # 残差网络的核心思想是让网络学习输入和输出之间的残差（即差异）。其中cond其包含着对于时间步的嵌入
      # 通过使用最后一个特征图，残差块可以更容易地学习到在当前层级上需要进一步处理的特征变化
      block = ResnetBlock(config, out_ch=n_embd, name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if config.with_attention:
        # 创建注意力块 同上
        h = AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h) #   将每次循环处理后的特征图 h 添加到列表 hs 的末尾

    # Middle 中间处理
    # 从下采样过程中接收最后一个特征图 hs[-1]。
    # 通过两个残差块 ResnetBlock 和一个注意力块 AttnBlock 进一步处理特征图
    h = hs[-1]
    h = ResnetBlock(config, name='mid.block_1')(h, cond, deterministic)[0]
    h = AttnBlock(num_heads=1, name='mid.attn_1')(h)
    h = ResnetBlock(config, name='mid.block_2')(h, cond, deterministic)[0]

    # Upsampling 上采样
    for i_block in range(self.config.sm_n_layer + 1):
      b = ResnetBlock(config, out_ch=n_embd, name=f'up.block_{i_block}')
      # jnp.concatenate([h, hs.pop()], axis=-1)通过将下采样得到的hs中的每个元素与h进行拼接，完成多尺度的特征融合
      # 将拼接后的特征图作为输入，连同条件 cond 和确定性标志 deterministic 一起传递给残差块 b。
      # b(...) 调用执行残差块的前向传播，其输出是一个包含多个元素的列表，[0] 表示我们只取列表中的第一个元素作为新的 h。
      h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
      if config.with_attention:
        h = AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)

    assert not hs #确保hs被清空

    # Predict noise 噪声预测
    normalize = nn.normalization.GroupNorm() #一个组归一化层 对每个特征通道进行归一化，而不是整个批次
    h = nn.swish(normalize(h)) #对特征先进行组归一化 然后是激活
    eps_pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h) #利用卷积层预测噪声

    # Base measure 将预测的噪声 eps_pred 与原始数据 z 相加 实现了残差连接
    # 预测的噪声可以被视为数据的残差，将其与原始数据相加是为了恢复模型在添加噪声时所失去的信息。
    eps_pred += z

    return eps_pred #预测的噪声


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=jnp.float32):
  """构建正弦嵌入向量（来自 Fairseq）。 也就是处理时间步嵌入

  这与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 第3.5节中的描述略有不同。
  Args:
    timesteps: jnp.ndarray: 在这些时间步生成嵌入向量
    embedding_dim: int:要生成的嵌入向量的维度
    dtype: 生成的嵌入向量的数据类型

  Returns:
   形状为 `(len(timesteps), embedding_dim)` 的嵌入向量
  """
  assert len(timesteps.shape) == 1 #确保 timesteps 是一维数组。

  # 这一步的处理因为包含时间步的嵌入的cond在上面也要进行傅里叶特征的提取
  # 所以要对这里进行一个重缩放
  timesteps *= 1000. #将时间步缩放1000倍，这可能是为了调整后续正弦和余弦函数的输入范围。
  half_dim = embedding_dim // 2  #后续一半用作正弦 一半用作余弦
  emb = np.log(10000) / (half_dim - 1) #用于生成正弦和余弦波形的对数尺度。
  '''生成正弦和余弦波形
  在许多序列模型中，特别是基于Transformer的模型，时间步嵌入通常设计为具有频率衰减的特性。
  这是因为我们希望模型能够捕捉到不同时间步之间的相对位置关系，而较高的频率（即较小的时间间隔）
  通常对应于这些细微的位置差异。通过让频率随着时间步的增加而递减，模型可以更敏感地响应序列中较短距离的关系。
  '''
  emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb) #生成的是一个递减的序列,对应于不同频率的振幅系数
  emb = timesteps.astype(dtype)[:, None] * emb[None, :]  # 每个时间步 timesteps 与递减的振幅系数序列相乘，从而为每个时间步生成一组特定的频率系数
  #合并正弦和余弦-生成实际的正余弦波形
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1) #shape (timesteps, embedding_dim)

  # 因为一半用作正弦 一半用作余弦，但是如果dim是奇数，就用0填充
  # ((0, 0, 0), (0, 1, 0)) 表示在第一个维度（时间步）的开始和结束不填充任何元素，
  # 在第二个维度（频率系数）的末尾填充一个元素。
  # 这样做的结果是，如果 embedding_dim 是奇数，就在 emb 的最后一列添加一个零。
  if embedding_dim % 2 == 1:  # 零填充
    '''
    jax.lax.pad(array, padding_value, padding_config)
    padding_config[[low_pad, high_pad],[low_pad, high_pad]]
    low_pad 表示在第一维度（行）的开始（顶部）添加的填充元素的数量。
    high_pad 表示在第一维度（行）的末尾（底部）添加的填充元素的数量。
    low_pad 表示在第二维度（列）的开始（左侧）添加的填充元素的数量。
    high_pad 表示在第二维度（列）的末尾（右侧）添加的填充元素的数量。
    '''
    emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


######### 三种 噪声计划 用来控制扩散过程中的噪声水平/信噪比 #########

class NoiseSchedule_Scalar(nn.Module):
  config: VDMConfig
  #  虽然噪声水平由一个单一的可学习参数控制，但这个标量值可以在整个训练过程中调整，以优化模型的整体性能。
  def setup(self):
    init_bias = self.config.gamma_min #噪声的初始水平
    init_scale = self.config.gamma_max - init_bias #噪声的范围
    self.w = self.param('w', constant_init(init_scale), (1,)) #噪声水平变化的幅度。
    self.b = self.param('b', constant_init(init_bias), (1,)) #噪声水平的偏置

  @nn.compact
  def __call__(self, t):
    # 这是一个线性变换，表示噪声水平随时间步 t 的变化。从 t=0 开始噪声水平随时间的线性增加
    return self.b + abs(self.w) * t #输出实际上是一个标量值，它代表了在时间步 t 时的噪声水平

'''
NoiseSchedule_FixedLinear 和 NoiseSchedule_Scalar 都提供了线性的噪声增长模型，
但 NoiseSchedule_Scalar 通过引入可学习的参数，为模型提供了额外的优化能力和适应性。
这种适应性使得 NoiseSchedule_Scalar 在训练过程中可以根据数据的特征进行更精细的调整
'''
class NoiseSchedule_FixedLinear(nn.Module):
  config: VDMConfig
  # 提供了一种预定的噪声衰减或增长模式，通常是线性的，意味着噪声水平随时间线性增加或减少。
  # 噪声水平在时间 t 从 gamma_min 线性增加到 gamma_max
  @nn.compact
  def __call__(self, t):
    config = self.config
    return config.gamma_min + (config.gamma_max-config.gamma_min) * t


class NoiseSchedule_NNet(nn.Module):
  # 一个可学习的网络 就是文中提到的 三层权重限制为正的单调网络
  config: VDMConfig
  n_features: int = 1024 #第二层上有1024个输出
  nonlinear: bool = True # 决定网络是否包含非线性变换

  def setup(self):
    config = self.config

    n_out = 1
    kernel_init = nn.initializers.normal()

    init_bias = self.config.gamma_min # 噪声的初始水平
    init_scale = self.config.gamma_max - init_bias #噪声的范围

    # DenseMonotone代表全连接层
    self.l1 = DenseMonotone(n_out,
                            kernel_init=constant_init(init_scale),
                            bias_init=constant_init(init_bias)) # 第一层 输出为1维 用于生成初始噪声水平
    if self.nonlinear:
      self.l2 = DenseMonotone(self.n_features, kernel_init=kernel_init) # 第二层 引入非线性 将数据映射到1024维
      self.l3 = DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)  # 第三层 引入非线性 重新映射回1 维

  @nn.compact
  def __call__(self, t, det_min_max=False):
    # 声明 输入t的类型支持标量、零维和一维输入
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    # 将标量或零维数组的t 分别通过乘以（1，1）数组和reshape变为一个二维数组
    '''
    将输入数据扩充成二维数组（通常是矩阵形式）是为了满足全连接层（Dense layers）的输入要求。
    全连接层期望输入具有固定的维度，这通常包括两个维度：一个批量大小维度（batch size）和一个特征维度。
    即使批量大小为1，这种二维表示也是必需的，因为它允许使用矩阵乘法来高效地实现层的计算。
    '''
    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((1, 1))
    else:
      t = jnp.reshape(t, (-1, 1))

    h = self.l1(t) #通过第一层 h.shape ==(-1,1)
    if self.nonlinear: #如果启用非线性变换
      _h = 2. * (t - .5)  # 将输入缩放至 [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h #非线性层的输出 _h 加到初步噪声水平估计 h 上，合并为最终的噪声计划输出
    #   这就实现了一个学习时间步 t 与噪声水平之间复杂关系的模型

    #确保输出是一维的，即使 h 原本是二维但只有一个元素。
    return jnp.squeeze(h, axis=-1)


def constant_init(value, dtype='float32'):
  '''创建具有恒定值的初始化操作 用于NN中的W和b的初始化

  Args:
    value: 用于初始化的值
    dtype: 输出数组的类型

  Returns:创建了一个所有元素都是 value 的数组。

  '''
  def _init(key, shape, dtype=dtype):
    return value * jnp.ones(shape, dtype)
  return _init


class DenseMonotone(nn.Dense):
  """严格单调递增的全连接层."""
  # 这个就像是噪声网络中的用来产生噪声的了
  @nn.compact
  def __call__(self, inputs):
    #  将输入转为数组 并且将数据类型完成转换
    inputs = jnp.asarray(inputs, self.dtype)
    #  初始化权重矩阵
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    #  保证所有到的权重非负 从而实现 噪声的单调
    kernel = abs(jnp.asarray(kernel, self.dtype))
    # 将输入数据 inputs 与权重矩阵 kernel 相乘，从而将噪声的权重加到了输入上。
    # precision: 这个参数设置操作的数值精度
    # 其中参数dimension_numbers((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))
    # 重点解释 用于指定哪一维度进行收缩（contracting）和批处理（batching）
    # 其中收缩 指在矩阵乘法（dot product）中执行的操作，它决定了输入张量和核张量在哪些维度上进行乘积和求和
    # 其次 批处理是指在指定的批次维度上并行执行计算，而不跨批次进行操作。 可以理解为控制是否要跨样本计算
    y = jax.lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


######### ResNet block #########
# 下面的方法都是用来处理特征 用作计算预测噪声
class ResnetBlock(nn.Module):
  """卷积残差块包含两个卷积层,完成特征提取从而在噪声预测模型中使用"""
  #   输入的是原始特征图，返回的是经过处理的特征图
  config: VDMConfig
  out_ch: Optional[int] = None # 一个可选的整数，表示输出通道数。如果未指定，则使用输入的通道数。

  @nn.compact
  def __call__(self, x, cond, deterministic: bool, enc=None):
    config = self.config

    nonlinearity = nn.swish #定义非线性激活函数
    normalize1 = nn.normalization.GroupNorm()
    normalize2 = nn.normalization.GroupNorm()

    if enc is not None: #可选的编码输入，如果提供，将与 x 在通道维度上进行拼接
      # 因为要在噪声预测模型中使用,要根据当前的时间步等特征输出对于当前的噪声,所以如果有额外的指定要与特征进行拼接
      x = jnp.concatenate([x, enc], axis=-1)

    B, _, _, C = x.shape  # pylint: disable=invalid-name Pylint 是一个用于检查 Python 代码错误,
    # 在这里是告诉 Pylint 在检查代码时忽略所有与命名规范不符合的错误或警告。这通常用于暂时性地关闭 Pylint 对命名约定的检查，
    # (batch_size, height, width, channels)
    # 在这里我们只要他的B 和C
    out_ch = C if self.out_ch is None else self.out_ch #设置输出通道数 out_ch

    # 具体处理 提取特征
    h = x
    h = nonlinearity(normalize1(h))
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1')(h)

    # 将条件向量与特征图融合
    # cond 条件向量通过一个全连接层映射，然后将映射的结果添加到特征图 h 上。这个过程确保了 h 中的特征包含了条件向量 cond 的信息
    if cond is not None:
      # 确保条件向量的第一个维度与输入一致,并且确保cond只有两个维度
      assert cond.shape[0] == B and len(cond.shape) == 2
      # 融合 通过一个全连接层实现 参数[:, None, None, :]是确保经过全连接层的
      # cond之后变成形状为四维张量可以和H进行叠加
      h += nn.Dense(
          features=out_ch, use_bias=False, kernel_init=nn.initializers.zeros,
          name='cond_proj')(cond)[:, None, None, :]

    h = nonlinearity(normalize2(h))
    # 也是一种正则化技术 通过随机丢弃一些网络输出从而防止网络过拟合
    h = nn.Dropout(rate=config.sm_pdrop)(h, deterministic=deterministic)
    # 又过一边卷积
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(h)

    if C != out_ch: #控制最后输出通道数
      x = nn.Dense(features=out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape #确保调整通道后的 x 和经过残差块处理的 h 在形状上是相同的
    x = x + h
    # 方法返回残差连接的结果 x，以及原始输入 x
    return x, x


class AttnBlock(nn.Module):
  """自注意力机制的残差模块"""

  num_heads: int #注意力头

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape  # pylint: disable=invalid-name,unused-variable
    assert C % self.num_heads == 0 #保证每个头可以关注相同数量的通道，没有剩余。

    normalize = nn.normalization.GroupNorm() #组归一化

    h = normalize(x)
    if self.num_heads == 1:
      q = nn.Dense(features=C, name='q')(h) #query
      k = nn.Dense(features=C, name='k')(h) #key
      v = nn.Dense(features=C, name='v')(h) #value
      h = dot_product_attention(
          q[:, :, :, None, :],
          k[:, :, :, None, :],
          v[:, :, :, None, :],
          axis=(1, 2))[:, :, :, 0, :] #计算点积注意力 [:, :, :, None, :]这个表示的是在原本的（B,H,W,C）中增加一个维度
      # 确保张量具有适当的形状，以便进行后续的计算或操作
      # 变为（B,H,W,1,,C）通过新增这个维度 可以保持原有的序列维度 并且可以实现在批次维度上的广播(就可以理解为给每个批次作为一个维度)
      h = nn.Dense(
          features=C, kernel_init=nn.initializers.zeros, name='proj_out')(h) #将注意力输出映射回原始通道数 C
    else:
      head_dim = C // self.num_heads #每个头处理部分通道
      # 指定了输出张量的形状应该是 (自我注意力头数, 每个头的通道数)。通过输入features的不同，分别计算出每个头的qkv
      # 这意味着每个头都会生成一个形状为 (head_dim,) 的向量，而所有头的输出将组合成一个更大的张量，其形状为 (num_heads, head_dim)。
      q = nn.DenseGeneral(features=(self.num_heads, head_dim), name='q')(h)
      k = nn.DenseGeneral(features=(self.num_heads, head_dim), name='k')(h)
      v = nn.DenseGeneral(features=(self.num_heads, head_dim), name='v')(h)
      assert q.shape == k.shape == v.shape == (
          B, H, W, self.num_heads, head_dim)
      h = dot_product_attention(q, k, v, axis=(1, 2))
      h = nn.DenseGeneral(
          features=C,
          axis=(-2, -1),
          kernel_init=nn.initializers.zeros,
          name='proj_out')(h)

    assert h.shape == x.shape
    return x + h


def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          # broadcast_dropout=True,
                          # dropout_rng=None,
                          # dropout_rate=0.,
                          # deterministic=False,
                          precision=None):
  """这个函数计算了基于注意力机制的点积注意力（Dot-Product Attention），用于处理给定的查询（query）、键（key）和数值（value）

这是应用注意力的核心函数，基于论文 https://arxiv.org/abs/1706.03762。
它根据查询和键计算注意力权重，并利用这些注意力权重组合数值。该函数支持多维输入。

  Args:
    query:查询向量，形状为 [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    key: 键向量，形状同 query，用于计算注意力。
    value: 值向量，形状为 [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels]，用于根据注意力权重组合。
    dtype:计算时的数据类型（默认为 float32）。
    bias: 注意力权重的偏置，用于包含自回归掩码、填充掩码或接近性偏置。
    axis:应用注意力的轴。
    broadcast_dropout:布尔值，是否在批量维度上使用广播丢弃。
    dropout_rng: JAX PRNGKey，用于丢弃操作
    dropout_rate: 丢弃率。
    deterministic: 布尔值，是否确定性丢弃。
    precision: 计算的数值精度，参见 jax.lax.Precision。

  Returns:
    形状为 [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels] 的输出，表示经过注意力权重计算后的结果
  """
  assert key.shape[:-1] == value.shape[:-1] #检查 key 和 value 张量在除了最后一个维度之外的所有维度上形状是否相
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1]) #确保 query 和 key 在第一个维度（通常是批次大小）上的形状相同。
  # 并且确保 query 和 key 在最后一个维度（通常是特征或头的数量）上的形状相同。因为查询和键的点积是在最后一个维度上计算
  assert query.dtype == key.dtype == value.dtype
  input_dtype = query.dtype

  # 确保在注意力机制中的轴参数和张量形状符合预期，以确保注意力计算的正确性
  if axis is None:
    axis = tuple(range(1, key.ndim - 2)) #注意力轴将包括所有除了批次轴和最后两个轴之外的轴。
  if not isinstance(axis, Iterable):
    axis = (axis,) #保证axis的可迭代
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  # 轴范围验证
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2): #要保证axis 必须在 1 到 query 维度数量减去 2 之间（即排除批次轴和最后两个轴）。
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1] #通道数
  n = key.ndim

  # batch_dims is  <bs, <non-attention dims>, num_heads> axis 在这里对应所有指定的注意力轴 n-1对应的是最后一个轴（通道轴）
  batch_dims = tuple(np.delete(range(n), axis + (n - 1,))) #计算批次维度，所有与后续计算输出无关的轴删去剩下的轴的数量就是批次维度

  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,) #新轴的顺序  将无用的轴放在了前面 有用的放在了后面
  key = key.transpose(qk_perm) #使用 qk_perm 对 key 和 query 进行转置，
  query = query.transpose(qk_perm)# 使它们的形状变为 (bs, <non-attention dims>, num_heads, <attention dims>, channels)。

  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  # value 的新轴顺序是 {无用轴，输出通道轴，注意力轴}
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  key = key.astype(dtype)
  query = query.astype(dtype) / np.sqrt(depth) #缩放query
  batch_dims_t = tuple(range(len(batch_dims))) #获得批次维度的索引
  attn_weights = jax.lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision) #点积注意力权重 在最后一个维度（通道轴）上进行点积，并指定哪些维度上的批次处理应该进行广播
  # 换句话说(batch_dims_t, batch_dims_t)就是通过明确定义哪些维度被视为批次维度，并要求这些批次维度并行运算

  # 应用注意力偏置：掩码、丢弃、邻近偏置等
  '''
  这里的bias 可以实现多种功能 例如：
  掩码、在处理序列数据时，可能需要屏蔽（masking）未来的时间步，以防止信息泄露。
  丢弃、随机将一些权重设置为零，以防止模型过拟合。
  邻近偏置、给予与当前元素位置接近的元素更高的权重。
  '''
  if bias is not None:
    attn_weights = attn_weights + bias

  # 正则化注意力权重
  # attn_weights.ndim - len(axis)在这里表示的是不直接参与到注意力计算中的
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = jax.nn.softmax(attn_weights, axis=norm_dims) #处理的就是从norm_dims开始
  assert attn_weights.dtype == dtype
  attn_weights = attn_weights.astype(input_dtype)

  # 根据注意力权重计算新value
  # 没懂
  assert attn_weights.dtype == value.dtype
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = jax.lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # 恢复为 (批次大小, 维度1, 维度2, ..., 维度N, 头数, 通道数)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  assert y.dtype == input_dtype
  return y


def _invert_perm(perm):
  '''
  反转一个给定的排列,将这个序列中的每个元素放回到它原来的位置。
  Args:
    perm:perm 数组在这里代表维度的排列顺序。每个元素 j 在 perm 中的位置 i 表示原始维度 j 在排列后移动到了第 i 个维度。

  Returns:给定的排列的维度如何移动 恢复到原始的排列

  '''
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
  #每个元素的索引 i 和对应的值 j。
    perm_inv[j] = i
  return tuple(perm_inv)


class Base2FourierFeatures(nn.Module):
  # 负责生成傅里叶特征
  start: int = 0 #生成 Fourier 特征的起始频率
  stop: int = 8 #生成 Fourier 特征的结束频率
  step: int = 1 #生成 Fourier 特征的频率步长

  @nn.compact
  def __call__(self, inputs):
    # 生成一个频率序列
    # 每个元素 f 在序列 freqs 中代表了一个特定的频率指数,
    # 这些频率指数将被用来计算每个频率对应的 2的f次幂 * 2π 值，这些值随后用于生成正弦和余弦特征
    freqs = range(self.start, self.stop, self.step)

    # Create Base 2 Fourier features
    # 这些特征通过将输入数据与不同频率的正弦和余弦函数相结合来捕获数据的周期性

    # 计算每个频率对应的权重 w 这个权重是由频率周期2pi以及一个指数函数频率计算得到的
    # 从而实现对于不同频率有不同的权重  从而有助于捕捉输入数据中的高频特征
    # 为每个频率索引 f 生成一个对应的权重，形式为 (2^f) * 2π。这些权重随后将用于与输入数据的正弦和余弦变换相结合，生成傅里叶特征。
    w = 2.**(jnp.asarray(freqs, dtype=inputs.dtype)) * 2 * jnp.pi
    # 将权重向量的扩展
    '''
    w[None, :]中的None 类似于 numpy 中的 np.newaxis,对权重矩阵增加了一个维度 形状变为 (1, n_freqs)
    jnp.tile 函数将 w[None, :] 沿第二个维度(列)复制 inputs 的最后一个维度的大小次数。形状变为  (1, inputs.shape[-1])或(1, len(freqs))
    从而w 就可以与 inputs 进行广播
    '''
    w = jnp.tile(w[None, :], (1, inputs.shape[-1]))

    # Compute features
    # 对于 inputs 中的最后一个维度上的每个元素，jnp.repeat 会根据 len(freqs) 的值重复这些元素。
    '''
    例如，如果 inputs 是形状为 (batch_size, features) 的二维数组，freqs 有 3 个频率点，那么也就是为每个频率点都复制一份特征
inputs 的一个元素，比如第 i 个样本的第 j 个特征，inputs[i, j]，将被重复 3 次，变成 h[i, j, :] = [inputs[i, j], inputs[i, j], inputs[i, j]]。
结果 h 的形状将是 (batch_size, features, len(freqs))。
    '''
    h = jnp.repeat(inputs, len(freqs), axis=-1)
    # 对不同的特征频率赋予不同的权重 从而实现每个特征在不同频率上的放大和缩小
    h = w * h
    # 捕捉傅里叶特征
    # 通过将h 中的每个元素计算对应的sin和cos值并沿最后一个维度拼接起来
    # h 变成了一个形状为 (batch_size, num_features, 2 * len(freqs))
    h = jnp.concatenate([jnp.sin(h), jnp.cos(h)], axis=-1)
    return h
