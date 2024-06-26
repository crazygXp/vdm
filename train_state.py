# Copyright 2022 The VDM Authors, Flax Authors.
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

"""
Adapted from
https://flax.readthedocs.io/en/latest/_modules/flax/
training/train_state.html#TrainState.

But with added EMA of the parameters.
"""

import copy
from typing import Any, Callable, Optional

from flax import core
from flax import struct
import jax
import optax


class TrainState(struct.PyTreeNode):

  """针对只有一个 Optax 优化器的常见情况的简单训练状态

   创建训练实例和更新训练实例的参数,这两个方法共同支持了机器学习模型训练的完整流程：
   首先是初始化训练状态，然后是迭代地更新这些状态以优化模型性能
  Synopsis:

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx)
    grad_fn = jax.grad(make_loss_fn(state.apply_fn))
    for batch in data:
      grads = grad_fn(state.params, batch)
      state = state.apply_gradients(grads=grads)

  请注意，你可以通过子类化这个数据类来轻松扩展它，用于存储额外的数据（例如额外的变量集合）。).
  对于更特殊的用例（例如多个优化器），最好是fork该类并进行修改

  属性:
    step: 计数器从0开始，每次调用 .apply_gradients() 时递增。
    apply_fn:通常设置为 model.apply()。为了在训练循环中的 train_step() 函数中简化参数列表而保留在这个数据类中。
    tx:一个 Optax 梯度变换。
    opt_state: tx 的状态
  """
  # 类属性
  step: int #训练步数
  params: core.FrozenDict[str, Any] #不可变的字典core.FrozenDict 存储模型参数
  ema_params: core.FrozenDict[str, Any] # 一个不可变的字典，存储指数移动平均参数，用于平滑模型参数。
  opt_state: optax.OptState # 存储优化器状态
  tx_fn: Callable[[float], optax.GradientTransformation] = struct.field(
      pytree_node=False) # 返回一个 optax.GradientTransformation 对象通常用于优化过程中，比如计算梯度更新。
  # optax.GradientTransformation 是 Optax 库中用于定义如何将梯度转换为参数更新的接口。
  # 简单来说就是 根据传入的梯度和当前参数，计算出参数的更新值，并返回新的参数和更新后的优化器状态。
  apply_fn: Callable = struct.field(pytree_node=False) # 模型的前向传播函数。

  def apply_gradients(self, *, grads, lr, ema_rate, **kwargs):
    """函数更新返回值中的 step、params、opt_state 和 **kwargs

    在给定梯度 grads 的情况下更新模型参数、优化器状态，并应用指数移动平均（EMA）到参数上
    请注意，该函数在内部首先调用 .tx.update()，然后调用 optax.apply_updates() 来更新 params 和 opt_state。

    Args:
      grads: 梯度，其结构与 .params 相同。
      **kwargs: 应该通过 .replace() 替换的额外数据类属性。

    Returns:
      更新后的 self 实例，其中 step 增加了一，params 和 opt_state 通过应用 grads 进行了更新，
      并且额外的属性按照 kwargs 中指定的进行了替换。
    """
    tx = self.tx_fn(lr) #这个对象 tx 包含了执行梯度更新所需的所有信息和方法，特别是 update 方法，它将被用来应用梯度下降步骤。
    '''
    GradientTransformation 对象。这个对象包含了两个主要的方法：
    init：初始化优化器状态。这通常在训练开始之前调用一次。   
    update：根据传入的梯度和当前参数，计算出参数的更新值，并返回新的参数和更新后的优化器状态。
    这里的tx就是GradientTransformation 对象。
    '''
    # 计算更新值
    updates, new_opt_state = tx.update(
        grads, self.opt_state, self.params)
    # 应用更新值
    new_params = optax.apply_updates(self.params, updates)
    new_ema_params = jax.tree_multimap(
        lambda x, y: x + (1. - ema_rate) * (y - x),
        self.ema_params,
        new_params,
    )
    # 创建当前数据类的一个新的实例，同时可以替换掉其中的一些字段的值
    return self.replace(
        step=self.step + 1,
        params=new_params,
        ema_params=new_ema_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(_class, *, apply_fn, variables, optax_optimizer, **kwargs):

    """创建一个新实例，其中 step 设为 0，并初始化 opt_state."""
    '''
    Args:
    _class：这是一个指向 TrainState 类的引用，允许在方法内部创建类的实例。
    apply_fn：模型的前向传播函数，通常设置为模型的 .apply() 方法。
    variables：一个包含模型参数的字典，键为 "params"。
    optax_optimizer：一个函数，用于创建 optax.GradientTransformation 对象，这个对象定义了优化算法。
    **kwargs：接受任意数量的额外关键字参数，这些参数将被传递给 _class 的构造函数。
    
    Returns: 一个新创建的类实例
    '''
    # _class is the TrainState class
    params = variables["params"]
    opt_state = optax_optimizer(1.).init(params)  #初始化优化器状态
    ema_params = copy.deepcopy(params) #初始化ema
    return _class(
        step=0,
        apply_fn=apply_fn,
        params=params,
        ema_params=ema_params,
        tx_fn=optax_optimizer,
        opt_state=opt_state,
        **kwargs,
    )
