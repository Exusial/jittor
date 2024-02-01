import jittor as jt
from jittor import nn

def _weight_norm(v, g, dim):
    "\n权重归一化。权重归一化是一种重新参数化神经网络，提高网络泛化能力的常用方法。具体公式为 :math:`v_{norm} = v * (g / || v ||_2)`\n\n    参数:\n        - v (Jittor.Var): 待归一化的权重。\n        - g (Jittor.Var): 缩放因子，用于调整归一化权重的范围。\n        - dim (int): 归一化操作在该维度上进行。\n\n    返回值:\n        Jittor.Var: 归一化后的权重。\n\n    示例：\n        >>> from jittor import weightnorm\n        >>> v = jt.array([1.0, 2.0, 3.0])\n        >>> g = jt.array([1.0])\n        >>> weightnorm._weight_norm(v, g, 0)\n        jt.Var([0.26726124 0.5345225  0.8017837 ], dtype=float32)\n\n\n"
    return v * (g / jt.norm(v, 2, dim, keepdim=True))

class WeightNorm(object):
    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: nn.Module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int):
        if hasattr(module, '__fhook2__') and isinstance(module.__fhook2__, WeightNorm):
            raise RuntimeError("Cannot register two weight_norm hooks on "
                                "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        # todo: add check
        # remove w from parameter list
        # del module._parameters[name]
        delattr(module, name)

        # add g and v as new parameters and express w as g/||v|| * v
        module.__setattr__(name + '_g', jt.norm(weight, 2, dim, keepdim=True).detach())
        module.__setattr__(name + '_v', weight.detach())
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        # todo: support multiple hook in a module
        module.register_pre_forward_hook(fn)
        return fn

    def remove(self, module: nn.Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        delattr(module, self.name + '_g')
        delattr(module, self.name + '_v')
        setattr(module, self.name, weight.detach())

    def __call__(self, module: nn.Module, inputs) -> None:
        setattr(module, self.name, self.compute_weight(module))

def weight_norm(module, name, dim):
    "\n对模块增加一个权重归一化操作。通过应用WeightNorm函数实现，即对权重矩阵每一个切片（在维度dim上）进行 :math:`L_2` 范数归一化，数学公式描述如下：设权重矩阵为W，经过归一化后的权重矩阵为W'，则有 :math:`W' = \\frac{W}{||W||_2}`\n\n    参数:\n        - module: 输入的模型，类型为模型对象\n        - name (Jittor.Var): 指定的参数的名称\n        - dim (int): 进行权重归一化的维度\n\n    返回值:\n        处理后的模型对象。\n\n    代码示例：\n        >>> import jittor as jt\n        >>> from jittor import weightnorm\n        >>> class jt_module(jt.nn.Module):\n        >>>         def __init__(self, weight):\n        >>>             super().__init__()\n        >>>             self.linear = jt.array(weight)\n        >>> \n        >>>         def execute(self, x):\n        >>>             return jt.matmul(self.linear, x)\n        >>> \n        >>> jm = jt_module(weight)\n        >>> weightnorm.weight_norm(jm, 'linear', -1)\n\n"
    WeightNorm.apply(module, name, dim)
    return module

def remove_weight_norm(module, name: str = 'weight'):
    "\n移除模块的权重归一化。该函数通过检查模块是否有权重归一化相关的属性 ``__fhook2__`` 来判断是否对该模块进行过权重归一化操作，如果存在，则删除该属性，用于消除对模块的权重归一化操作。\n\n    参数:\n        - module (Module): 需要移除权重归一化的模块\n        - name (str, 可选): 权重属性的名称. 默认值： 'weight'\n\n    返回值:\n        若模块存在权重归一化属性，返回删除权重归一化属性后的模块。如果模块中没有找到指定的权重归一化属性，会抛出ValueError。\n\n    代码示例：\n        >>> from jittor import weightnorm\n        >>> from jittor import nn\n        >>> model = nn.Linear(20, 40)\n        >>> model = weightnorm.weight_norm(model, 'weight', -1)\n        >>> hasattr(model,\"__fhook2__\")\n        True\n        >>> model = weightnorm.remove_weight_norm(model, 'weight')\n        >>> hasattr(model,\"__fhook2__\")\n        False\n\n"
    if hasattr(module, "__fhook2__") and isinstance(module.__fhook2__, WeightNorm):
        delattr(module, "__fhook2__")
        return module
    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))