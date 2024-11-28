""""by lyuwenyu
"""

import os
import yaml 
import inspect
import importlib

__all__ = ['GLOBAL_CONFIG', 'register', 'create', 'load_config', 'merge_config', 'merge_dict']


GLOBAL_CONFIG = dict()
INCLUDE_KEY = '__include__'


def register(cls: type):
    '''
    Args:
        cls (type): Module class to be registered.
    '''
    if cls.__name__ in GLOBAL_CONFIG:
        raise ValueError('{} already registered'.format(cls.__name__))

    if inspect.isfunction(cls):
        GLOBAL_CONFIG[cls.__name__] = cls
    
    elif inspect.isclass(cls):
        # 提取 cls 类的相关信息并组织成 schema { 隐藏参数(下划线参数)、声明参数 }
        GLOBAL_CONFIG[cls.__name__] = extract_schema(cls)

    else:
        raise ValueError(f'register {cls}')

    return cls 


def extract_schema(cls: type):
    '''
    Args:
        cls (type),
    Return:
        Dict, 
    '''
    # 获取相关类的详细参数信息
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    num_defualts = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defualts

    # 将类的相关信息整合进一个字典里**
    schame = dict()
    schame['_name'] = cls.__name__
    # __module__ 找到导入 cls 的所在模块，importlib.import_module 则在运行中动态导入该模块
    schame['_pymodule'] = importlib.import_module(cls.__module__)
    # __inject__ 表示需要从外部注入的项，它所包含参数的参数值可能来自配置文件
    # __share__ 表示共享参数，通常在全局配置中定义，并在多个模块之间共享
    schame['_inject'] = getattr(cls, '__inject__', [])
    schame['_share'] = getattr(cls, '__share__', [])

    # 将参数列表中所有参数提取出来并存入 schame，带有默认值的则同时提取默认值
    # 提取默认值的依据：其他参数在前，默认参数在后
    for i, name in enumerate(arg_names):
        if name in schame['_share']:
            # 如果参数在共享参数列表中，必须有默认值，否则报错
            assert i >= num_requires, 'share config must have default value.'
            value = argspec.defaults[i - num_requires]
        # 判断当前从 arg_names 获取的参数是否为具有默认值的参数
        elif i >= num_requires:
            # 如果参数有默认值，从 defaults 中提取默认值
            value = argspec.defaults[i - num_requires]

        else:
            value = None 

        schame[name] = value
        
    return schame



def create(type_or_name, **kwargs):
    '''
    对于没有 __inject__ 和 __share__ 参数的类，create 执行的操作 -> 提取全局变量中保存的参数值，初始化 create 所制定的类 ，create 传入的往往是类名
    GLOBAL_CONIFG 所保存的变量包括类参数列表的初始值，其中有些具有默认值，有些则没有，和配置文件所补充或覆盖的值
    当需要初始化或修改类的某个参数时，则修改配置文件
    '''
    assert type(type_or_name) in (type, str), 'create should be class or name.'

    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    # 判断该类是否已经实例化（有 __dict__ 属性），已经实例化则直接返回该实例。
    if name in GLOBAL_CONFIG:
        if hasattr(GLOBAL_CONFIG[name], '__dict__'):
            return GLOBAL_CONFIG[name]
    else:
        raise ValueError('The module {} is not registered'.format(name))

    cfg = GLOBAL_CONFIG[name]

    # 如果 cfg 是字典类型，并且包含 'type' 字段，则表示这是一个嵌套配置，需要先解析 'type' 中指定的类 || coco_detection.yml
    # type 键值对只出现在配置文件中
    if isinstance(cfg, dict) and 'type' in cfg:
        _cfg: dict = GLOBAL_CONFIG[cfg['type']]
        _cfg.update(cfg) # update global cls default args || 用配置文件的信息去更新参数列表的相关信息
        _cfg.update(kwargs) # TODO
        name = _cfg.pop('type')
        
        return create(name)

    # 取出未初始化的类及待初始化的参数
    cls = getattr(cfg['_pymodule'], name)
    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']

    # 保存初始化类时可能用到的参数信息，比如默认参数
    cls_kwargs = {}
    cls_kwargs.update(cfg) # kw 初始值，处理完成 _share 和 _inject 后会更新 kw
    
    # shared var
    # 遍历配置中的共享参数列表（_share）
    for k in cfg['_share']:
        # 如果共享参数存在于GLOBAL_CONFIG中，从全局配置中获取；否则使用类配置中的默认值。
        if k in GLOBAL_CONFIG:
            cls_kwargs[k] = GLOBAL_CONFIG[k]
        else:
            cls_kwargs[k] = cfg[k]

    # inject
    for k in cfg['_inject']:
        _k = cfg[k]

        if _k is None:
            continue
        # 如果参数为字符串（类名），则从 GLOBAL_CONFIG 中获取对应配置，递归调用 create() 实例化对象。
        if isinstance(_k, str):            
            if _k not in GLOBAL_CONFIG:
                raise ValueError(f'Missing inject config of {_k}.')

            # 获取 inject 项在全局变量中的对应信息，一般是类的相关信息（隐含参数 + 声明参数）
            _cfg = GLOBAL_CONFIG[_k]
            
            if isinstance(_cfg, dict):
                cls_kwargs[k] = create(_cfg['_name'])
            else:
                cls_kwargs[k] = _cfg 

        # 如果参数为字典，并且包含 'type' 字段，则根据 'type' 指定的类实例化对象。
        # 例子: '_inject': ['dependency'] -> _k -> 'dependency': {'type': 'DependencyClass'}
        # dict 情况下，注入项必须具备 type 属性，且其属性值必须为已注册全局变量的类的字符串
        elif isinstance(_k, dict):
            if 'type' not in _k.keys():
                raise ValueError(f'Missing inject for `type` style.')

            _type = str(_k['type'])
            if _type not in GLOBAL_CONFIG:
                raise ValueError(f'Missing {_type} in inspect stage.')

            # TODO modified inspace, maybe get wrong result for using `> 1`
            _cfg: dict = GLOBAL_CONFIG[_type]
            # _cfg_copy = copy.deepcopy(_cfg)
            _cfg.update(_k) # update || 注入项: 注入项的类别信息
            # 将注入项转为已经实例化的类
            cls_kwargs[k] = create(_type)
            # _cfg.update(_cfg_copy) # resume

        else:
            raise ValueError(f'Inject does not support {_k}')


    cls_kwargs = {n: cls_kwargs[n] for n in arg_names}

    return cls(**cls_kwargs)



def load_config(file_path, cfg=dict()):
    '''load config
    '''
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        if file_cfg is None:
            return {}

    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)

            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            with open(base_yaml) as f:
                base_cfg = load_config(base_yaml, cfg)
                merge_config(base_cfg, cfg) # 这一步我认为多此一举，注释后不影响最终结果

    return merge_config(file_cfg, cfg)



def merge_dict(dct, another_dct):
    '''merge another_dct into dct
    '''
    for k in another_dct:
        # 如果键存在于 dict 中，并且两个字典的值都是字典类型，则递归合并
        if (k in dct and isinstance(dct[k], dict) and isinstance(another_dct[k], dict)):
            merge_dict(dct[k], another_dct[k])
        else:
            # 否则，用 another_dct 中的值覆盖 dct 中的值 || “无则填补，有则覆盖”
            dct[k] = another_dct[k]

    return dct



def merge_config(config, another_cfg=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.

    Returns: global config
    """
    global GLOBAL_CONFIG
    dct = GLOBAL_CONFIG if another_cfg is None else another_cfg
    
    return merge_dict(dct, config)



