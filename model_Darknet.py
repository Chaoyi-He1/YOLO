from build_utils.layers import *
from build_utils.parse_config import *

ONNX_EXPORT = False

def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除解析cfg列表中的第一个配置(对应[net]的配置)
    modules_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    # 统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef["type"] == "convolutional":
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 如果该卷积操作没有bn层，意味着该层为yolo的predictor
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            pass

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            layers = mdef["layers"]
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class DarkNet(nn.Module):
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(DarkNet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的.cfg文件
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构一层一层去搭建
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YOLOLayer层的索引
        # self.yolo_layers = get_yolo_layers(self)

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print('0', x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train
            return x
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])
            return p
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(in_features=16*16*1024, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=3)

    def forward(self, x):
        x = self.flat(x)
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        return x
