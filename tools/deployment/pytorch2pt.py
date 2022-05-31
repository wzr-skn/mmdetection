import argparse
import warnings
import mmcv
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


def recursive_fuse_conv(module, prefix=''):
    for name, child in module._modules.items():
        if not hasattr(child, 'fuse_conv'):
            recursive_fuse_conv(child, prefix + name + '.')
        else:
            child.fuse_conv()

class NhwcWrapper(torch.nn.Module):
    def __init__(self, model):
        super(NhwcWrapper, self).__init__()
        self._model = model

    def forward(self, input_tensor):
        nchw_input_tensor = input_tensor.permute(0, 3, 1, 2)
        return self._model(nchw_input_tensor)


def pytorch2pt(model,
               input_shape,
               transpose=False,
               output_file='tmp.pt',
               ):
    forward_dummy = model.forward_dummy
    model.forward = forward_dummy

    if transpose:
        nhwc_model = NhwcWrapper(model)
        trace_data = torch.randn(input_shape).permute(0, 2, 3, 1)
        trace_model = torch.jit.trace(nhwc_model.cpu().eval(), (trace_data))
        torch.jit.save(trace_model, output_file)

    else:
        trace_data = torch.randn(input_shape)
        trace_model = torch.jit.trace(model.cpu().eval(), (trace_data))
        torch.jit.save(trace_model, output_file)

def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMDetection models to pt')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='tmp.pt')
    parser.add_argument('--transpose', type=bool, default=False, help='whether to convert nchw format to nhwc format')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 128, 128],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    warnings.warn('Arguments like `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and \
        will be removed in future releases.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)
    model.eval()

    try:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    except IOError:
        res = input("checkpoint doesn't exist, still export onnx with random init model? yes/no:")
        if res == "yes" or "y":
            pass
    recursive_fuse_conv(model)
    input_shape = args.shape
    transpose = args.transpose
    # convert model to pt file
    pytorch2pt(
        model,
        input_shape,
        transpose,
        output_file=args.output_file)
