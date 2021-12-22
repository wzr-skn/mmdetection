import asyncio
from argparse import ArgumentParser
import os
import warnings
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_file', help='Image file')
    parser.add_argument('out_file', help='output file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')

    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args

def show_result_pyplot(model,
                       img,
                       out_file,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=False,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        out_file=out_file)

def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    assert os.path.isdir(args.out_file)
    img_file_list = os.listdir(args.img_file)
    for img_name in img_file_list:
        if  img_name[-3:] not in ["jpg", "png", "bmp"]:
            warnings.warn(f"{img_name} is not a image name")
            continue
        img_path = os.path.join(args.img_file, img_name)
        out_path = os.path.join(args.out_file, img_name)

        result = inference_detector(model, img_path)
        show_result_pyplot(model, img_path, out_path, result, score_thr=args.score_thr)


    # show the results

# ../../../../media/traindata/coco/val2017/images/   ../../../../media/traindata/yolox_test/

async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results

    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
