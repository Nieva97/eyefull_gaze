# conda env list
# conda activate nieva_TFM

import argparse
import sys
import os

# gaze imports
import cv2
import math
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib as mpl
from model import ModelSpatial
from utils import imutils, evaluation
from config import *

# YOLO
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

mpl.use('Qt5Agg')

labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']  # class names

parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='./models/gaze_follow/model_demo.pt')
parser.add_argument('--image_dir', type=str, help='images',
                    default='/home/alvaro.nieva/Documents/nieva_TFM/data/gif_cocina_avocado/frames')
parser.add_argument('--head', type=str, help='head bounding boxes',
                    default='/home/alvaro.nieva/Documents/nieva_TFM/data/gif_cocina_avocado/person1.txt')
# /home/alvaro.nieva/Documents/nieva_TFM/data/ETRI/used_videos/A025_P030_G001_C002.txt
parser.add_argument('--save_fig', type=int, help='guarda las imag', default=0)
parser.add_argument('--results_dir', type=str, help='saved images directory',
                    default='/home/alvaro.nieva/Documents/nieva_TFM/resultados/full_run_iou_fixed_50percent/')
parser.add_argument('--vis_mode', type=str, help='heatmap, arrow or yolo plot mode', default='heatmap')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
parser.add_argument('--iou_punto', type=int, help='mira a un objeto detectado', default=0)
parser.add_argument('--iou_threshold', type=float, help='IoU % needed for detection', default=0.4)
parser.add_argument('--heatmap_threshold', type=float, help='threshold used as % in heatmap ', default=0.7)

# YOLO arguments
parser.add_argument('--weights', nargs='+', type=str, default='./yolov5x.pt', help='model path(s)')
parser.add_argument('--data', type=str, default='./data/coco128.yaml', help='(optional) dataset.yaml path')
parser.add_argument('--conf-thres', type=float, default=0.40, help='confidence threshold')  # Default: 0.25
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--augment', default="False", action='store_true', help='augmented inference')
parser.add_argument('--visualize', default="False", action='store_true', help='visualize features')
parser.add_argument('--classes', nargs='+', default="None", help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--agnostic-nms', default="False", action='store_true', help='class-agnostic NMS')
parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
args = parser.parse_args()


def _get_transform():
    # Hace un resize y normaliza la imagen ¿Por qué se normaliza?
    # Normalization helps get data within a range and reduces the skewness which helps learn faster and better.
    # Normalization can also tackle the diminishing and exploding gradients problems.

    transform_list = []

    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    # En el resize, no acabo de ver de donde sacar las variables input_resolution

    transform_list.append(transforms.ToTensor())
    # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor.
    # and scales the image’s pixel intensity values in the range [0, 1.]
    # Pasas a tensor porque normalize solo se puede utilizar en tipo Tensor

    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # De donde puede sacar estos valores? Los ha sacado de un tutorial de Pytorch

    return transforms.Compose(transform_list)
    # Permite encadenar todas las transformaciones anteriores


def run():
    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(args.head, names=column_names, index_col=0)
    df['left'] -= (df['right'] - df['left']) * 0.1
    df['right'] += (df['right'] - df['left']) * 0.1
    df['top'] -= (df['bottom'] - df['top']) * 0.1
    df['bottom'] += (df['bottom'] - df['top']) * 0.1

    # set up data transformation
    test_transforms = _get_transform()

    # gaze detection model
    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # YOLO detection model
    size_yolo = 640
    imgsz = [640, 640]
    device = select_device(args.device)
    model_yolo = DetectMultiBackend(args.weights, device=device, dnn=args.dnn, data=args.data, fp16=args.half)
    stride, names, pt = model_yolo.stride, model_yolo.names, model_yolo.pt
    bs = 1  # batch size
    model_yolo.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    model.cuda()
    model.train(False)

    # Figure & other global variables
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
    label_yolo = []
    number_images = df.shape[0]
    seen = 0
    seen_inside = 0
    dt = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for i in df.index:
            seen += + 1
            percentage = 100 * seen / number_images
            print('\rTracking frames: {}%, total of frames: {}'.format(percentage.__round__(2), number_images), end='')
            t1 = time_sync()
            frame_raw_cv_transpose = cv2.imread(os.path.join(args.image_dir, i))
            # gaze algorithm needs an RGB data with channel last (PIL image), YOLO needs a BGR with channels first
            frame_raw_cv = cv2.cvtColor(frame_raw_cv_transpose, cv2.COLOR_BGR2RGB)
            frame_raw = Image.fromarray(frame_raw_cv)
            width, height = frame_raw.size

            # if it is smaller than
            if width and height >= 960:
                do_crop = 1
            else:
                do_crop = 0

            # there is a bug on pandas, sometimes it returns the same value two times. It is fixed with this If
            aux = df.loc[i, 'left']
            coord = []
            if aux.size > 1:
                k = 0
                for n in column_names:
                    if n == "frame":
                        continue
                    aux = df.loc[i, n]
                    coord.append(aux[0])
                    k = k + 1
                head_box = [coord[0], coord[1], coord[2], coord[3]]
            else:
                head_box = [df.loc[i, 'left'], df.loc[i, 'top'], df.loc[i, 'right'], df.loc[i, 'bottom']]

            # If the is no Face Detection the value returned is 0,0,0,0, else: 954.0,409.0,1115.0,624.0
            if head_box[0] == head_box[1] == head_box[2] == head_box[3] == 0:
                plt.clf()
                fig.canvas.manager.window.move(0, 0)
                plt.axis('off')
                plt.imshow(frame_raw)
                # fig.suptitle("No Detection", fontsize=14, fontweight='bold')
            else:
                # If there is a bounding box, the gaze orientation is estimated
                head = frame_raw.crop((head_box))  # head crop
                # plt.imshow(head)
                # plt.show()

                # transform inputs
                head = test_transforms(head)
                frame = test_transforms(frame_raw)
                # generate binary image
                head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width,
                                                            height,
                                                            resolution=input_resolution).unsqueeze(0)

                head = head.unsqueeze(0).cuda()
                # .Cuda -> cpu Tensor to gpu tensor
                # Unsqueeze changes array dimensions, como pone 0 se queda en 1D

                frame = frame.unsqueeze(0).cuda()
                head_channel = head_channel.unsqueeze(0).cuda()

                # forward pass the CNN
                raw_hm, _, inout = model(frame, head_channel, head)

                # heatmap modulation
                raw_hm = raw_hm.cpu().detach().numpy() * 255
                # To go from a Tensor that requires_grad to one that does not, use .detach()
                # To go from a gpu Tensor to cpu Tensor, use .cpu().
                # Tp gp from a cpu Tensor to np.array, use .numpy().

                raw_hm = raw_hm.squeeze()
                # To squeeze a tensor, we use the torch.squeeze() method. It returns a new tensor with all the
                # dimensions of the input tensor but removes size 1. For example, if the shape of the input tensor is
                # (M ☓ 1 ☓ N ☓ 1 ☓ P), then the squeezed tensor will have the shape (M ☓ M ☓ P).

                # plt.imshow(raw_hm)
                # plt.show()

                inout = inout.cpu().detach().numpy()
                # scalar alpha which quantifies whether the person’s focus of attention is located inside or outside the
                # frame. The modulation is performed by an element-wise subtraction of the (1−alpha) from the normalized
                # full-sized feature map, followed by clipping of the heatmap such that its minimum values are ≥ 0
                # Por lo tanto, inout es un solo número de 0 a 1

                inout = 1 / (1 + np.exp(-inout))
                inout = (1 - inout) * 255

                size_cv = (width, height)
                norm_map = cv2.resize(raw_hm, dsize=size_cv) - inout

                # Timing gaze algorithm + open image
                t2 = time_sync()
                dt[0] += t2 - t1

                plt.clf()
                # clear figure: it does not pop up the fig window
                fig.canvas.manager.window.move(0, 0)
                plt.axis('off')

                ax = plt.gca()
                # THE FOLLOWING LINE ADDS HEAD BOUNDING-BOX
                rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2] - head_box[0],
                                         head_box[3] - head_box[1],
                                         linewidth=2, edgecolor=(0, 1, 0), facecolor='none')

                label_yolo = []
                # If the person is looking at an object inside the camera's plane, a cropped image is performed
                t3 = time_sync()
                if inout < args.out_threshold:  # in-frame gaze, va de 0 a 255, con un treshold de 100
                    looking_outside = 0
                    seen_inside += + 1
                    pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    # heatmap maximum value (64,64)

                    norm_p = [pred_x / output_resolution, pred_y / output_resolution]
                    # Normalized units to pixels [0.53125, 0.453125] -> 1030, 450

                    coor_y = norm_p[1] * size_cv[1]
                    coor_x = norm_p[0] * size_cv[0]
                    # reescale to frame size

                    # This snippet crops the image, centered on heatmap maximum value
                    left_bot_x = coor_x - size_yolo / 2
                    left_bot_x = round(left_bot_x)
                    left_bot_y = coor_y - size_yolo / 2
                    left_bot_y = round(left_bot_y)
                    right_top_x = coor_x + size_yolo / 2
                    right_top_x = round(right_top_x)
                    right_top_y = coor_y + size_yolo / 2
                    right_top_y = round(right_top_y)
                    # If the cropped image has area outside the image frame, the box is recalculated
                    if right_top_x > width:
                        dif = right_top_x - width
                        left_bot_x = left_bot_x - dif
                        # right_top_x = width
                    if right_top_y > height:
                        dif = right_top_y - height
                        left_bot_y = left_bot_y - dif
                        # right_top_y = height
                    if left_bot_x < 0:
                        # not used coord
                        # dif = -left_bot_x - 0
                        # right_top_x = right_top_x + dif
                        left_bot_x = 0
                    if left_bot_y < 0:
                        # not used coord
                        # dif = -left_bot_y - 0
                        # right_top_y = right_top_y + dif
                        left_bot_y = 0

                    # Cropped image and heatmap
                    norm_map_yolo = norm_map[left_bot_y:left_bot_y + 640, left_bot_x:left_bot_x + 640]
                    yolo_raw_np = frame_raw_cv_transpose[left_bot_y:left_bot_y + 640, left_bot_x:left_bot_x + 640]
                    yolo_raw_np = np.ascontiguousarray(yolo_raw_np)

                    # Heatmap Normalization
                    norm = (norm_map_yolo - np.min(norm_map_yolo)) / (np.max(norm_map_yolo) - np.min(norm_map_yolo))
                    # generate a binary image: 1 higher than threshold else 0
                    norm_threshold = norm > args.heatmap_threshold
                    # https://stackoverflow.com/questions/66597852/reading-an-2d-array-or-list-with-opencv
                    # If uint8 type is not used, it asserts an error in findCountours
                    norm_threshold = np.array(norm_threshold, np.uint8)
                    heatmap_gray = Image.fromarray(norm_threshold)
                    heatmap_gray = np.asarray(heatmap_gray)
                    # contours return the points around a binary image (the heatmap)
                    contours, hierarchy = cv2.findContours(heatmap_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour = np.array([list(pt[0]) for ctr in contours for pt in ctr])
                    # generates heatmap maximum value bounding box: top left and botton right corners
                    x3 = contour[:, 0].min()
                    y3 = contour[:, 1].min()
                    x4 = contour[:, 0].max()
                    y4 = contour[:, 1].max()
                    width_hm = abs(x4 - x3)
                    height_hm = abs(y4 - y3)
                    # print(hm__left_x, hm__left_y, hm__right_x, hm__right_y, sep=' ')
                    cv2.rectangle(heatmap_gray, (x3, y3), (x4, y4), (36, 255,12), 2)

                    # YOLO
                    yolo_raw_np_transpose = yolo_raw_np.transpose((2, 0, 1))[::-1]
                    yolo_raw_np_transpose = np.ascontiguousarray(yolo_raw_np_transpose)
                    # https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
                    im = torch.from_numpy(yolo_raw_np_transpose).to(device)
                    im = im.half() if model_yolo.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]

                    # Inference
                    t3_0 = time_sync()
                    pred = model_yolo(im, augment=args.augment, visualize=args.visualize)
                    classes = None
                    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes, args.agnostic_nms,
                                               max_det=args.max_det)

                    # Timing YOLO prediction
                    t3_1 = time_sync()
                    dt[1] += t3_1 - t3_0

                    for j, det in enumerate(pred):
                        gn = torch.tensor(yolo_raw_np.shape)[[1, 0, 1, 0]]
                        annotator = Annotator(yolo_raw_np, line_width=args.line_thickness, example=str(names))
                        if len(det):
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], yolo_raw_np.shape).round()
                            """for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s = names[int(c)]"""
                            # En resumen, line tiene la siguiente info: número de clase + las 4 coordenadas que da YOLO
                            for *xyxy, conf, cls in reversed(det):
                                # YOLO bounding-box, center, width and height
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (int(cls), *xywh)
                                center_x = line[1] * size_yolo
                                center_y = line[2] * size_yolo
                                width_yolo = line[3] * size_yolo
                                height_yolo = line[4] * size_yolo
                                # x1,y1 top left corner; x2,y2 bot left corner
                                x1 = center_x - width_yolo / 2
                                y1 = center_y - height_yolo / 2
                                x2 = center_x + width_yolo / 2
                                y2 = center_y + height_yolo / 2

                                # Label for YOLO print on image
                                c = int(cls)  # integer class
                                label = None if args.hide_labels else (
                                    names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                                # Object looked at
                                if args.iou_punto == 1:
                                    norm_p_yolo = evaluation.argmax_pts(norm_map_yolo)
                                    if (center_x + width_yolo / 2) >= norm_p_yolo[0] >= (center_x - width_yolo / 2):
                                        if (center_y + height_yolo / 2) >= norm_p_yolo[1] >= (
                                                center_y - height_yolo / 2):
                                            label_yolo.append(names[line[0]])
                                else:
                                    # First, top left corners must be compared to know the top left corner
                                    x_inter1 = max(x1, x3)
                                    y_inter1 = max(y1, y3)
                                    # Then, the botton right corner
                                    x_inter2 = min(x2, x4)
                                    y_inter2 = min(y2, y4)
                                    # intersection area is the rectangle formed
                                    # width_inter = abs(x_inter2 - x_inter1)
                                    # height_inter = abs(y_inter2 - y_inter1)
                                    # area_inter = width_inter * height_inter
                                    area_inter = abs(max((x_inter2 - x_inter1, 0)) * max((y_inter2 - y_inter1), 0))
                                    # diferencia = interArea - area_inter
                                    # Union of boxes => total area covered by them
                                    area_box1 = abs(width_yolo * height_yolo)
                                    area_box2 = abs(width_hm * height_hm)
                                    area_union = area_box1 + area_box2 - area_inter
                                    iou = area_inter / area_box2
                                    if iou > args.iou_threshold:
                                        label_yolo.append(names[line[0]])
                                        # label_yolo.append(str(diferencia))

                    # After for loop color is changed to being plotted with plt
                    yolo_raw_np = cv2.cvtColor(yolo_raw_np, cv2.COLOR_BGR2RGB)
                else:
                    looking_outside = 1

            # Timing general YOLO inference (inference + IoU/object looked)
            t4 = time_sync()
            dt[2] += t4 - t3

            # Plot the data
            if args.vis_mode == 'arrow':
                if inout < args.out_threshold:  # in-frame gaze, va de 0 a 255, con un treshold de 100
                    frame_raw = np.asarray(frame_raw)
                    frame_raw[left_bot_y:left_bot_y + 640, left_bot_x:left_bot_x + 640] = yolo_raw_np
                    frame_raw = Image.fromarray(frame_raw)
                    plt.imshow(frame_raw)
                    ax.add_patch(rect)
                    # Yellow ball
                    circ = patches.Circle((norm_p[0] * width, norm_p[1] * height), height / 50.0, facecolor=(1, 1, 0),
                                          edgecolor='none')
                    ax.add_patch(circ)
                    # Green line
                    plt.plot((norm_p[0] * width, (head_box[0] + head_box[2]) / 2),
                             (norm_p[1] * height, (head_box[1] + head_box[3]) / 2), '-', color=(0, 1, 0, 1))
                else:
                    plt.imshow(frame_raw)

            elif args.vis_mode == 'heatmap':
                if looking_outside == 0:
                    frame_raw = np.asarray(frame_raw)
                    frame_raw[left_bot_y:left_bot_y + 640, left_bot_x:left_bot_x + 640] = yolo_raw_np
                    frame_raw = Image.fromarray(frame_raw)
                plt.imshow(frame_raw)
                ax.add_patch(rect)
                plt.imshow(norm_map, cmap='jet', alpha=0.3, vmin=0, vmax=255)

            elif args.vis_mode == 'yolo':
                # plt.imshow(heatmap_gray)
                # plt.imshow(norm_map)
                plt.imshow(yolo_raw_np)
                plt.imshow(norm_map_yolo, cmap='jet', alpha=0.3, vmin=0, vmax=255)

            # Write image's title
            # ax.set_title(titulo, fontsize=32)
            if label_yolo:
                titulo = 'Looking at: ' + str(label_yolo)
                fig.suptitle(str(titulo), fontsize=28, fontweight='bold')
            elif looking_outside == 1:
                fig.suptitle("Looking outside the plane", fontsize=28, fontweight='bold')
            else:
                fig.suptitle("No object recognized", fontsize=28, fontweight='bold')

            # Save image or plot, else current figure changes size , feels weird. If you do a clear
            # figure it resets its size, and default is 640*480 pixels and it is needed 1920x1080
            if args.save_fig == 1:
                if looking_outside == 0 or args.vis_mode == "heatmap" or args.vis_mode == "arrow":
                    file = str(args.results_dir + i)
                    fig.set_size_inches(19.20, 10.80)
                    plt.savefig(file)  # , format="png")
            else:
                plt.show(block=False)
                plt.pause(0.04)  # 25 fps
                # plt.show()

            # Timing plotting/saving image
            t5 = time_sync()
            dt[3] += t5 - t4 - 0.04  # plt.pause

        print('\nDONE!')
        # Calculate average time in each section
        t = tuple((x / number_images) * 1E3 for x in dt)
        LOGGER.info(f'Speed: %.1fms gaze algorithm inference, %.1fms inference, %.1fms general YOLO inference, '
                    f'%.1fms plotting/save image' % t)


if __name__ == "__main__":
    run()
