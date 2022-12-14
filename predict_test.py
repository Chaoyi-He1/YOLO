import os
import json
import time

import pandas as pd
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_objs


def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "./weights/yolov3spp-139.pt"  # 改成自己训练好的权重文件
    json_path = "./data/classes.json"  # json标签文件
    # img_path = "./my_yolo_dataset/val/images/00001.csv"
    img_list_dir = "./my_yolo_dataset/val/images"
    save_path = "./predict_results"
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
        print("The saving directory is created")
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    # assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.to(device)
    process_time = 0

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        image_files = tqdm(os.listdir(img_list_dir), desc="Predicting image")
        for file_name in image_files:
            img_path = os.path.join(img_list_dir, file_name)
            assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)
            # img_o = cv2.imread(img_path)  # BGR
            img_o = np.array(pd.read_csv(img_path, header=None))
            if len(img_o.shape) != 3:
                img_o = np.repeat(img_o[..., None], 3, axis=-1)  # BGR
            assert img_o is not None, "Image Not Found " + img_path

            img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
            img_print = (img / (np.max(np.abs(img))) * 255).astype(np.uint8)
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # scale (0, 255) to (0, 1)
            img = img.unsqueeze(0)  # add batch dimension

            t1 = torch_utils.time_synchronized()
            pred = model(img)[0]  # only get inference result
            t2 = torch_utils.time_synchronized()
            # print(t2 - t1)

            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
            t3 = time.time()
            process_time += (t3 - t1)
            # print(t3 - t2)
            # print(t3 - t1)

            if pred is None:
                print("No target detected.")
                exit(0)

            # process detections
            pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
            # print(pred.shape)

            bboxes = pred[:, :4].detach().cpu().numpy()
            scores = pred[:, 4].detach().cpu().numpy()
            classes = pred[:, 5].detach().cpu().numpy().astype(int) + 1

            pil_img = Image.fromarray(img_print[:, :, ::-1])
            plot_img = draw_objs(pil_img,
                                 bboxes,
                                 classes,
                                 scores,
                                 category_index=category_index,
                                 box_thresh=0.2,
                                 line_thickness=3,
                                 font='arial.ttf',
                                 font_size=20)
            with open(os.path.join(save_path, file_name.split(".")[0]) + ".txt", 'a') as f:
                info = np.concatenate([np.reshape(classes, (-1, 1)), np.reshape(scores, (-1, 1)), bboxes], axis=-1)
                info = [info[i, :].tolist() for i in range(info.shape[0])]
                for info_str in info:
                    f.write(str(info_str) + "\n")
            # plt.imshow(plot_img)
            # plt.show()
            # 保存预测的图片结果
            plot_img.save(os.path.join(save_path, file_name.split(".")[0]) + ".jpg")
        print("Average process per image: ", process_time / len(os.listdir(img_list_dir)))


if __name__ == "__main__":
    main()
