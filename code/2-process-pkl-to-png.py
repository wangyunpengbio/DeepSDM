import argparse
import pickle
from tqdm import tqdm
from pathlib import Path

import cv2

import numpy as np
import pandas as pd
from collections import defaultdict

from utils.helpers import load_yaml

import importlib
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

from utils.mask_functions import mask2rle

## 对推断后生成的pkl文件进行处理，生成对应的png文件（生成文件路径由yaml参数文件中的RESULT_PNG_DIC来指定）
## 使用示例：python 2-process-pkl-to-png.py -config_file experiments/demo/process.yaml
## 具体参数放在yaml文件中
def argparser():
    parser = argparse.ArgumentParser(description='Pneumothorax pipeline')
    parser.add_argument('-config_file', default='experiments/demo/process.yaml', type=str, nargs='?', help='process config file path')
    return parser.parse_args()

def extract_largest(mask, n_objects):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(c) for c in contours]
    contours = np.array(contours)[np.argsort(areas)[::-1]] # 计算面积后，选取最大的
    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours[:n_objects],
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def remove_smallest(mask, min_contour_area):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours,
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def apply_thresholds(predicted_single, binarizer_class, binarizer_threshold, only_largest, min_contour_area, opening):
    # 下面这行要重新用=赋值以后才起作用
    mask = binarizer_class.apply_transform_numpy_perimage(threshold = binarizer_threshold, **predicted_single)
    
    if min_contour_area > 0:# 默认不使用该约束。下面三个依次是去掉面积小于指定面积的轮廓，选取最大的轮廓，和不进行任何改动
        choosen = remove_smallest(mask, min_contour_area)
    elif only_largest: # 默认不使用该约束。
        choosen = extract_largest(mask, n_objects)
    else: # 默认为不进行任何改动
        choosen = mask * 255
    if opening:
        choosen = opening_opt(choosen)
    if mask.shape[0] == 1024:
        reshaped_mask = choosen
    else:
        reshaped_mask = cv2.resize(
            choosen,
            dsize=(1024, 1024),
            interpolation=cv2.INTER_LINEAR
        )
    return reshaped_mask

def opening_opt(img):
    img = np.array(img,dtype = np.uint8)
    img = img.reshape(1024,1024,1)
    kernel = np.ones((10, 10), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def mergeImgAndMask(mask_applied, name, result_dic):
    img_path = result_dic / '..' / 'test' / "{}.png".format(name)
    # img and mask prepare
    img = cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE)
    mask_applied = np.uint8(mask_applied)
    img_rbg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    mask_applied_rgb = cv2.cvtColor(mask_applied,cv2.COLOR_GRAY2BGR)
    # 修改mask的颜色
    lower_white = np.array([200,200,200])
    upper_white = np.array([255,255,255])
    mask_white = cv2.inRange(mask_applied_rgb,lower_white,upper_white)
    color_green=[0,255,255] # 修改为绿色
    mask_applied_rgb[mask_white!=0]=color_green
    # merge img and mask 
    rate_for_image = 0.7
    result = cv2.addWeighted(img_rbg, rate_for_image, mask_applied_rgb, 1 - rate_for_image, 0)
    return img, mask_applied, result

def build_result_png(infer_pkl_dict, result_dic, binarizer_class, binarizer_threshold, only_largest=False, min_contour_area=False, opening=True):
    rle_dict = {}
    
    output_mask_dict = infer_pkl_dict["mask"]
    output_distancemap_dict = infer_pkl_dict["distancemap"]
    distancemap_exist = True if len(output_distancemap_dict.keys()) != 0 else False # 检测是否有distance map
    
    for image_single_id, mask_predict in tqdm(output_mask_dict.items()):
        predicted_single = {"mask_predict":output_mask_dict[image_single_id]}
        if distancemap_exist:
            predicted_single["distancemap_predict"] = output_distancemap_dict[image_single_id]
        
        # 直接调用上面，应用阈值得到结果
        mask_applied = apply_thresholds(predicted_single, binarizer_class, binarizer_threshold, only_largest, min_contour_area, opening)
        rle_dict[image_single_id] = mask2rle(mask_applied.T, 1024, 1024)
        img, mask_applied, result = mergeImgAndMask(mask_applied, image_single_id, result_dic)
        # 保存图片
        img_PIL = Image.fromarray(img)
        mask_applied_PIL = Image.fromarray(mask_applied)
        result_PIL = Image.fromarray(result)
        
        img_PIL_rotate = img_PIL.transpose(Image.ROTATE_270)
        img_PIL_rotate = img_PIL_rotate.transpose(Image.FLIP_TOP_BOTTOM)
        mask_applied_PIL_rotate = mask_applied_PIL.transpose(Image.ROTATE_270)
        mask_applied_PIL_rotate = mask_applied_PIL_rotate.transpose(Image.FLIP_TOP_BOTTOM)
        
        img_nii = nib.Nifti1Image(np.array(img_PIL_rotate), np.eye(4))
        mask_applied_nii = nib.Nifti1Image(np.array(mask_applied_PIL_rotate)/255, np.eye(4))
        
        img_PIL.save(result_dic / "{}.1.img.png".format(image_single_id))
        mask_applied_PIL.save(result_dic / "{}.2.mask.png".format(image_single_id))
        result_PIL.save(result_dic / "{}.3.merge.png".format(image_single_id))
        
        img_nii.to_filename(result_dic / "{}.4.img.nii.gz".format(image_single_id))
        mask_applied_nii.to_filename(result_dic / "{}.5.mask.nii.gz".format(image_single_id))
    return rle_dict

def buid_submission(rle_dict):
    sub = pd.DataFrame.from_dict([rle_dict]).T.reset_index()
    sub.columns = ['ImageId', 'EncodedPixels'] # 设置列名
    sub.loc[sub.EncodedPixels == '', 'EncodedPixels'] = -1
    return sub
    
def main():
    args = argparser()
    config_file = args.config_file
    process_config = load_yaml(config_file)

    print('start loading mask results....')
    inference_pkl_path = process_config['INFERENCE_PKL_FILE']
    with open(inference_pkl_path, 'rb') as handle:
        infer_pkl_dict = pickle.load(handle)

    only_largest = process_config.get('ONLY_LARGEST', False) # 为False ，默认不使用该约束
    min_contour_area = process_config.get('MIN_CONTOUR_AREA', 0) # 为0 ，默认不使用该约束
    opening = process_config.get('OPENING', True) # 为True ，默认使用开操作

    binarizer_module = importlib.import_module(process_config['MASK_BINARIZER']['PY'])
    binarizer_class = getattr(binarizer_module, process_config['MASK_BINARIZER']['CLASS'])
    if process_config['MASK_BINARIZER'].get('THRESHOLD', False):
        binarizer_threshold = process_config['MASK_BINARIZER']['THRESHOLD']
        binarizer_class =  binarizer_class('Inference Time MaskBinarization') # 仅仅是初始化需要个参数，因为需要调用类中的apply transform
    else:
        print("Please set the THRESHOLD of MASK_BINARIZER ")

    result_dic = Path(process_config['RESULT_PNG_DIC'])
    if os.path.isdir(result_dic):
        shutil.rmtree(result_dic)
    os.makedirs(result_dic, exist_ok=True)
    rle_dict = build_result_png(infer_pkl_dict, result_dic, binarizer_class, binarizer_threshold, only_largest, min_contour_area, opening)
    
    kaggle_test = process_config.get('KAGGLE_TEST', False)
    if kaggle_test:
        sub = buid_submission(rle_dict)
        sub.to_csv('submit.csv', index=False)
    
if __name__ == "__main__":
    main()
