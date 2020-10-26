import argparse
from tqdm import tqdm
import os
from os.path import join
import importlib
from pathlib import Path
import pickle
from glob import glob

import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader
import torch

from dataset.dataset import generate_transforms, PneumothoraxDataset
from utils.helpers import load_yaml, init_seed, init_logger

## 对数据集进行推断，并保存成pkl文件（示例中生成文件名为inference.pkl）
## 使用示例：python 1-inference-save-pkl.py -config_file experiments/demo/inference.yaml
## 具体参数放在yaml文件中
def build_checkpoints_list(cfg, submit_best, fold_id):
    pipeline_path = cfg['MODEL']['PRETRAINED']['PIPELINE_PATH'] # 由config设置
    pipeline_name = cfg['MODEL']['PRETRAINED']['PIPELINE_NAME'] # 由config设置

    checkpoints_list = []
    if submit_best:# 如果是SUBMIT_BEST，就选择checkpoint目录下面的多折结果，就是每折只选一个权重文件".pth"
        best_checkpoints_folder = Path(
            pipeline_path, 
            cfg['CHECKPOINTS']['BEST_FOLDER']
        )
        filename = '{}_fold{}.pth'.format(pipeline_name, fold_id)
        checkpoints_list.append(Path(best_checkpoints_folder, filename))
    else:# 选中多折和对应的epoch(对应的epoch通过glob获得)
        checkpoints_list = glob(str(Path(pipeline_path,cfg['CHECKPOINTS']['FULL_FOLDER'],"fold"+fold_id,"*pth")))
        checkpoints_list = [Path(checkpoint) for checkpoint in checkpoints_list] # 再从str转回Path对象的类型
    return checkpoints_list
    
def inference_image(model, images, device):# 计算单个batch图像的mask
    images = images.to(device)
    predicted = model(images)
    masks_predict = predicted # 先默认是下述的D类型
    distancemap_exist = False
    if isinstance(predicted,tuple): # 是否有多个output。A:(masks, distancemap, labels)；B:(masks, distancemap)； C:(masks, labels)；D:masks
        masks_predict = predicted[0]
        if len(predicted[1].shape) == 4: # 说明从一开始数第二个的output为distancemap
            distancemap_exist = True
            distancemap_predict = predicted[1]
            distancemap_predict = torch.tanh(distancemap_predict)
            distancemap_predict = distancemap_predict.squeeze(1).cpu().detach().numpy() # 只有维度为1时才会去掉,此时shape第2个元素，即1，维度为1,[batchsize,channel,width,height]
    masks_predict = torch.sigmoid(masks_predict)
    masks_predict = masks_predict.squeeze(1).cpu().detach().numpy() # 只有维度为1时才会去掉,此时shape第2个元素，即1，维度为1,[batchsize,channel,width,height]
    if distancemap_exist:
        return {"distancemap_exist":True, "masks_predict":masks_predict, "distancemap_predict":distancemap_predict}
    else:
        return {"distancemap_exist":False, "masks_predict":masks_predict}
    
def flipped_inference_image(model, images, device):
    flipped_imgs = torch.flip(images, dims=(3,))
    predicted = inference_image(model, flipped_imgs, device)
    if predicted['distancemap_exist'] == True:
        return {"distancemap_exist":True, "masks_predict":np.flip(predicted['masks_predict'], axis=2), "distancemap_predict":np.flip(predicted['distancemap_predict'], axis=2)}
    else:
        return {"distancemap_exist":False, "masks_predict":np.flip(predicted['masks_predict'], axis=2)}

def argparser():
    parser = argparse.ArgumentParser(description='Inference each model and save the output in the pkl file')
    parser.add_argument('-config_file', default='experiments/demo/inference.yaml', type=str, nargs='?', help='inference config file path')
    return parser.parse_args()

def main():
    args = argparser()
    config_file = args.config_file
    inference_config = load_yaml(config_file)
    print(inference_config)

    batch_size = inference_config['BATCH_SIZE']
    device = inference_config.get('DEVICE',"cuda") # DEVICE默认是cuda
    
    if "DEVICE_LIST" in inference_config:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, inference_config["DEVICE_LIST"]))

    module = importlib.import_module(inference_config['MODEL']['PY'])
    model_class = getattr(module, inference_config['MODEL']['CLASS'])
    model = model_class(**inference_config['MODEL'].get('ARGS', None)).to(device)
    model.eval()

    usefolds = map(str, inference_config['FOLD']['USEFOLDS'])

    num_workers = inference_config['WORKERS']

    image_size = inference_config.get('IMAGE_SIZE',1024)
    train_transform, valid_transform = generate_transforms(image_size)

    dataset_folder = inference_config['DATA_DIRECTORY']

    dataset = PneumothoraxDataset(
        data_folder=dataset_folder, mode='test', 
        transform=valid_transform,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, 
        num_workers=num_workers, shuffle=False
    )

    use_flip = inference_config['FLIP']

    submit_best = inference_config['SUBMIT_BEST']

    checkpoints_list = []
    for fold_id in usefolds:
        checkpoints_list.extend(build_checkpoints_list(inference_config, submit_best=submit_best, fold_id=fold_id))

    output_mask_dict = defaultdict(int)
    output_distancemap_dict = defaultdict(int)
    for pred_idx, checkpoint_path in enumerate(checkpoints_list):
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        # 模型推理
        for item in tqdm(dataloader):
            image_ids, images = item
            predicted = inference_image(model, images, device) # batch级别的inference

            masks_predict = predicted["masks_predict"]
            if predicted["distancemap_exist"]: # 如果存在distance map
                distancemap_predict = predicted["distancemap_predict"]

            if use_flip: # 如果flip的话，会覆盖同名变量
                predicted_flipped = flipped_inference_image(model, images, device)
                masks_predict = (predicted["masks_predict"] + predicted_flipped["masks_predict"]) / 2
                if predicted_flipped["distancemap_exist"]: # 如果存在distance map
                    distancemap_predict = (predicted["distancemap_predict"] + predicted_flipped["distancemap_predict"]) / 2

            # 把一个batch图像拆开
            for index,(image_single_id, mask_single_predict) in enumerate(zip(image_ids, masks_predict)):
                output_mask_dict[image_single_id] = (output_mask_dict[image_single_id] * pred_idx + mask_single_predict) / (pred_idx + 1) # 将结果取平均
                if predicted["distancemap_exist"]:
                    output_distancemap_dict[image_single_id] = (output_distancemap_dict[image_single_id] * pred_idx + distancemap_predict[index]) / (pred_idx + 1) # 将结果取平均

    print('Number of mask: {}, number of distance map: {}'.format(len(output_mask_dict.keys()),len(output_distancemap_dict.keys())))

    result_dict = {"mask":output_mask_dict,"distancemap":output_distancemap_dict}

    result_path = Path(inference_config['MODEL']['PRETRAINED']['PIPELINE_PATH'], inference_config['RESULT_PKL_FILE'])
    with open(result_path, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    main()