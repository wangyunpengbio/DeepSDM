import pandas as pd
import argparse
import os
from glob import glob
from tqdm import tqdm

def argparser():
    parser = argparse.ArgumentParser(description='Split the dataset into folds')
    parser.add_argument('-train_path', default='../../input/1-stage-dcm-all/', type=str, nargs='?', help='directory with train')
    parser.add_argument('-rle_path', default='../../input/stage_2_train.csv', type=str, nargs='?', help='path for rle csv file')
    parser.add_argument('-fold_num', default=5, type=int, nargs='?', help='number of folds')
    parser.add_argument('-out_path', default='../folds', type=str, nargs='?', help='path for saving meta csv')
    parser.add_argument('-n_threads', default=4, type=int, nargs='?', help='number of using threads')
    return parser.parse_args()

def get_mask(encode):
    if encode == [] or encode == ['-1'] or encode == [' -1']:
        return 0
    else:
        return len(encode)
    
def main():
    args = argparser()
    rle = pd.read_csv(args.rle_path)
    train_fns = sorted(glob('{}/*/*/*.dcm'.format(args.train_path)))
    print('Read %s (%d rle records)' % (args.rle_path, len(rle)))
    print('Dicom directory %s (%d dicom files in total)' % (args.train_path, len(train_fns)))
    dic_out = {}
    for num,path in tqdm(enumerate(train_fns)):
        dicomFileName = path.split("/")[-1]
        fileID = dicomFileName.strip(".dcm")
        encode = list(rle.loc[rle.ImageId == fileID,"EncodedPixels"].values)
        pne_num = get_mask(encode)
        dic_out[num] = [fileID,num % args.fold_num, pne_num]
    dataframe_out = pd.DataFrame.from_dict(dic_out,orient='index')
    dataframe_out.columns = ["fileID","fold","pne_num"]
    dataframe_out.fold = dataframe_out.fold.sample(frac=1,random_state=666).values # 随机排列
    os.makedirs(args.out_path, exist_ok=True)
    dataframe_out.to_csv(os.path.join(args.out_path,"train_folds_"+str(args.fold_num)+".csv"),index = False)
    
if __name__ == '__main__':
    main()