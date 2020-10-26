import pandas as pd
import argparse
import os
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
import functools
import pydicom
import pickle

import misc

def argparser():
    parser = argparse.ArgumentParser(description='Extract the meta data from dicom files')
    parser.add_argument('-train_path', default='../../input/1-stage-dcm-all/', type=str, nargs='?', help='directory with train')
    parser.add_argument('-rle_path', default='../../input/stage_2_train.csv', type=str, nargs='?', help='path for rle csv file')
    parser.add_argument('-out_path', default='../folds', type=str, nargs='?', help='path for saving meta csv')
    parser.add_argument('-n_threads', default=4, type=int, nargs='?', help='number of using threads')
    return parser.parse_args()

def create_record(path):

    dicom = pydicom.dcmread(path)
    dicomFileName = path.split("/")[-1]
    fileID = dicomFileName.strip(".dcm")
    
    record = {
        'ID': fileID,
    }
    record.update(misc.get_dicom_raw(dicom))

    raw = dicom.pixel_array

    record.update({
        'raw_max': raw.max(),
        'raw_min': raw.min(),
        'raw_mean': raw.mean(),
        'raw_diff': raw.max() - raw.min(),
    })
    return record

def create_df(ids, args):
    print('making records...')
    with Pool(args.n_threads) as pool:
        records = list(tqdm(
            iterable=pool.map(
                functools.partial(create_record),
                ids
            ),
            total=len(ids),
        ))
    return pd.DataFrame(records).sort_values('ID').reset_index(drop=True)


def main():
    
    args = argparser()
    rle = pd.read_csv(args.rle_path)
    train_fns = sorted(glob('{}/*/*/*.dcm'.format(args.train_path)))
    print('Read %s (%d rle records)' % (args.rle_path, len(rle)))
    print('Dicom directory %s (%d dicom files in total)' % (args.train_path, len(train_fns)))
    df_output = create_df(train_fns, args)
    
    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path,"train.pkl"), 'wb') as f:
        pickle.dump(df_output, f)

    # 最后就变成一张表，放在cache目录下
    print('converted dicom to dataframe (%d records)' % len(df_output))
    print('saved to %s' % args.out_path)

if __name__ == '__main__':
    main()
