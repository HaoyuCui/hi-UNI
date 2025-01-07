import os
import random
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def read_dir(path):
    file_list = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    return file_list


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Generate KFold split')
    parser.add_argument('--dir', type=str, help='Directory to process')
    parser.add_argument('--csv', type=str, default='example.csv', help='CSV label file to process')
    parser.add_argument('--k', '-k', type=int, default=5, help='K-fold cross validation, number of folds')
    parser.add_argument('--on', type=str, choices=['name', 'slide'], default='slide',
                        help='Split on name or slide')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for shuffling')
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    patch_dir = args.dir + '/' + [x for x in os.listdir(args.dir) if x.endswith('_s')][0]
    assert patch_dir, 'Patch directory not found'
    assert len([x for x in os.listdir(args.dir) if x.endswith('_s')]) == 1, \
        'Ambiguous patch directory, please make sure there is only one directory ends with "_s"'
    print('Patch directory:', patch_dir)

    total_split = args.k
    df = pd.read_csv(args.csv)[['name', 'slide', 'label']]

    assert df['label'].nunique() != 2, 'Only multi-label classification is supported'
    # label must be digits
    assert df['label'].dtype == 'int64', "Label col 'label' must be integer"

    os.makedirs('kf', exist_ok=True)
    skf = StratifiedKFold(n_splits=total_split, shuffle=True, random_state=args.seed).split(df[args.on], df['label'])

    for fold, (train_index, test_index) in enumerate(skf):
        fold = fold + 1
        print('creating fold {}'.format(fold))
        slide_train, slide_val = df[args.on].iloc[train_index], df[args.on].iloc[test_index]
        df_train = pd.DataFrame(columns=['name', 'tile-s', 'tile-m', 'tile-l', 'label'])
        df_val = pd.DataFrame(columns=['name', 'tile-s', 'tile-m', 'tile-l', 'label'])

        for slide in df[args.on]:
            print('processing slide {}'.format(slide))
            slides_loc = os.path.join(patch_dir, slide)
            if not os.path.exists(slides_loc):
                print('slide {} not found'.format(slide))
                continue
            slides_loc_list = read_dir(slides_loc)
            name = df[df[args.on] == slide]['name'].tolist()[0]
            label = df[df[args.on] == slide]['label'].tolist()[0]

            slides_loc_list = random.sample(slides_loc_list, len(slides_loc_list))

            slides_loc_list_s = slides_loc_list
            slides_loc_list_m = [x.replace('224_s', '224_m') for x in slides_loc_list]
            slides_loc_list_l = [x.replace('224_s', '224_l') for x in slides_loc_list]
            df_new = pd.DataFrame({'name': name,
                                   'tile-s': slides_loc_list_s, 'tile-m': slides_loc_list_m,
                                   'tile-l': slides_loc_list_l,
                                   'label': label})
            if slide in slide_train.tolist():
                df_train = pd.concat([df_train, df_new], ignore_index=True)
            elif slide in slide_val.tolist():
                df_val = pd.concat([df_val, df_new], ignore_index=True)
            else:
                print('[warning] slide {} not found'.format(slide))

        train_patients = len(df_train['name'].unique())
        val_patients = len(df_val['name'].unique())
        print(f'Fold {fold}: train-set patients: {train_patients} val-set patients: {val_patients}')

        df_train.to_csv(f'kf/{fold}_train.csv', index=False)
        df_val.to_csv(f'kf/{fold}_val.csv', index=False)


if __name__ == '__main__':
    main()
