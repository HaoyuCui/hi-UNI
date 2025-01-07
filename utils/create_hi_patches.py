import argparse
import os
import numpy as np
from PIL import Image

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Create hierarchical patches")
    parser.add_argument('--input', type=str, help='Original patch directory, recommend res: 1024 x 1024')
    parser.add_argument('--output', type=str, help='Output directory')
    # non-blank (selective sampling): exclude almost white or black patches
    parser.add_argument('--how', type=str, choices=['center', 'non-blank'],
                        help='How to extract P_s patches, non-blank (proposed, selective sampling) or center (default)')
    args = parser.parse_args()
    return args


def tile_is_not_empty(_img, threshold_white=30):
    histogram = _img.histogram()
    whiteness_check = [0, 0, 0]
    for channel_id in (0, 1, 2):
        whiteness_check[channel_id] = np.median(
            histogram[256 * channel_id: 256 * (channel_id + 1)][100:200]
        )
    if all(c <= threshold_white for c in whiteness_check):
        return False
    return True


if __name__ == '__main__':
    args = parse_args()
    if args.input and args.output:
        input_dir = args.input
        output_dir = args.output
        input_dir_name = os.path.basename(input_dir)

        target_dir_s = os.path.join(output_dir, f'{input_dir_name}_s')
        target_dir_m = os.path.join(output_dir, f'{input_dir_name}_m')
        target_dir_l = os.path.join(output_dir, f'{input_dir_name}_l')
    else:
        raise ValueError('Specify input and output directory by --input and --output')

    total_num = len(os.listdir(input_dir))

    flag_dispose = False

    for idx, _dir in enumerate(os.listdir(input_dir)):
        os.makedirs(os.path.join(target_dir_s, _dir), exist_ok=True)
        os.makedirs(os.path.join(target_dir_m, _dir), exist_ok=True)
        os.makedirs(os.path.join(target_dir_l, _dir), exist_ok=True)
        img_num = len(os.listdir(os.path.join(input_dir, _dir)))
        print(f'{idx} / {total_num} - {img_num} images in {_dir} directory')
        for file in tqdm(os.listdir(os.path.join(input_dir, _dir))):
            if file.endswith('.jpg'):
                flag_dispose = True if args.how == 'non-blank' else False
                img = Image.open(os.path.join(input_dir, _dir, file))

                # Resize to 1024, P_l patches
                img = img.resize((1024, 1024))
                img.resize((224, 224)).save(os.path.join(target_dir_l, _dir, file))

                # Center crop: 1024 x 1024 -> 512 x 512, P_m patches
                img = img.crop((256, 256, 768, 768))
                img.resize((224, 224)).save(os.path.join(target_dir_m, _dir, file))

                if args.how == 'non-blank':  # Selective crop: exclude almost white or black patches
                    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        candidate = img.crop((i * 256, j * 256, i * 256 + 256, j * 256 + 256))

                        # Keep the first patch that is not almost white or black
                        if tile_is_not_empty(candidate):
                            candidate.resize((224, 224)).save(os.path.join(target_dir_s, _dir, file))
                            flag_dispose = False
                            break

                    if flag_dispose:
                        # Delete previously saved files
                        os.remove(os.path.join(target_dir_l, _dir, file))
                        os.remove(os.path.join(target_dir_m, _dir, file))

                elif args.how == 'center':  # Center crop
                    # Center crop: 512 x 512 -> 256 x 256 -> 224 x 224
                    img = img.crop((128, 128, 384, 384))
                    img.resize((224, 224)).save(os.path.join(target_dir_s, _dir, file))


