<img width="782" alt="hi-UNI" src="https://github.com/user-attachments/assets/1b305393-f7ae-431d-b05f-c2aa3e39564c">

# hi-UNI 

> hierarchical UNI for whole slide image classification. Using weakly supervised pipeline.


### Installation

Install the dependencies 

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Prepare the data in the following structure, png or jpeg format is supported. Note that extract patches only from the tumor region is recommended.

```markdown
├── data
│   ├── slide_1
│   │   ├── patch_1.png
│   │   ├── patch_2.png
│   │   ├── ...
│   ├── slide_2
│   │   ├── patch_1.png
│   │   ├── patch_2.png
│   │   ├── ...
│   ├── ...
│   └── slide_n
│       ├── ...
│       └── patch_n.png
```

It is also recommended to extract raw patches to at least 1024x1024 resolution, use [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) or [DeepZoom](https://github.com/ncoudray/DeepPATH/blob/master/DeepPATH_code/00_preprocessing/0b_tileLoop_deepzoom6.py) for patch extraction.


2. Create a hierarchical structure for the data.

    ```bash
    python utils/create_hi_patches.py --input <INPUT_DIR> --output <OUTPUT_DIR> --how non-blank
    ```
    
    `--how` : **center** (center-crop) or **non-blank** (selective-sampling, proposed in the paper)

3. Organize your data like `example.csv`. Create k-fold split for the data.

    ```bash
    python utils/gen_kfold_split.py --csv <CSV_PATH>  --dir <STEP_2_OUTPUT_DIR> --k 5 --on slide
    ```
    
    `--on slide` split the data on slide level
    
    `--on patient` split the data on patient level (use name column)
   
   A directory named `kf` will be created in the current directory.

4. Apply for the UNI model from <a href="https://huggingface.co/MahmoodLab/UNI"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000"/></a> and download the `pytorch_model.bin`.

5. Modify the [config.yaml](config.yaml) file to set hyperparameters and UNI's storage path.

    - Hyperparameters: **batch_size**, **lr**, **epochs**, **iters_to_val**, **save_best**
    
    - UNI config: **freeze_ratio** (for ViT blocks), **cmb** (hi-UNI combinations), **UNI_path** 
    
    - Task-specific config: **class_names**

### Train and evaluate

1. Train & evaluate a single fold (e.g., fold 1) and evaluate on the validation set
    ```bash
    python train.py --fold 1
    ```

2. Train & evaluate all folds (for Windows)
    ```bash
    python ./scripts/train_kf.py
    ```
3. Train & evaluate all folds (for Linux)
    ```bash
    sh ./scripts/train_kf.sh
    ```

4. The results will be saved in the `runs/}` directory.

   In the format of:
   ```markdown
    ├── runs
    │   ├── {cmbs}_{freeze_ration}  # configuration
    │   │   ├── 1
    │   │   │   ├── {fold}_best.pth  # best model
    │   │   │   ├── slide_{iter}.png  # slide-level ROC
    │   │   │   ├── ...
    │   │   ├── ...
    │   ...
   ```
   

### Comparison experiments

We are grateful to the authors for sharing their code. We use CLAM for data preprocessing and feature extraction in comparison experiments.

- CLAM (Lu et al.) [https://github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM)
- DTFD-MIL (Zhang et al.) [https://github.com/hrzhang1123/DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL)
- SETMIL (Zhao et al.) [https://github.com/Louis-YuZhao/SETMIL](https://github.com/Louis-YuZhao/SETMIL)
- TransMIL (Shao et al.) [https://github.com/szc19990412/TransMIL](https://github.com/szc19990412/TransMIL)
- im4MEC (Fremind et al.) [https://github.com/AIRMEC/im4MEC](https://github.com/AIRMEC/im4MEC)

