
# _BridgeQA: Bridging the Gap between 2D and 3D Visual Question Answering: A Fusion Approach for 3D VQA_ - Official Codebase

This work is accepeted by AAAI 2024.

 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bridging-the-gap-between-2d-and-3d-visual/3d-question-answering-3d-qa-on-scanqa-test-w)](https://paperswithcode.com/sota/3d-question-answering-3d-qa-on-scanqa-test-w?p=bridging-the-gap-between-2d-and-3d-visual)

## Installation
Please follow the procedure in [INSTALLATION](docs/installation.md).

## Data Preparation
Please follow the same procedure from [DATASET](docs/dataset.md). Also refer to ScanQA and ScanRefer. 


## Results
### ScanQA
We listed the performance on two test splits (test w/ obj / test w/o obj)
|        Method                              |     EM@1    |     B-1     |     B-4     |      R      |      M      |      C      |
|:------------------------------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|       Reported                             | 31.29/30.82 | 34.49/34.41 | 24.06/17.74 | 43.26/41.18 | 16.51/15.60 | 83.75/79.34 |
| This Implementation (Declaration re-generated with `gpt-3.5-1106`) | 30.73/30.41 | 33.70/33.90 | 20.96/17.87 | 42.46/40.79 | 16.11/15.43 | 81.75/78.16 |
| This Implementation (Using fixed declaration from `gpt-3.5-0301`) | 31.31/31.31 | 34.09/33.90 | 24.94/17.93 | 43.15/41.73 | 16.40/15.85 | 83.38/80.22 |

### SQA
|        Method       |  Acc  |
|:-------------------:|:-----:|
|       Reported      | 52.91 |
| This Implementation |  🚧   |

## Training
All model outputs and checkpoints will be saved under `./outputs/` path. You can find the checkpoints and logs of each run after training.
We also provide pretrained or pre-converted files [HERE](#checkpoints-and-pre-converted-files)

### Quesiton-Conditional View Selection
#### Question-Declaration Transform
To transform question to corresponding declaration, run following command:
```shell
export OPENAI_API_KEY = <your-openai-key>
python compose_decl_from_qa.py --output_qa_file <path/to/decl_file>
```
The output JSON file of declarations is saved as specfied in `output_qa_file` option. 
Replication note: since OpenAI will deprecate its older version GPT-4 of `gpt-3.5-0301`, and the randomness of nucleus sampling, you might not be able to acquire the same result declaration as ours. You can refer to our [pre-converted file](https://drive.google.com/file/d/10bqVuPE7bsUHh-HH8n52UXN0v0JFy7yx/view?usp=sharing) that is used in our reported performance.

#### View Selection
To select views for questions, run following command:
```shell
python eval_scene_best_views.py \
    --outfile <path/to/result>  --topk_images 1 \
    --dset_views_path <path/to/views_folder> --nocheck_blank --split "train,val,test_w_obj,test_wo_obj"  \
    --use_composed_qa --composed_qa_json <path/to/composed_decl> \
```
The result `.pkl` file (written to specified `outfile` option) is used for later training procedure as the `--i2tfile` parameter. 
You can also use the original question to find the best view at inference time.
For SQA, replace the `split` option to "train,val,test".

### Pretraining Detector
To pretrain a VoteNet detector, simply run following command without 2D VLM and QA-related losses:
```shell
export PORT=$(shuf -i 2000-3000 -n 1)
export SLURM_GPUS=4
torchrun --nproc_per_node=$SLURM_GPUS --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
    scripts/train.py --ddp --use_color --tag detection --optim_name adamw --image_size 512 \
    --use_multiview \
    --dset_views_path <path/to/views_folder> \
    --i2tfile <path/to/i2tfile> \
    --train_batch_size 16 --val_batch_size 64 --lr_blip "5e-5" --wd_blip "0.0" --lr "5e-4" --lr_decay_rate 0.2 \
    --val_step 200 --scene_feature_type full \
    --lr_decay_step 15 35 --val_step 200 --scheduler_type step --lr_blip 5e-5 --wd_blip 0.0 --lr 5e-4 \
    --stage "DET" --cur_criterion "loss" --no_reference
```
`dset_views_path` should be set to the dataset frames path where you download in [Data Preparation](#data-preparation).
`i2tfile` should be set to the `.pkl` file you obtained at the [View Selection](#view-selection) step.
The training checkpoints, logs and configs will be saved in a new directory with training date-time info under `./outputs`.
We simply take the last checkpoint `model_last.pth` as the choice for later VQA training.

NOTE: wandb is used to track the training statistics, if you want to disable it, simply set `WANDB_MODE=disabled`

### Training
To train the VQA model, simply run following command:
```shell
export PORT=$(shuf -i 2000-3000 -n 1)
export SLURM_GPUS=8
torchrun --nproc_per_node=$SLURM_GPUS --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
    scripts/train.py --ddp --use_color --tag allanswer --optim_name adamw --val_step 500 --image_size 512 --scheduler_type step \
    --use_multiview --use_blip \
    --dset_views_path <path/to/views_folder> \
    --train_batch_size 16 --val_batch_size 2 --lr_blip "1e-5" --wd_blip "0.0" --lr "5e-4"  \
    --val_step 200 --scene_feature_type full \
    --stage VQA --answer_loss_weight 3.0 \
    --i2tfile <path/to/i2tfile> \
    --first_stage_ckpt_path <path/to/detector_ckpt> \
    --use_text_decoder --share_decoder \
    --scene_feature_position paralleltwin --lr_blip3d "3e-5" --scheduler_type step_except_2d \
    --epoch 10 --lr_decay_step 5 8 --lr_decay_step_2d 3 5 7
```
`dset_views_path` should be set to the dataset frames path where you download in [Data Preparation](#data-preparation).
`i2tfile` should be set to the `.pkl` file you obtained at the [View Selection](#view-selection) step.
`first_stage_ckpt_path` should be set to the model result folder under `./outputs` at detector pretrain stage.
The training checkpoints, logs and configs will be saved in a new directory with training date-time info under `./outputs`.

### Inference
To inference and make a prediction, simply run following command:
```shell
export PORT=$(shuf -i 2000-3000 -n 1)
torchrun --nproc_per_node=$SLURM_GPUS --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
    scripts/predict.py \
    --folder <path/to/traininin_output> \
    --i2tfile <path/to/i2tfile> \
    --test_type <split-to-test> --batch_size 2 \
```
and the prediction can be found at the same folder as the training output. 
We use the best checkpoint (the `model.pth`) verified on validation set to predict the results on test splits.

## Checkpoints and Pre-converted files
We also provide the model checkpoint (pretrained detector and VQA) and other pre-computed files (question-view correspondece, declaration from quetion) here.
|         Checkpoint/Mapping         |                                        Pretrained File                                        |
|:----------------------------------:|:---------------------------------------------------------------------------------------------:|
|         Pretrained VoteNet         | [Link](https://drive.google.com/file/d/134r4TUTKFz0M8J-a6MB4SP9KS689tnFx/view?usp=drive_link) |
| Declaration from Question (ScanQA) | [Link](https://drive.google.com/file/d/18qKP-2YkDH8oYFcHyO9xX9v7j_4V8u9M/view?usp=drive_link) |
|   Question-View Mapping (ScanQA)   | [Link](https://drive.google.com/file/d/18lHk2eTwL8urK5xjZhDTjA-THBOQR06M/view?usp=drive_link) |
|          BridgeQA (ScanQA)         | [Link](https://drive.google.com/file/d/1qaYi24XpKHS-mVGKjAmgg9j9TR_xf3DG/view?usp=drive_link) |
|   Declaration from Question (SQA)  |                                            [🚧]()                                           |
|     Question-View Mapping (SQA)    |                                            [🚧]()                                           |
|           BridgeQA (SQA)           |                                            [🚧]()                                           |
The converted declaration file is generated by `gpt-3.5-0301`.

## TODO
- [x] Make copy of BLIP codes
- [x] Clean-up model codes
- [x] Clean-up training codes
- [x] Test training
- [x] Clean-up prediction codes
- [x] Test prediction
- [x] Clean-up q2d codes
- [x] Test q2d codes
- [x] Clean-up image-question selection codes
- [x] Test image-question selection codes
- [x] Clean-up detector pre-training.
- [x] Test detector pre-training.
- [x] Clean-up dependencies.
- [x] Report performance with this cleaned implementation
- [x] Update view-selection, training instructions
- [x] Update evaluation instructions
- [x] Update q2d instructions
- [x] Upload pretrained checkpoints and i2t mappings for ScanQA
- [ ] Add and combine SQA3D training codes
- [ ] Upload pretrained checkpoints and i2t mappings for SQA

## Acknowledgements
We would like to thank [facebookresearch/votenet](https://github.com/facebookresearch/votenet) for the 3D object detection, [daveredrum/ScanRefer](https://github.com/daveredrum/ScanRefer) for the 3D localization codebase and [ScanQA](https://github.com/ATR-DBI/ScanQA/) for 3D question answering codebase.
We also thank [BLIP](https://github.com/salesforce/BLIP/) for the 2D Vision-Language Model and architecture codebase.

## License
BridgeQA is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) Wentao Mo, 2024.
