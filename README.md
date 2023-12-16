
# Official codebase for _BridgeQA: Bridging the Gap between 2D and 3D Visual Question Answering: A Fusion Approach for 3D VQA_.



## Installation
Please follow the procedure in [INSTALLATION](docs/installation.md).

## Data Preparation
Please follow the same procedure from [DATASET](docs/dataset.md). Also refer to ScanQA and ScanRefer. 


## Results
### ScanQA
We listed the performance on two test splits (test w/ obj / test w/o obj)
|        Method       |     EM@1    |     B-1     |     B-4     |      R      |      M      |      C      |
|:-------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|       Reported      | 31.29/30.82 | 34.49/34.41 | 24.06/17.74 | 43.26/41.18 | 16.51/15.60 | 83.75/79.34 |
| This Implementation |             |             |             |             |             |             |
### SQA
|        Method       |  Acc  |
|:-------------------:|:-----:|
|       Reported      | 52.91 |
| This Implementation |       |

## Training
All model outputs and checkpoints will be saved under `./outputs/` path. You can find the checkpoints and logs of each run after training.
We also provide pretrained or pre-converted files [HERE](#checkpoints-and-pre-converted-files)

### Quesiton-Conditional View Selection
#### Question-Declaration Transform
To transform question to corresponding declaration, run following command:
```shell
```
Replication note: since OpenAI has deprecated its older version of `gpt-3.5-0xxx`, and the randomness of nucleus sampling, you might not be able to acquire the same result as ours. You can refer to our [result](#checkpoints-and-pre-converted-files), or try a newer GPT model, or try out powerful open-source LLM alternatives like [Mistral-MoE](https://mistral.ai/news/mixtral-of-experts/).

#### View Selection
To select view for a question, run following command:
```shell
stdbuf -o0 -e0 python eval_scene_best_views.py  \
    --outfile <path/to/result>  --topk_images 1 \
    --answer_freq_threshold 0  --max_answer_count 3000 \
    --dset_views_path <path/to/views_folder> --nocheck_blank --split "train,val,test_w_obj,test_wo_obj"  \
    --use_composed_qa --composed_qa_json <path/to/composed_decl> \
```
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
We simply take the last checkpoint as the choice for later VQA training.

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

## Checkpoints and Pre-converted files
We also provide the model checkpoint (pretrained detector and VQA) and the pre-extracted question-view Mapping file here.
|         Checkpoint/Mapping         |                                        Pretrained File                                        |
|:----------------------------------:|:---------------------------------------------------------------------------------------------:|
|         Pretrained VoteNet         | [Link](https://drive.google.com/file/d/134r4TUTKFz0M8J-a6MB4SP9KS689tnFx/view?usp=drive_link) |
| Declaration from Question (ScanQA) |                                            [🚧]()                                           |
|   Question-View Mapping (ScanQA)   | [Link](https://drive.google.com/file/d/18lHk2eTwL8urK5xjZhDTjA-THBOQR06M/view?usp=drive_link) |
|          BridgeQA (ScanQA)         | [Link](https://drive.google.com/file/d/1qaYi24XpKHS-mVGKjAmgg9j9TR_xf3DG/view?usp=drive_link) |
|   Declaration from Question (SQA)  |                                            [🚧]()                                           |
|     Question-View Mapping (SQA)    |                                            [🚧]()                                           |
|           BridgeQA (SQA)           |                                            [🚧]()                                           |

## TODO
- [x] Make copy of BLIP codes
- [x] Clean-up model codes
- [x] Clean-up training codes
- [x] Test training
- [ ] Clean-up prediction codes
- [ ] Test prediction
- [ ] Clean-up image-question selection codes
- [ ] Test image-question selection codes
- [x] Clean-up detector pre-training.
- [x] Test detector pre-training.
- [x] Clean-up dependencies.
- [ ] Report performance with this cleaned implementation
- [ ] Update view-selection, training, evaluation instructions
- [x] Upload pretrained checkpoints and i2t mappings for ScanQA
- [ ] Add and combine SQA3D training codes
- [ ] Upload pretrained checkpoints and i2t mappings for SQA

## Acknowledgements
We would like to thank [facebookresearch/votenet](https://github.com/facebookresearch/votenet) for the 3D object detection, [daveredrum/ScanRefer](https://github.com/daveredrum/ScanRefer) for the 3D localization codebase and [ScanQA](https://github.com/ATR-DBI/ScanQA/) for 3D question answering codebase.
We also thank [BLIP](https://github.com/salesforce/BLIP/) for the 2D Vision-Language Model and architecture codebase.

## License
BridgeQA is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License](LICENSE).

Copyright (c) 2023.
