## Finetuning for SeqCls 

1.) Download the json files from https://drive.google.com/drive/folders/19muqXjjyl6xMUy6afepaWV07dPcPmIMY,  there should be 4 files:

    > train.json ---> train dataset for seqcls save in path: ChlorineProject\pubmedqa_hf\train.json
    > dev.json -----> validation dataset for seqcls save in path: ChlorineProject\pubmedqa_hf\dev.json
    > test.json ----> test dataset for seqcls save in path: ChlorineProject\pubmedqa_hf\test.json
    > config.json --> config file for training save in path: ChlorineProject\finetune\seqcls\config.json

2.) Run "finetuning_for_seqcls.ipynb" in ChlorineProject\notebooks. This notebook install all the requierements and finetunes the model. The model with new weights will be saved in ClorineProject\runs. By running the last cell of the notebook a csv files with predictions for the test set will be saved in ChlorineProject.
