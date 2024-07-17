# BioMedLM

Code used for pre-training and fine-tuning the [BioMedLM](https://huggingface.co/stanford-crfm/pubmedgpt) model.

Note: This model was previously known as PubMedGPT, but the NIH has asked us to change the name since they hold the trademark on "PubMed", so the new name is BioMedLM!

### Links

[Blog](https://crfm.stanford.edu/2022/12/15/pubmedgpt.html)

[Model](https://huggingface.co/stanford-crfm/pubmedgpt/tree/main)

[MosaicML Composer](https://github.com/mosaicml/composer)

### Example Usage

```
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda")

tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM")

model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM").to(device)

input_ids = tokenizer.encode(
    "Photosynthesis is ", return_tensors="pt"
).to(device)

sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=50)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```


## Finetuning for SeqCls 

The same info is in README.md in ChlorineProject\finetune\seqcls\README.md

1.) Download the json files from https://drive.google.com/drive/folders/19muqXjjyl6xMUy6afepaWV07dPcPmIMY,  there should be 4 files:

    > train.json ---> train dataset for seqcls save in path: ChlorineProject\pubmedqa_hf\train.json
    > dev.json -----> validation dataset for seqcls save in path: ChlorineProject\pubmedqa_hf\dev.json
    > test.json ----> test dataset for seqcls save in path: ChlorineProject\pubmedqa_hf\test.json
    > config.json --> config file for training save in path: ChlorineProject\finetune\seqcls\config.json

2.) Run "finetuning_for_seqcls.ipynb" in ChlorineProject\notebooks. This notebook install all the requierements and finetunes the model. The model with new weights will be saved in ClorineProject\runs. By running the last cell of the notebook a csv files with predictions for the test set will be saved in ChlorineProject.


The finetuning parameters were settled according https://crfm.stanford.edu/2022/12/15/biomedlm.html, this process should yield similar results. After finetuning this model should be used to zero-shot classify the abstracts between Relevant and Irrelevant