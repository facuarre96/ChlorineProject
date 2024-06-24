import os
import csv
import json
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm


def dump_jsonl(data, fpath):
    with open(fpath, "w") as outf:
        for d in data:
            print (json.dumps(d), file=outf)


######################### BLURB sequence classification #########################
root = "data"
os.system(f"mkdir -p {root}")


def process_chlorine(fname):
    dname = "chlorine_safety"
    print(dname, fname)
    file_path = f"finetune/papercls/raw_data/{fname}.txt"
    df = pd.read_csv(file_path, sep="\t", header=0,dtype=str)
    outs, lens = [], []
    for _, row in df.iterrows():
        print('file done')
        id = row[0].strip()
        title = row[1].strip()
        abstract = row[2].strip()
        label = row[3].strip()
        assert label in ["Relevant", "Irrelevant"]
        outs.append({"id": id, "sentence1": title, "sentence2": abstract, "label": label})
        lens.append(len(title) + len(abstract))
    print("total", len(outs), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))
    #
    root = os.getcwd()
    os.makedirs(f"{root}/{dname}_hf", exist_ok=True)
    dump_jsonl(outs, f"{root}/{dname}_hf/{fname}.json")

process_chlorine("test")
process_chlorine("dev")
process_chlorine("train")

