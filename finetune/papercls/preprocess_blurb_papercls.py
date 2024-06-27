import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def dump_jsonl(data, fpath):
    with open(fpath, "w", encoding='utf-8') as outf:
        for d in data:
            print(json.dumps(d, ensure_ascii=False), file=outf)

def process_chlorine(fname, sample_size=None, train_size=0.75, val_size=0.1, test_size=0.15, random_state=42):
    dname = "chlorine_safety_mockup2"
    print(dname, fname)
    print(os.getcwd())
    df = pd.read_excel(f"finetune/papercls/raw_data/{fname}.xlsx", dtype=str)
    
    # If sample_size is provided, take a random sample of the data
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=random_state)
    # Check the sizes sum to 1
    assert train_size + val_size + test_size == 1.0, "Train, validation and test sizes should sum to 1.0"
    
    # Shuffle and split the dataset
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=random_state, stratify=temp_df['label'])
    
    # Convert dataframes to lists of dicts
    def df_to_dict_list(df):
        outs = []
        label_map = {"Relevant": 1, "Irrelevant": 0}
        lens = []
        for _, row in df.iterrows():
            id         = row[0].strip()
            title      = row[1].strip()
            abstract   = row[2].strip()
            label      = row[3].strip()
            assert label in ["Relevant", "Irrelevant"]
            label = label_map[label]
            outs.append({"id": id, "title": title, "abstract": abstract, "label": label})
            lens.append(len(title) + len(abstract))
        return outs, lens
    
    train_data, train_lens = df_to_dict_list(train_df)
    val_data, val_lens = df_to_dict_list(val_df)
    test_data, test_lens = df_to_dict_list(test_df)
    
    print(f"Train: total {len(train_data)}, seqlen mean {int(np.mean(train_lens))}, median {int(np.median(train_lens))}, 95th {int(np.percentile(train_lens, 95))}, max {np.max(train_lens)}")
    print(f"Validation: total {len(val_data)}, seqlen mean {int(np.mean(val_lens))}, median {int(np.median(val_lens))}, 95th {int(np.percentile(val_lens, 95))}, max {np.max(val_lens)}")
    print(f"Test: total {len(test_data)}, seqlen mean {int(np.mean(test_lens))}, median {int(np.median(test_lens))}, 95th {int(np.percentile(test_lens, 95))}, max {np.max(test_lens)}")
    
    root = os.getcwd()
    os.makedirs(f"{root}/{dname}_hf", exist_ok=True)
    dump_jsonl(train_data, f"{root}/{dname}_hf/mockup2_train.json")
    dump_jsonl(val_data, f"{root}/{dname}_hf/mockup2_val.json")
    dump_jsonl(test_data, f"{root}/{dname}_hf/mockup2_test.json")

# Example usage
process_chlorine("CHE_files_testing",sample_size=100)

