import os
from openai import OpenAI
from os import getenv
import time
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from argparse import ArgumentParser
import datasets
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

DSET_PATH_SCANQA = {
    "test_w_obj": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_test_w_obj.json"),
    "test_wo_obj": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_test_wo_obj.json"),
    "train": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_train.json"),
    "val": os.path.join(DATA_DIR, "qa/ScanQA_v1.0_val.json"),
}
DSET_PATH_SQA = {
    "test": os.path.join(DATA_DIR, "qa/SQA_test.json"),
    "train": os.path.join(DATA_DIR, "qa/SQA_train_scanqa.json"),
    "val": os.path.join(DATA_DIR, "qa/SQA_val.json"),
}

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="scanqa", choices=["scanqa", "sqa"])
parser.add_argument("--output_qa_file", type=str, default="composed_decl_scanqa_qonly_reimpl.json")
parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106")

args = parser.parse_args()

if args.dataset == "scanqa":
    SCANQA_ANNO = DSET_PATH_SCANQA
elif args.dataset == "sqa":
    SCANQA_ANNO = DSET_PATH_SQA
else:
    raise NotImplementedError(f"Dataset {args.dataset} not implemented")

# gets API Key from environment variable OPENAI_API_KEY
# use openrouter.ai API if we want to use mixtral
client = OpenAI(
  base_url="https://openrouter.ai/api/v1" if "mistralai" in args.model else "https://api.openai.com/v1",
  api_key=os.getenv("OPENAI_API_KEY"),
)

TEMPLATE="""Turn following question into an declarative sentence like an image caption. Generate natural and fluent sentence consistent to the question. Replace the unknown answer with appropriate indefinite pronoun (e.g., something, some color, somewhere). The given question corresponds to a indoor scene not given. DO NOT include extra output.
---
Question: {q}"""

def robust_query(max_retry=100, **query_kwargs):
    for r in range(max_retry):
        try:
            response = client.chat.completions.create(
                **query_kwargs
                )
            return response
        except Exception as e:
            print(e)
            print(f"Retrying...{r+1}/{max_retry}")
            time.sleep(1)
            continue
    raise Exception(f"Reached {max_retry} times retry, aborting...")

def compose_declaration(q, a, model, max_tries=100, include_system=True):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. "},
        {"role": "user", "content": TEMPLATE.format(q=q,a=a)},
    ]
    if not include_system:
        messages = messages[1:]
    completion = robust_query(
        max_retry=max_tries,
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    completion = completion.choices[0].message.content.strip()
    completion = completion.replace("Answer:", "").strip()
    completion = re.sub(r"\s+", " ", completion)
    completion = completion.strip()
    return completion

# print(compose_declaration("What is the color of the sofa?", "red"))
# print(compose_declaration("What is the position of the red sofa in front of the larger red sofa?", "against wall beneath picture"))

dset = {}

if os.path.exists(args.output_qa_file):
    COMPOSE_RESULTS = json.load(open(args.output_qa_file, "r"))
else:
    COMPOSE_RESULTS = {}

for split, filename in SCANQA_ANNO.items():
    dset[split] = json.load(open(filename, "r"))
    tmp = {
        "question": [],
        "question_id": [],
        # "situation": [],
    }
    if split in ["train", "val"]:
        tmp["answers"] = []
    for item in dset[split]:
        for key in tmp.keys():
            tmp[key].append(item[key])
    
    dset[split] = datasets.Dataset.from_dict(tmp)

print(dset)

def run(dset):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {sample["question_id"]: executor.submit(compose_declaration, 
                                                            sample["question"], 
                                                            sample["answers"][0] if "answers" in sample else "",
                                                            args.model,
                                                            100,
                                                            "mistralai" in args.model,
                                                          ) for sample in dset}
        for qid, future in tqdm(futures.items(), total=len(futures)):
            COMPOSE_RESULTS[qid] = future.result().strip()
            # save every 100
            if len(COMPOSE_RESULTS) % 20 == 0:
                print(f"Saving {len(COMPOSE_RESULTS)} results")
                with open(args.output_qa_file, "w") as f:
                    json.dump(COMPOSE_RESULTS, f)

for split in dset.keys():
    run(dset[split])
    with open(args.output_qa_file, "w") as f:
        json.dump(COMPOSE_RESULTS, f)

