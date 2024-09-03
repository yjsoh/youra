import json
import logging
import os
import subprocess
import multiprocessing

import click
import pymongo
import tqdm

from transformers import AutoTokenizer
import datasets
import stanza
from Levenshtein import distance as lev_dist

from src import targetsentences2sentences

DS_NAME = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
]

MODELPATH2MODELNAME = {
    "meta-llama/Llama-2-7b-chat-hf": "llama2",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistralai",
}


def init_tokenizer(model_name):
    """Initialize the tokenizer in each process"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def init_stanza():
    """Initialize the Stanza NLP pipeline in each process"""
    nlp = stanza.Pipeline(
        "en",
        processors="tokenize,mwt,ner",
        download_method=None,
        logging_level="WARN",
        use_gpu=False,
        # download_method=stanza.pipeline.core.DownloadMethod.DOWNLOAD_RESOURCES,
    )
    return nlp


def get_sentences(col, ds_name, q_index):
    """Load target sentences from mongodb"""
    doc = col.find_one({"_id": f"{ds_name}{q_index:03}"})
    return doc["target_sentences"]


def already_processed(col, model_name_key, ds_name, q_index, git_hash):
    """Check if the document has already been processed"""
    doc = col.find_one(
        {
            "_id": f"{ds_name}{q_index:03}",
            f"{model_name_key}": {"$exists": True},
            "git_hash": git_hash,
        }
    )
    return doc is not None


def cleanup_logging(listener=None, handlers=None):
    """
    Cleans up logging resources by stopping the QueueListener and closing handlers.

    :param listener: The QueueListener instance to stop.
    :param handlers: A list of logging handlers to close.
    """
    # Stop the listener
    if listener is not None:
        listener.stop()

    # Close each handler
    if handlers is not None:
        for handler in handlers:
            handler.close()


def get_git_hash():
    """
    Get the git hash
    """
    # run git describe --always --dirty to get the git hash
    git_hash = (
        subprocess.check_output(["git", "describe", "--always", "--dirty"])
        .strip()
        .decode("utf-8")
    )
    return git_hash


# Worker function to process data and write to MongoDB
def process_data(item):
    ds_name, q_index, ctxt, tol, model_name, target_sentences = item

    tokenizer = init_tokenizer(model_name)
    sentences = target_sentences
    git_hash = get_git_hash()

    log = logging.getLogger(f"{ds_name}_{q_index:03}")
    log.setLevel(logging.INFO)

    result = compute(tokenizer, sentences, ctxt, tol, log, ds_name, q_index, git_hash)

    cleanup_logging(handlers=log.handlers)

    return result


# Simulate a computation
def compute(tokenizer, sentences, ctxt, tol, log, ds_name, q_index, git_hash):
    d = targetsentences2sentences(tokenizer, sentences, ctxt, tol, log)
    ts = d["target_sentences"]
    s = d["sentences"]
    assert len(ts) == len(s)

    dist = [lev_dist(ts_i, s_i) for ts_i, s_i in zip(ts, s)]
    mean_dist = sum(dist) / len(dist)
    nz = [d for d in dist if d > 0]
    non_zero_mean_dist = sum(dist) / len(nz) if len(nz) > 0 else 0

    return d | {
        "ds_name": ds_name,
        "q_index": q_index,
        "git_hash": git_hash,
        "mean_dist": mean_dist,
        "non_zero_mean_dist": non_zero_mean_dist,
    }


# Main function to setup multiprocessing
@click.command()
@click.option("--model_name")
@click.option("--target_sentence_file")
@click.option("--data_path", default="data")
def main(model_name, target_sentence_file, data_path):

    if not os.path.exists(target_sentence_file):
        raise FileNotFoundError(
            f"{target_sentence_file} not found. Consider running scripts/1_data.sh to generate the target sentences."
        )

    # Load target sentences
    with open(target_sentence_file, "r", encoding="utf-8") as f:
        target_sentences = json.load(f)
    per_ds = {ds_name: [[]] * 200 for ds_name in DS_NAME}
    for ts in target_sentences:
        per_ds[ts["ds_name"]][ts["q_index"]] = ts["target_sentences"]

    ds_path = "THUDM/LongBench"
    tol = 30

    all_ds = []
    for ds_name in DS_NAME:
        dataset = datasets.load_dataset(ds_path, name=ds_name, split="test")

        all_ds.extend(
            [
                (ds_name, i, data["context"], tol, model_name, per_ds[ds_name][i])
                for i, data in enumerate(dataset)
            ]
        )

    # Number of worker processes
    num_workers = min(1, int(multiprocessing.cpu_count() - 10))

    # Create a pool of workers
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_data, all_ds)

    # Write results to a file
    _model_name = MODELPATH2MODELNAME[model_name]
    os.makedirs(data_path, exist_ok=True)
    with open(f"{data_path}/easy.{_model_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
