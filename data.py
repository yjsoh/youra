import os
import json

import click
import scipy.spatial
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy
import numpy as np

from datasets import load_dataset

from src import Config
from src import init_nlp, init_tokenizer, init_llm
from src import context2sentences
from src import get_reaction_vec
from src import PKVManager
from src import SentenceTokenSequenceIndex, SentenceTokenSequenceChunk

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

encode_options = {"add_special_tokens": False, "return_tensors": "pt"}


@click.group()
@click.pass_context
@click.option("--config", help="Path to the config file")
@click.option("--data_path", default="data", help="Path to save the processed data")
def cli(ctx, config, data_path):
    """CLI for processing the data"""
    ctx.ensure_object(dict)
    with open(config, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    ctx.obj["config"] = Config.from_dict(loaded)
    ctx.obj["model_name"] = ctx.obj["config"].config_dict["model_name"]
    ctx.obj["nlp_name"] = ctx.obj["config"].config_dict["nlp"]

    ctx.obj["data_path"] = data_path
    os.makedirs(data_path, exist_ok=True)


@cli.command()
@click.pass_context
def count_tokens(ctx):
    """Count the number of tokens in each context"""

    tokenizer = init_tokenizer(ctx.obj["config"])
    model_name = MODELPATH2MODELNAME[ctx.obj["model_name"]]
    data_path = ctx.obj["data_path"]

    output_filename = f"{data_path}/data.count_tokens.{model_name}.csv"
    if os.path.exists(output_filename):
        print("Already processed token counts")
        print(f"Delete the file ({output_filename}) to reprocess")
        return

    to_ret = []
    for ds_name in DS_NAME:
        ds = load_dataset("THUDM/LongBench", ds_name, split="test")

        for q_index, _ds in tqdm(enumerate(ds)):

            ntokens = tokenizer.encode(_ds["context"], **encode_options).shape[-1]

            to_ret.append(
                {
                    "q_index": q_index,
                    "ds_name": ds_name,
                    "model_name": model_name,
                    "ntokens": ntokens,
                }
            )

    df = pd.DataFrame(to_ret)
    df.to_csv(output_filename, index=False)

    df["ds_name"] = pd.Categorical(df["ds_name"], DS_NAME)

    df.groupby("ds_name")["ntokens"].mean().astype(int).reset_index().to_csv(
        f"data/ntokens.{model_name}.csv"
    )


@cli.command()
@click.pass_context
def get_target_sentences(ctx):
    """Get target sentences for each question in the dataset"""

    nlp = init_nlp(ctx.obj["config"])
    nlp_name = ctx.obj["nlp_name"]
    data_path = ctx.obj["data_path"]

    output_filename = f"{data_path}/data.target_sentences.{nlp_name}.json"
    if os.path.exists(output_filename):
        print("Already processed target sentences")
        print(f"Delete the file ({output_filename}) to reprocess")
        return

    to_ret = []
    for ds_name in DS_NAME:
        ds = load_dataset("THUDM/LongBench", ds_name, split="test")
        for q_index, _ds in tqdm(enumerate(ds)):

            doc = nlp.process(_ds["context"])
            target_sentences = [sentence.text for sentence in doc.sentences]

            to_ret.append(
                {
                    "q_index": q_index,
                    "ds_name": ds_name,
                    "model_name": nlp_name,
                    "target_sentences": target_sentences,
                }
            )

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(to_ret, f)


@cli.command()
@click.pass_context
def teaser(ctx):
    """Get teaser data"""
    sentences = [
        "A woman is wearing a red dress.",
        "She is walking on the street.",
        "Her name is Mary.",
    ]

    query = "Where is Mary?"

    topk = 1

    append_prompt_str = "Answer the following question with simple word. Do not any additional text. Question: "

    # Embedding based retrieval
    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_path)

    q_emb = model.encode(query)
    embeddings = model.encode(sentences)

    sentence_dist_pairs = [
        (sentence, scipy.spatial.distance.cosine(q_emb, emb))
        for sentence, emb in zip(sentences, embeddings)
    ]

    print(f"Query: {query}")
    print(f"Sentences: {sentences}")
    print("-" * 100)
    print("Sorted by distance")
    for sentence, dist in sorted(sentence_dist_pairs, key=lambda x: x[1]):
        print(f'"{sentence:40}": {dist:.2f}')

    tokenizer = init_tokenizer(ctx.obj["config"])
    llm = init_llm(ctx.obj["config"])
    emb_retrieved = " ".join(
        [s for s, _ in sorted(sentence_dist_pairs, key=lambda x: x[1])[:topk]]
    )
    emb_prompt = f"{emb_retrieved} Question: {query}"
    emb_prompt_ids = tokenizer.encode(emb_prompt, **encode_options).to(llm.device)
    generated = llm.generate(
        emb_prompt_ids, max_new_tokens=100, do_sample=True, return_dict_in_generate=True
    )
    decoded = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
    print(decoded)
    print("-" * 100)

    # LLM based retrieval
    context = "\n".join(sentences)

    nlp = init_nlp(ctx.obj["config"])
    c2s = context2sentences(tokenizer, nlp, context)

    pkvm = PKVManager(ctx.obj["config"], tokenizer, llm)

    ctxt_ids = tokenizer.encode(context, **encode_options)

    quant_indices = c2s["sentence_quant_indices"]
    sentences = c2s["target_sentences"]
    target_sentences = c2s["target_sentences"]

    attention_method = ctx.obj["config"].config_dict["YOURA_RETRIEVER"][
        "attention_method"
    ]
    s_index = SentenceTokenSequenceIndex((0, quant_indices[-1]), 1, 0, 0)
    chunk = SentenceTokenSequenceChunk(llm, s_index, sentences, target_sentences)
    b, a, r = get_reaction_vec(
        tokenizer,
        llm,
        pkvm,
        ctxt_ids,
        query,
        chunk,
        attention_method=attention_method,
        append_prompt_str=append_prompt_str,
    )

    reaction_vec = r
    sentence_rs_pairs = []
    for i, (l, r) in enumerate(zip(quant_indices[:-1], quant_indices[1:])):
        rs = scipy.stats.gmean(reaction_vec[l:r])
        sentence_rs_pairs.append((target_sentences[i], rs))

    rs_sorted = sorted(sentence_rs_pairs, key=lambda x: x[1], reverse=True)
    for i, (sentence, rs) in enumerate(rs_sorted):
        print(f"{sentence:40}: {rs:.9f}")

    rs_retrieved = " ".join([s for s, _ in rs_sorted[:topk]])
    rs_prompt = f"{rs_retrieved} Question: {query}"
    rs_prompt_ids = tokenizer.encode(rs_prompt, **encode_options).to(llm.device)
    generated = llm.generate(
        rs_prompt_ids, max_new_tokens=15, do_sample=True, return_dict_in_generate=True
    )
    decoded = tokenizer.decode(generated.sequences[0], skip_special_tokens=True)
    print(decoded)
    print("-" * 100)


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=no-value-for-parameter
