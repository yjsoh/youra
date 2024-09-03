""" youra.py Generate vllm_input for (model, setup, dataset) triplet. """

import os
import logging
import json
from datetime import datetime

import click
from tqdm import tqdm
import datasets
import torch
import pandas as pd


from src import Config
from src import init_llm, init_tokenizer, set_seed
from src import (
    SentenceTokenSequenceLoadingChunkenizer,
    AttentionFlatRetriever,
    LongBenchAugmenter,
    TransparentGenerator,
    VLLMInputGenerator,
    LongBenchGrader,
    LongBenchEvaluator,
    PKVManager,
)

DS_NAME = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
]

Q_INDEX_HELP = "Question index -1 means all question"

MODELPATH2MODELNAME = {
    "meta-llama/Llama-2-7b-chat-hf": "llama2",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistralai",
}


@click.group()
@click.option("--config", type=click.Path(exists=True), default="config.json")
@click.option("--ds_path", type=str, default="THUDM/LongBench", multiple=False)
@click.option("--ds_name", type=click.Choice(DS_NAME), default=DS_NAME, multiple=True)
@click.option("--q_index", type=int, default=-2, help=Q_INDEX_HELP)
@click.option("--q_indices", "-q", type=int, default=[], multiple=True)
@click.option("--output_path", type=str, default=datetime.now().strftime("%F-%H-%M-%S"))
@click.option("--data_path", type=str, default="data")
@click.option("--local_files_only/--generate_if_needed", default=False)
@click.option("--mem_frac", type=float, default=1)
@click.option("--dry_run", type=bool, default=True)
@click.option("--verbose", "-v", count=True)
@click.option("--transparent_generate", type=bool, default=True)
@click.option("--exp_name", type=str, default="")
@click.option("--exp_description", type=str, default="")
@click.option("--git_hash", type=str, default="")
@click.option("--reuse_dag", type=bool, default=False)
@click.option("--reuse_retrieved", type=bool, default=False)
@click.option("--reuse_augmented", type=bool, default=False)
@click.option("--reuse_generated", type=bool, default=False)
@click.pass_context
def cli(
    ctx,
    config,
    ds_path,
    ds_name,
    q_index,
    q_indices,
    output_path,
    data_path,
    local_files_only,
    mem_frac,
    dry_run,
    verbose,
    transparent_generate,
    **kwargs,
):
    """
    Evaluate the methods end-to-end.

    kwargs:
    dry_run,
    exp_name,
    exp_description,
    git_hash,
    reuse_dag,
    reuse_retrieved,
    reuse_augmented,
    reuse_generated
    """

    # q_index and q_indices are mutually exclusive
    assert not (q_index != -2 and len(q_indices) > 0)  # both
    assert not (q_index == -2 and len(q_indices) == 0)  # neither

    ctx.ensure_object(dict)

    ## experiment configurations
    ctx.obj["EXP_NAME"] = kwargs.get("exp_name", "")
    ctx.obj["EXP_DESCRIPTION"] = kwargs.get("exp_description", "")
    ctx.obj["GIT_HASH"] = kwargs.get("git_hash", "")
    with open(config, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    config = Config.from_dict(loaded)
    git_hash = config.get_git_hash()
    if ctx.obj["GIT_HASH"] == "":
        ctx.obj["GIT_HASH"] = git_hash
    elif ctx.obj["GIT_HASH"] != git_hash:
        print(f"Error: git hash mismatch: {ctx.obj['GIT_HASH']} != {git_hash}")
        exit(1)
    ctx.obj["CONFIG"] = config
    ctx.obj["DS_PATH"] = ds_path
    ctx.obj["DS_NAME"] = ds_name
    ctx.obj["DRY_RUN"] = dry_run
    ctx.obj["Q_INDEX"] = q_index
    ctx.obj["Q_INDICES"] = q_indices

    # run configurations
    ctx.obj["OUTPUT_PATH"] = output_path
    ctx.obj["DATA_PATH"] = data_path
    ctx.obj["LOCAL_FILES_ONLY"] = local_files_only
    ctx.obj["MEM_FRAC"] = mem_frac

    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/config.json", "w", encoding="utf-8") as f:
        json.dump(loaded, f, indent=2)

    if mem_frac < 1:
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        print(f"Memory fraction: {mem_frac}")

    if verbose == 0:
        log_level = logging.WARNING
    elif verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    ctx.obj["LOG_LEVEL"] = log_level

    ctx.obj["TRANSPARENT_GENERATE"] = transparent_generate


###################################################################################################
@cli.command()
@click.pass_context
def youra(ctx):
    """
    Evaluate the method.
    """
    exp_name = ctx.obj["EXP_NAME"]
    exp_desc = ctx.obj["EXP_DESCRIPTION"]
    git_hash = ctx.obj["GIT_HASH"]
    config = ctx.obj["CONFIG"]
    ds_path = ctx.obj["DS_PATH"]
    ds_name = ctx.obj["DS_NAME"]
    q_index = ctx.obj["Q_INDEX"]
    output_path = ctx.obj["OUTPUT_PATH"]
    data_path = ctx.obj["DATA_PATH"]
    dry_run = ctx.obj["DRY_RUN"]
    log_level = ctx.obj["LOG_LEVEL"]
    transparent_generate = ctx.obj["TRANSPARENT_GENERATE"]

    # initialize the components
    _llm = init_llm(config)
    _tokenizer = init_tokenizer(config)
    _chunkenizer = SentenceTokenSequenceLoadingChunkenizer(
        config,
        _tokenizer,
        _llm,
        how_root=config.config_dict["YOURA_CHUNKENIZER"]["ROOT_HANDLING_METHOD"],
        context_window_size=config.config_dict["YOURA_CHUNKENIZER"][
            "CONTEXT_WINDOW_SIZE"
        ],
        root_split_offset=config.config_dict["YOURA_CHUNKENIZER"]["ROOT_SPLIT_OFFSET"],
        how_quant=config.config_dict["YOURA_CHUNKENIZER"]["QUANT_METHOD"],
        quant=config.config_dict["YOURA_CHUNKENIZER"]["QUANT"],
        data_path=data_path,
        force_construct_dag=config.config_dict["YOURA_CHUNKENIZER"][
            "FORCE_CONSTRUCT_DAG"
        ],
        add_prepend_prompt=config.config_dict["YOURA_CHUNKENIZER"][
            "ADD_PREPEND_PROMPT"
        ],
        update_dag=config.config_dict["YOURA_CHUNKENIZER"]["UPDATE_DAG"],
        log_level=log_level,
    )
    _retriever = AttentionFlatRetriever(
        config,
        _tokenizer,
        _llm,
        context_window_size=config.config_dict["YOURA_RETRIEVER"][
            "context_window_size"
        ],
        attn_agg=config.config_dict["YOURA_RETRIEVER"]["attn_agg"],
        reaction_agg=config.config_dict["YOURA_RETRIEVER"]["reaction_agg"],
        append_prompt_file=config.config_dict["YOURA_RETRIEVER"]["append_prompt_file"],
        prepend_prompt_file=config.config_dict["YOURA_RETRIEVER"][
            "prepend_prompt_file"
        ],
        compression_ratio=config.config_dict["YOURA_RETRIEVER"]["compression_ratio"],
        retain_ctxt_order=config.config_dict["YOURA_RETRIEVER"]["retain_ctxt_order"],
        postprocess_merge_retrieved=config.config_dict["YOURA_RETRIEVER"][
            "postprocess_merge_retrieved"
        ],
        retrieval_budget=config.config_dict["YOURA_RETRIEVER"]["retrieval_budget"],
        add_prepend_prompt=config.config_dict["YOURA_CHUNKENIZER"][
            "ADD_PREPEND_PROMPT"
        ],  # depends on the chunkenizer
        attention_method=config.config_dict["YOURA_RETRIEVER"]["attention_method"],
        log_level=log_level,
    )
    model_name = MODELPATH2MODELNAME[config.config_dict["model_name"]]
    os.makedirs(data_path, exist_ok=True)

    all_df = []
    for _ds_name in ds_name:

        if q_index == -1:
            if _ds_name == "multifieldqa_en":
                n_questions = 150
            else:
                n_questions = 200
            q_indices = [x for x in range(n_questions)]
        elif q_index == -2:
            q_indices = ctx.obj["Q_INDICES"]
        else:
            q_indices = [q_index]

        _augmenter = LongBenchAugmenter(
            config, _tokenizer, log_level=log_level, ds_name=_ds_name
        )
        if transparent_generate:
            _generator = TransparentGenerator(
                config, _tokenizer, _llm, log_level=log_level, ds_name=_ds_name
            )
        else:
            _generator = VLLMInputGenerator(
                config, _tokenizer, _llm, log_level=log_level, ds_name=_ds_name
            )
        _grader_list = []
        _grader_list.append(LongBenchGrader(config))
        _evaluator = LongBenchEvaluator(
            config,
            _tokenizer,
            _llm,
            _chunkenizer,
            _retriever,
            _augmenter,
            _generator,
            _grader_list,
            output_path=output_path,
            log_level=log_level,
        )
        ds = datasets.load_dataset("THUDM/LongBench", _ds_name, split="test")

        evaluated = []
        all_graded = []
        for _q_index in tqdm(q_indices):
            try:
                query = ds[_q_index]["input"]
                ctxt = ds[_q_index]["context"]
                truth = ds[_q_index]["answers"]
                pkvm = PKVManager(config, _tokenizer, _llm, log_level=log_level)

                _evaluated = _evaluator.evaluate(
                    ds_path,
                    _ds_name,
                    _q_index,
                    ctxt,
                    query,
                    truth,
                    pkvm=pkvm,
                    dry_run=dry_run,
                    exp_name="youra",
                    exp_description="youra",
                    git_hash=git_hash,
                    output_path=output_path,
                )
                evaluated.append(_evaluated)
                all_graded.append(
                    {
                        "exp_name": exp_name,
                        "exp_description": exp_desc,
                        "git_hash": git_hash,
                        "ds_path": ds_path,
                        "ds_name": _ds_name,
                        "q_index": _q_index,
                        "query": query,
                        "truth": truth,
                        "ntokens": _evaluated["ntokens"],
                        "retrieved_ntokens": _evaluated["retrieved_ntokens"],
                        "token_level_compression_rate": _evaluated[
                            "token_level_compression_rate"
                        ],
                        "generated": _evaluated["purely_generated_str"],
                        "generated_ntokens": _evaluated["generated_ntokens"],
                        "purely_generated_ntokens": _evaluated[
                            "purely_generated_ntokens"
                        ],
                        "score": round(
                            _evaluated["graded"]["LongBenchGrader"][0] * 100, 2
                        ),
                    }
                )
                with open(f"{output_path}/results.csv", "a", encoding="utf-8") as f:
                    f.write(
                        f"{exp_name},{exp_desc},{git_hash},{ds_path},{_ds_name},{_q_index},{_evaluated['ntokens']},{_evaluated['retrieved_ntokens']},{_evaluated['token_level_compression_rate']},{_evaluated['generated_ntokens']},{_evaluated['purely_generated_ntokens']},{_evaluated['graded']['LongBenchGrader'][0]}\n"
                    )

                with open(
                    f"{data_path}/vllm_input.{model_name}.{_ds_name}.youra.jsonl",
                    "a",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "q_index": _q_index,
                            "ds_name": _ds_name,
                            "query": query,
                            "truth": truth,
                            "prompt": _evaluated["generated_str"],
                        },
                        f,
                    )
                    f.write("\n")

            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM: {e}")
                evaluated.append(
                    {
                        "exp_name": exp_name,
                        "exp_description": exp_desc,
                        "git_hash": git_hash,
                        "ds_path": ds_path,
                        "ds_name": _ds_name,
                        "q_index": _q_index,
                        "query": query,
                        "truth": truth,
                        "status": f"error-{str(e)}",
                    }
                )
                raise e

        with open(f"{output_path}/graded_{_ds_name}.json", "w", encoding="utf-8") as f:
            json.dump(evaluated, f, indent=2)

        with open(f"{output_path}/summary_{_ds_name}.json", "w", encoding="utf-8") as f:
            json.dump(all_graded, f, indent=2)

        df = pd.DataFrame(all_graded)
        with open(f"{output_path}/summary_{_ds_name}.csv", "w", encoding="utf-8") as f:
            df.to_csv(f)

        all_df.append(df)

        with open(f"{output_path}/stat_{_ds_name}.csv", "w", encoding="utf-8") as f:
            df.describe().reset_index().to_csv(f)

        print(df.describe())

    concat_df = pd.concat(all_df)
    with open(f"{output_path}/summary_all.csv", "w", encoding="utf-8") as f:
        concat_df.to_csv(f)

    with open(f"{output_path}/stat_all.csv", "w", encoding="utf-8") as f:
        concat_df.groupby("ds_name").describe().reset_index().to_csv(f)

    with open(f"{output_path}/mean_all.csv", "w", encoding="utf-8") as f:
        concat_df.groupby("ds_name").mean(numeric_only=True).reindex(
            DS_NAME
        ).transpose().to_csv(f)


if __name__ == "__main__":
    set_seed(42)  # Same as the longbench
    cli(obj={})  # pylint: disable=no-value-for-parameter
