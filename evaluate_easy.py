import json
import os

import click
import pandas as pd
import pymongo
import scipy


MODELPATH2MODELNAME = {
    "meta-llama/Llama-2-7b-chat-hf": "llama2",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3",
    "mistralai/Mistral-7B-Instruct-v0.2": "mistralai",
    "llama2": "llama2",
    "llama3": "llama3",
    "mistralai": "mistralai",
}

DS_NAME = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
]


###################################################################################################
def process_table4(df, result_path):
    df = df[["model_name", "ds_name", "avg_sr"]]
    df["ds_name"] = pd.Categorical(df["ds_name"], DS_NAME)
    df["model_name"] = df["model_name"].map(MODELPATH2MODELNAME)
    df.pivot(index="model_name", columns="ds_name").sort_index(
        axis="columns", level="ds_name"
    ).round(2).to_csv(f"{result_path}/table4.csv")


def process_table5(df, result_path):
    df = df[["model_name", "ds_name", "avg_dist", "avg_nzdist"]]
    melt_df = pd.melt(
        df,
        id_vars=["model_name", "ds_name"],
        value_vars=["avg_dist", "avg_nzdist"],
        var_name="Metric",
        value_name="Value",
    )
    melt_df.pivot(
        index=["model_name", "Metric"], columns="ds_name", values=["Value"]
    ).round(2).to_csv(f"{result_path}/table5.csv")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("git_hash")
@click.option("--data_path", default="data")
@click.option("--result_path", default="results")
def easy_mongodb(git_hash: str, data_path: str, result_path: str):
    """
    Evaluate the method.
    """

    # MongoDB connection details
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    client = pymongo.MongoClient(mongo_uri)
    db = client["stanza"]

    models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]

    lines = []
    for model_name in models:
        for ds in DS_NAME:
            collection = db[ds]
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "avg_sr": {"$avg": f"${model_name}.{git_hash}.success_rate"},
                        "avg_dist": {"$avg": f"${model_name}.{git_hash}.mean_dist"},
                        "avg_nzdist": {
                            "$avg": f"${model_name}.{git_hash}.non_zero_mean_dist"
                        },
                    }
                }
            ]

            stats = list(collection.aggregate(pipeline))

            if stats:
                stats_obj = stats[0]
                lines.append(
                    f"{model_name},{ds},{stats_obj['avg_sr']},{stats_obj['avg_dist']},{stats_obj['avg_nzdist']}"
                )
                print(lines[-1])
            else:
                print(f"Error: {model_name} {ds}")

    client.close()

    with open(f"{data_path}/easy.csv", "w", encoding="utf-8") as f:
        f.write("model_name,ds_name,avg_sr,avg_dist,avg_nzdist\n")
        for line in lines:
            f.write(f"{line}\n")

    os.makedirs(result_path, exist_ok=True)
    df = pd.read_csv(f"{data_path}/easy.csv")
    process_table4(df, result_path)

    df = pd.read_csv(f"{data_path}/easy.csv")
    process_table5(df, result_path)


@cli.command()
@click.option("--data_path", default="data")
@click.option("--result_path", default="results")
def easy(data_path, result_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path ({data_path}) does not exist.")

    models = MODELPATH2MODELNAME.values()

    objs = []
    for model in models:
        with open(f"{data_path}/easy.{model}.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)

        for _loaded in loaded:
            objs.append(
                {
                    "model_name": model,
                    "ds_name": _loaded["ds_name"],
                    "q_index": _loaded["q_index"],
                    "avg_sr": _loaded["success_rate"],
                    "avg_dist": _loaded["mean_dist"],
                    "avg_nzdist": _loaded["non_zero_mean_dist"],
                }
            )

    df = pd.DataFrame(objs)
    df = (
        df.groupby(["model_name", "ds_name"])[["avg_sr", "avg_dist", "avg_nzdist"]]
        .mean(numeric_only=True)
        .round(2)
        .reset_index()
    )

    os.makedirs(result_path, exist_ok=True)
    process_table4(df, result_path)
    process_table5(df, result_path)


###################################################################################################
if __name__ == "__main__":
    cli()
