import os
import json
import logging
import multiprocessing as mp
import pandas as pd

from src import LongBenchGrader, Config

DS_NAME = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
]

MODEL_NAME = [
    "llama2",
    "llama3",
    "mistralai",
]

TYPE = ["youra"]


if __name__ == "__main__":

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    tq = mp.JoinableQueue()

    manager = mp.Manager()
    result_list = manager.list()

    def mp_grader(log, grader, task_queue):
        """Worker process for grading tasks."""
        global result_list

        while True:
            task = task_queue.get()

            if task is None:
                task_queue.task_done()
                break
            else:
                model_name, ds_name, q_index, _type, _truth, _generated = task

            try:
                _result = grader.grade("", _truth, _generated)
                result_list.append(
                    {
                        "model_name": model_name,
                        "ds_name": ds_name,
                        "q_index": q_index,
                        "type": _type,
                        "grader": _result["grader"],
                        "scores": _result["scores"],
                        "score": _result["max_score"],
                    }
                )
                log.debug(f"{task} Result put {_result}")
            except Exception as e:
                log.warn(e)
                log.warn("Error in grading")

            task_queue.task_done()  # Mark the task as done
        log.debug("Process done")

    # Start worker processes
    num_workers = os.cpu_count()
    workers = []
    config_filename = "config/config.llama2.json"
    with open(config_filename, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    config = Config.from_dict(loaded)

    for i in range(num_workers):
        lbgrader = LongBenchGrader(config)
        p = mp.Process(target=mp_grader, args=(log, lbgrader, tq))
        p.start()
        workers.append(p)

    vllm_df_list = []
    i = 0
    for model_name in MODEL_NAME:
        for ds_name in DS_NAME:
            for _type in TYPE:
                filename = f"results/{model_name}.{_type}.{ds_name}.json"
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                elapsed_time = data["elapsed_time"]
                num_requests = data["num_requests"]
                total_num_tokens = data["total_num_tokens"]
                requests_per_second = data["requests_per_second"]
                tokens_per_second = data["tokens_per_second"]
                generated = data["generated"]
                truth = data["truth"]

                vllm_df_list.append(
                    pd.DataFrame(
                        {
                            "model_name": model_name,
                            "ds_name": ds_name,
                            "type": _type,
                            "elapsed_time": elapsed_time,
                            "num_requests": num_requests,
                            "total_num_tokens": total_num_tokens,
                            "requests_per_second": requests_per_second,
                            "tokens_per_second": tokens_per_second,
                        },
                        index=[i],
                    )
                )
                i += 1

                for q_index, (g, t) in enumerate(zip(generated, truth)):
                    tq.put((model_name, ds_name, q_index, _type, t, [g]))

    df = pd.concat(vllm_df_list)
    df.to_csv("results/table3.csv", index=False)

    for _ in range(num_workers):
        tq.put(None)
    tq.join()  # Wait until all tasks are done

    for i, p in enumerate(workers):
        p.join()

    log.info("Collected results: %d", len(result_list))

    quality_df_list = []
    for i, r in enumerate(result_list):
        print(f"Result {i}: {r}")
        quality_df_list.append(pd.DataFrame(r))

    df = pd.concat(quality_df_list)

    pd.options.display.float_format = "{:,.2f}".format

    df.loc[
        (df["ds_name"] == "narrativeqa")
        | (df["ds_name"] == "qasper")
        | (df["ds_name"] == "multifieldqa_en"),
        "ds_type",
    ] = "SDoc"
    df.loc[
        (df["ds_name"] == "hotpotqa")
        | (df["ds_name"] == "musique")
        | (df["ds_name"] == "2wikimqa"),
        "ds_type",
    ] = "MDoc"

    df["ds_type"] = pd.Categorical(df["ds_type"], ["SDoc", "MDoc"])
    df["ds_name"] = pd.Categorical(
        df["ds_name"],
        ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"],
    )

    df = (
        pd.pivot_table(
            df,
            values=["score"],
            columns=["ds_type", "ds_name"],
            index=["model_name", "type"],
        )
        * 100
    )

    df["score"].groupby(level=0, axis=1).mean().to_csv("results/table4.csv")

    df["score"].to_csv("results/table6.csv")
