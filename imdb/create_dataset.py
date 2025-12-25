"""
Create a IMDB dataset
"""

import os
import datasets
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/processed/stanfordnlp/imdb",
        help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "stanfordnlp/imdb"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("text")

            question = question_raw

            answer_raw = example.pop("label")
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Please write a review about a movie.",
                    },
                    {
                        "role": "assistant",
                        "content": question,
                    },
                ],
                "label": answer_raw
            }
            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

