from datasets import load_dataset, Dataset

import multiprocessing as mp
import numpy as np
from typing import Callable, Dict, List

N_CPUS = mp.cpu_count()


def concat_item_metadata(dp):
    meta = ""
    flag = False
    if dp["title"] is not None:
        meta += dp["title"]
        flag = True
    if len(dp["features"]) > 0:
        if flag:
            meta += " "
        meta += " ".join(dp["features"])
        flag = True
    if len(dp["description"]) > 0:
        if flag:
            meta += " "
        meta += " ".join(dp["description"])
    dp["cleaned_metadata"] = meta.replace("\t", " ").replace("\n", " ").replace("\r", "").strip()
    return dp


def clean_review(review: Dict[str, List]) -> List[bool]:
    return [r != "" for r in review["text"]]


def clean_meta(meta: Dict[str, List]) -> List[bool]:
    return [m != "" for m in meta["description"]]


def load_data(
    top_k_reviews: int = 5000,
    category: str = "Books",
    review_filter: Callable | None = None,
    meta_filter: Callable | None = None,
):
    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        trust_remote_code=True,
    )
    reviews = reviews["full"]

    item_meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        trust_remote_code=True,
    )
    item_meta = item_meta["full"]

    ## get most useful reviews
    if review_filter is not None:
        reviews = reviews.filter(review_filter, batched=True, num_proc=N_CPUS)
    reviews = reviews.sort("helpful_vote", reverse=True).take(top_k_reviews)
    ## filter item_meta to only include rows where parent_asin is in candidate_asin
    candidate_asin = reviews.select_columns(["parent_asin"]).to_dict()["parent_asin"]
    item_meta_filtered = item_meta.filter(
        (lambda x: np.isin(x["parent_asin"], candidate_asin)),
        batched=True,
        num_proc=N_CPUS,
    )
    # filter item_meta, remove reviews without good meta
    if meta_filter is not None:
        item_meta_filtered = item_meta_filtered.filter(meta_filter, batched=True, num_proc=N_CPUS)
        good_meta_asin = item_meta_filtered.select_columns(["parent_asin"]).to_dict()["parent_asin"]
        reviews = reviews.filter(
            (lambda x: np.isin(x["parent_asin"], good_meta_asin)),
            batched=True,
            num_proc=N_CPUS,
        )

    all_meta = {}
    for row in item_meta_filtered:
        all_meta[row["parent_asin"]] = row

    # # merge reviews and meta
    # reviews =

    return reviews, all_meta


def sample_reviews(reviews: Dataset, n: int = 3):
    return reviews.select_columns(["review_id", "text"]).shuffle(seed=42).take(n)
