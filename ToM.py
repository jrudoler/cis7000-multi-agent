import pandas as pd
import numpy as np
import openai_utils as oai
from joblib import Parallel, delayed


def get_dataset(split="train"):
    splits = {
        "test": "test-00000-of-00001.parquet",
        "validation": "validation-00000-of-00001.parquet",
        "train": "train-00000-of-00001.parquet",
    }
    df = pd.read_parquet("hf://datasets/stanfordnlp/snli/plain_text/" + splits[split])

    return df


def get_m_few_shots_text(m_examples):
    """
    Get m few shot examples, sampled randomly from the train set
    """
    df_train = get_dataset("train")

    # Get m random examples
    df = df_train.sample(m_examples)

    # track the indices of the examples
    indices = df.index.tolist()

    text = ""

    for i, row in df.iterrows():
        text += f"Premise: {row['premise']}\n"
        text += f"Hypothesis: {row['hypothesis']}\n"
        text += f"Label: {row['label']}\n\n"

    return text, indices


def get_n_new_premises(n_new, df_processed):
    """
    Get n new premises from the test set that have not been processed yet
    """
    df = get_dataset("test")
    df = df[~df["premise"].isin(df_processed["premise"])]
    return df.sample(n=n_new)


def classify_hypothesis_single_agent(hypothesis, premise, label, m_examples):
    """
    Classify the hypothesis using a single agent
    """

    sys_prompt = """
    Premise and hypothesis classification task

    Instructions:

    1. Your task is to label a the relationship between a premise and a hypothesis. The label is a number between 0 and 2:
        0 - Entailment, indicating that the hypothesis entails the premise.
        1 - Neutral, indicating that the premise and hypothesis neither entail nor contradict each other
        2 - Contradiction, indicating that the hypothesis contradicts the premise

    2. Here you have a few examples to guide your classification:

    """
    ## Get the examples and their indices
    examples, indices = get_m_few_shots_text(m_examples)
    sys_prompt += examples

    sys_prompt += """
    3. For the given hypothesis and premise, your answer should be a number between 0 and 2. 
    Do not add any comment, explanation or unnecesary characters. 
    """

    testing_rel = "Premise: " + premise + "\n"
    testing_rel += "Hypothesis: " + hypothesis + "\n"
    testing_rel += "Label: "

    response = oai.get_GPT_response(
        prompt_sys=sys_prompt,
        prompt_user=testing_rel,
        model="gpt-4o-mini",
        temp=0.7,
    )

    # Round response
    response = int(response)

    # Get real classification
    result = int(response == label)

    return response, result


def vote_mean(votes):
    # rounding votes to get the 0, 1 or 2 classification
    return np.mean(votes).round(0)


def classify_hypothesis_n_agents(premise, hypothesis, label, n_agents, m_examples, vote_func=vote_mean):
    """
    Classify the hypothesis using n agents, using the vote function to aggregate the votes
    """
    votes = []

    votes = Parallel(n_jobs=n_agents, return_as="list")(
        delayed(classify_hypothesis_single_agent)(premise, hypothesis, label, m_examples) for _ in range(n_agents)
    )

    y_predicted = vote_func(votes)

    # Get 1 correct or 0 incorrect
    result = int(y_predicted == label)

    return y_predicted, result
