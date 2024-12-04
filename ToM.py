import pandas as pd
import numpy as np
import openai_utils as oai


def get_dataset(split="train"):
    splits = {
        "test": "plain_text/test-00000-of-00001.parquet",
        "validation": "plain_text/validation-00000-of-00001.parquet",
        "train": "plain_text/train-00000-of-00001.parquet",
    }
    df = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits[split])

    return df


def get_m_few_shots_text(m_examples):
    df_train = get_dataset("train")

    # Get m random examples
    df = df_train.sample(m_examples)

    text = ""

    for i, row in df.iterrows():
        text += f"Premise: {row['premise']}\n"
        text += f"Hypothesis: {row['hypothesis']}\n"
        text += f"Label: {row['label']}\n\n"

    return text


def get_n_new_premises(n_new, df_processed):
    df = get_dataset("test")
    df = df[~df["premise"].isin(df_processed["premise"])]
    return df.sample(n=n_new)


def get_premise_individual_class(hypothesis, premise, label, m_examples):
    sys_prompt = """
    Premise and hypothesis classification task

    Instructions:

    1. Your task is to label a the relationship between a premise and a hypothesis. The label is a number between 0 and 2:
        0 - Entailment, indicating that the hypothesis entails the premise.
        1 - Neutral, indicating that the premise and hypothesis neither entail nor contradict each other
        2 - Contradiction, indicating that the hypothesis contradicts the premise

    2. Here you have a few examples to guide your classification:

    """

    sys_prompt += get_m_few_shots_text(m_examples)

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
        # model = "gpt-3.5-turbo-0125",
        temp=0.7,
    )

    # Round response
    response = int(response)

    # Get real classification
    result = int(response == label)

    return response, result


def vote_mean(votes):
    return np.mean(votes).round(0)


def get_premise_class(premise, hypothesis, label, n_agents, m_examples):
    votes = []

    for i in range(0, n_agents):
        response, result = get_premise_individual_class(premise, hypothesis, label, m_examples)
        response = int(response)
        votes.append(response)

    # rounding votes to get the 0, 1 or 2 classification
    y_predicted = vote_mean(votes)

    # Get 1 correct or 0 incorrect
    result = int(y_predicted == label)

    return y_predicted, result
