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
    df_train = df_train.query("label != -1")

    # Get m random examples
    df = df_train.sample(m_examples)

    # track the indices of the examples
    indices = df.index.tolist()

    text = ""

    for i, row in df.iterrows():
        text += f"Premise: {row['premise']}\n"
        text += f"Hypothesis: {row['hypothesis']}\n"
        text += f"Label: {row['label']}\n\n"

    # remove the last \n\n
    text = text.rstrip()

    return text, indices


# def get_n_new_premises(n_new, df_processed):
#     """
#     Get n new premises from the test set that have not been processed yet
#     """
#     df = get_dataset("test")
#     df = df[~df["premise"].isin(df_processed["premise"])]
#     return df.sample(n=n_new)


def classify_hypothesis_single_agent(premise, hypothesis, label, m_examples, temp=0.1):
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

    sys_prompt += """\n\n
    3. For the given hypothesis and premise, provide a label in {0, 1, 2}. 
    Do not add any comment, explanation or unnecessary characters. 
    """

    testing_rel = "Premise: " + premise + "\n"
    testing_rel += "Hypothesis: " + hypothesis + "\n"
    testing_rel += "Label: "

    response = oai.get_GPT_response(
        prompt_sys=sys_prompt,
        prompt_user=testing_rel,
        model="gpt-4o-mini",
        temp=temp,  # changed from 0.7 to 0.1, we probably want the effect of few-shot to be more pronounced
    )

    # Round response
    response = int(response)

    return response


def vote_mean(votes):
    # rounding votes to get the 0, 1 or 2 classification
    return int(np.mean(votes).round(0))


def vote_majority(votes):
    return int(np.argmax(np.bincount(votes)))


def classify_hypothesis_n_agents(
    premise, hypothesis, label, n_agents, m_examples, vote_funcs=[vote_mean, vote_majority], temp=0.1
):
    """
    Classify the hypothesis using n agents, using the vote functions to aggregate the votes
    """
    votes = Parallel(n_jobs=n_agents, return_as="list")(
        delayed(classify_hypothesis_single_agent)(
            premise=premise, hypothesis=hypothesis, label=label, m_examples=m_examples, temp=temp
        )
        for _ in range(n_agents)
    )
    print(votes)
    results = {}
    for vote_func in vote_funcs:
        results[vote_func.__name__] = vote_func(votes)

    return results


def get_factors(n):
    """Return a list of factors of the given number n."""
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def propose_configurations(total_m_examples):
    """Propose configurations based on the factors of total_m_examples."""
    factors = get_factors(total_m_examples)
    if len(factors) <= 2:
        raise ValueError("total_m_examples must have more than two factors to propose configurations.")

    # Generate configurations by pairing factors
    configurations = []
    for i in range(1, len(factors) // 2 + 1):
        config = [factors[i], total_m_examples // factors[i]]
        configurations.append(config)

    return configurations


def pad_configurations(configurations):
    """
    Pad configurations to create an array of length max(n_agents) where the first n_agents entries equal n_examples
    and the rest are zero.
    """
    padded_configs = []
    for config in configurations:
        n_agents, n_examples = config
        length_array = max(n_agents for n_agents, _ in configurations)
        padded_array = np.pad(np.repeat([n_examples], n_agents), (0, length_array - n_agents))
        padded_configs.append(padded_array)

    return padded_configs


def classify_with_different_configurations(premise, hypothesis, total_m_examples, configurations=None, temp=0.1):
    """
    Classify the hypothesis using n agents with different configurations of few-shot examples.

    :param premise: The premise text.
    :param hypothesis: The hypothesis text.
    :param label: The true label for the premise-hypothesis pair.
    :param total_m_examples: Total number of few-shot examples to distribute among agents.
    :param configurations: Optional. Padded array of configurations for each agent, e.g., [[9, 0], [3, 3]].
    :return: Dictionary of results from different voting functions.
    """
    if configurations is None:
        configurations = propose_configurations(total_m_examples)

    # Ensure each configuration sums to total_m_examples
    for config in configurations:
        assert (np.prod(config) == total_m_examples).all(), "Each configuration must have total_m_examples"

    padded_configurations = pad_configurations(configurations)

    # Get the same m few-shot examples for all agents
    examples, indices = get_m_few_shots_text(total_m_examples)

    # Split examples according to configurations
    # split_indices = np.split(indices, np.cumsum(flat_configurations)[:-1])

    repeated_examples = np.tile(examples.split("\n\n"), len(padded_configurations))
    split_examples = np.split(repeated_examples, np.cumsum(padded_configurations)[:-1])
    # print(split_examples)

    results = []
    for config in padded_configurations:
        votes = []
        for i, num_examples in enumerate(config):
            agent_examples = "\n\n".join(split_examples.pop(0))
            print(num_examples, len(agent_examples.split("\n\n")), config)
            if num_examples > 0:
                # agent_examples should not be empty
                assert agent_examples != "", "agent_examples should not be empty"

                sys_prompt = f"""
                Premise and hypothesis classification task

                Instructions:

                1. Your task is to label a the relationship between a premise and a hypothesis. The label is a number between 0 and 2:
                    0 - Entailment, indicating that the hypothesis entails the premise.
                    1 - Neutral, indicating that the premise and hypothesis neither entail nor contradict each other
                    2 - Contradiction, indicating that the hypothesis contradicts the premise

                2. Here you have a few examples to guide your classification:

                {agent_examples}

                3. For the given hypothesis and premise, provide a label in {0, 1, 2}. 
                Do not add any comment, explanation or unnecessary characters. 
                """

                testing_rel = "Premise: " + premise + "\n"
                testing_rel += "Hypothesis: " + hypothesis + "\n"
                testing_rel += "Label: "

                response = oai.get_GPT_response(
                    prompt_sys=sys_prompt,
                    prompt_user=testing_rel,
                    model="gpt-4o-mini",
                    temp=temp,
                )

                # Round response
                votes.append(int(response))
        config_results = {}
        for vote_func in [vote_mean, vote_majority]:
            config_results[vote_func.__name__] = vote_func(votes)
        results.append(config_results)

    return results, configurations, indices
