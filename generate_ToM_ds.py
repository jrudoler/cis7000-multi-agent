from ToM import get_premise_class, get_dataset
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute

# Set the seed for reproducibility
np.random.seed(42)


# try:
#     # Load file if exists
#     df_results = pd.read_excel("data/results_ToM.csv")
# except:
#     df_results = pd.DataFrame(columns=["premise", "hypothesis", "label", "predicted", "result"])

df_results = pd.DataFrame(columns=["premise", "hypothesis", "label", "predicted", "result"])

# Config
few_shot = [1, 4, 8]  # [1, 4, 8, 16, 32]
agents = [1, 3, 5]  # [1, 3, 5, 9]

# The batch size of new samples
n_test_samples = 30

# Add new results
# new_samples = ToM.get_n_new_premises(new_n_samples, df_results)


@delayed
def process_sample(sample, few_shot, agents) -> list[dict]:
    sample_premise = sample["premise"]
    sample_hypothesis = sample["hypothesis"]
    sample_label = sample["label"]
    results = []

    for shot in few_shot:
        for agent in agents:
            # Get the prediction with combination of agents and few shot examples
            predicted, result = get_premise_class(sample_hypothesis, sample_premise, sample_label, agent, shot)
            new_row = {
                "premise": sample_premise,
                "hypothesis": sample_hypothesis,
                "n_agents": agent,
                "m_examples": shot,
                "label": sample_label,
                "predicted": predicted,
                "result": result,
            }
            results.append(new_row)
            print(f"Few shot examples: {shot} - Agents: {agent} - Predicted: {predicted} - Result: {result}")
    return results


test_samples = get_dataset(split="test")
test_samples = test_samples.sample(n_test_samples)
dask_df = dd.from_pandas(test_samples, npartitions=4)  # Adjust the number of partitions as needed

delayed_results = [process_sample(row, few_shot, agents) for _, row in dask_df.iterrows()]
results = compute(*delayed_results)

# Flatten the list of results and create a DataFrame
all_results = [item for sublist in results for item in sublist]
df_results = pd.concat([df_results, pd.DataFrame(all_results)], ignore_index=True)

# Debugging: Check if df_results is empty
if df_results.empty:
    print("Warning: df_results is empty. No data to save.")
else:
    # Save results with error handling
    try:
        df_results.to_csv("data/results_ToM.csv", index=False)
        print("Results saved!")
    except Exception as e:
        print(f"Error saving results: {e}")
