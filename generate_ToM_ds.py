
import ToM
import pandas as pd


try:
# Load file if exists
    df_results = pd.read_excel('data/results_ToM.xlsx')
except:
    df_results = pd.DataFrame(columns=['premise', 'hypothesis', 'label', 'predicted', 'result'])


# Config 
few_shot = [1, 4, 8, 16, 32]
agents = [1, 3, 5, 9]

# The batch size of new samples
new_n_samples = 10

# Add new results
new_samples = ToM.get_n_new_premises(new_n_samples, df_results)

count = 0

for i, sample in new_samples.iterrows():

    sample_premise = sample['premise']
    sample_hypothesis = sample['hypothesis']
    sample_label = sample['label']

    count += 1
    print(f"Classification for {count}/{new_n_samples} premises")

    # Testing M few shot examples and N agents examples
    for shot in few_shot:
        for agent in agents:

            # Get the prediction with combination of agents and few shot examples
            predicted, result = ToM.get_premise_class(sample_hypothesis, sample_premise, sample_label, agent, shot)


            # Create a new row as a DataFrame
            new_row = pd.DataFrame({'premise': [sample_premise],
                                    'hypothesis': [sample_hypothesis], 
                                    'n_agents': [agent], 
                                    'm_examples': [shot], 
                                    'label': [sample_label], 
                                    'predicted': [predicted], 
                                    'result': [result]})

            # Append the new row
            df_results = pd.concat([df_results, new_row], ignore_index=True)

            print(f"Few shot examples: {shot} - Agents: {agent} - Predicted: {predicted} - Result: {result}")

# Save results
df_results.to_excel('data/results_ToM.xlsx', index=False)

print("Results saved!")