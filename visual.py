import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Load the Excel file
file_path = "data/results_ToM.csv"

try:
    # Read the data
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"FileNotFoundError: {file_path} not found. Please check the path.")
    raise

# Ensure the necessary columns exist in the DataFrame
# required_columns = ["n_agents", "m_examples", "result", "premise"]
# if not all(col in df.columns for col in required_columns):
#     raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

df["result"] = df["vote_mean"] == df["label"]

# Group by n_agents and m_examples and calculate mean performance and standard error
pivot_df = (
    df.groupby(["n_agents", "m_examples"])
    .agg(
        performance=("result", "mean"),
        count=("result", "count"),  # Count for standard error calculation
        std_dev=("result", "std"),  # Standard deviation for standard error, ddof=1 by default
    )
    .reset_index()
)

# Calculate standard errors
pivot_df["std_error"] = pivot_df["std_dev"] / np.sqrt(pivot_df["count"])

# Calculate the total number of unique IDs
distinct_ids = df["premise"].nunique()

# Adjustable performance range (set your desired min and max)
performance_range = (0.5, 1)  # Example: restrict performance scale from 0.5 to 1

# Create a meshgrid for plotting
M = pivot_df["m_examples"].unique()
N = pivot_df["n_agents"].unique()
M, N = np.meshgrid(M, N)

# Map the performance and standard errors to the meshgrid
performance_matrix = np.zeros_like(M, dtype=float)
error_matrix = np.zeros_like(M, dtype=float)

for _, row in pivot_df.iterrows():
    i = np.where(N[:, 0] == row["n_agents"])[0][0]
    j = np.where(M[0, :] == row["m_examples"])[0][0]
    performance_matrix[i, j] = row["performance"]
    error_matrix[i, j] = row["std_error"]

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
surf = ax.plot_surface(M, N, performance_matrix, cmap="viridis", alpha=0.8)

# Add error bars (standard errors)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.plot(
            [M[i, j], M[i, j]],
            [N[i, j], N[i, j]],
            [performance_matrix[i, j] - error_matrix[i, j], performance_matrix[i, j] + error_matrix[i, j]],
            color="red",
        )

# Add color bar with adjustable range
cbar = fig.colorbar(surf, shrink=0.5, aspect=10, boundaries=np.linspace(*performance_range, 100))
cbar.set_label("Average Performance")

# Label axes
ax.set_xlabel("M (examples)")
ax.set_ylabel("N (agents)")
ax.set_zlabel("Performance")

# Add a centered title with the number of distinct IDs
plt.title(f"Premise-Hypothesis Classification Examples-Agents (n = {distinct_ids})", loc="center", fontsize=14)

print(pivot_df.head())
print(fig)

# Show plot
plt.show()
