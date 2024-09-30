import numpy as np
import matplotlib.pyplot as plt

# Load the .npy files
freq_matrix = np.load('freq_matrix.npy')
trace_matrix_baseline = np.load('trace_matrix_baseline.npy')

# Verify the shapes
print("freq_matrix shape:", freq_matrix.shape)
print("trace_matrix_baseline shape:", trace_matrix_baseline.shape)

# Extract the first rows
freq_first_row = freq_matrix[0, :]
trace_first_row = trace_matrix_baseline[0, :]

# Ensure the first rows are compatible
if freq_first_row.shape[0] != trace_first_row.shape[0]:
    print("Error: The first rows of freq_matrix and trace_matrix_baseline have different lengths.")
    print(f"Length of freq_first_row: {freq_first_row.shape[0]}")
    print(f"Length of trace_first_row: {trace_first_row.shape[0]}")
else:
    # Plot trace_matrix_baseline as a function of freq_matrix
    plt.figure(figsize=(10, 6))
    plt.plot(freq_first_row, trace_first_row, linestyle='-')
    plt.title('trace_matrix_baseline First Row vs. freq_matrix First Row')
    plt.xlabel('freq_matrix First Row Values')
    plt.ylabel('trace_matrix_baseline First Row Values')
    plt.grid(True)
    plt.show()

# Save the new dependence as .npy file
# np.save('./comb_frequencies.npy', freq_first_row)
# np.save('./comb_amplitudes.npy', trace_first_row)