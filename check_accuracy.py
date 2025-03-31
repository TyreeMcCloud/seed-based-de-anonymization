import pandas as pd
import os

# Function to read the solution and seed mapping files
def read_mapping(file_path):
    mapping = pd.read_csv(file_path, sep=" ", header=None, names=["G1", "G2"])
    return dict(zip(mapping["G1"], mapping["G2"]))

# Function to calculate accuracy
def calculate_accuracy(predicted_mapping, actual_mapping):
    correct_matches = 0
    total_matches = len(actual_mapping)
    
    for g1_node, g2_node in actual_mapping.items():
        # Check if the predicted mapping for g1_node matches the actual g2_node
        if predicted_mapping.get(g1_node) == g2_node:
            correct_matches += 1
    
    accuracy = correct_matches / total_matches * 100
    return accuracy

# Define the file paths
solution_file = "Solution2.txt"
seed_mapping_test_file = os.path.join("data", "seed_mapping.txt")

# Load the predicted mapping from solution.txt
predicted_mapping = read_mapping(solution_file)

# Load the actual seed mapping from seed_mapping_test.txt
actual_mapping = read_mapping(seed_mapping_test_file)

# Calculate accuracy
accuracy = calculate_accuracy(predicted_mapping, actual_mapping)

# Print the accuracy
print(f"Accuracy for the seed mapping test: {accuracy:.2f}%")
