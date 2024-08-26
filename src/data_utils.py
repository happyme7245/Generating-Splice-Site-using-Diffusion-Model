import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

class Utils:
    def __init__(self, ddpm, device):
        self.ddpm = ddpm
        self.device = device

    def max_element(self, tensor):
        # Get the indices of the maximum values in each column
        _, max_indices = torch.max(tensor, dim=0)

        # Create a binary tensor with the same shape as the original tensor
        binary_tensor = torch.zeros_like(tensor)

        # Use the indices to set the maximum values to 1
        cols = torch.arange(tensor.size(1))  # Column indices
        binary_tensor[max_indices, cols] = 1
        return binary_tensor

    def transform_and_combine(self, matrix):
        # Define the nucleotide mapping
        nucleotide_map = {
            0: '',    # For the positions where there is no '1'
            1: 'A',   # First row maps to 'A'
            2: 'T',   # Second row maps to 'T'
            3: 'G',   # Third row maps to 'G'
            4: 'C'    # Fourth row maps to 'C'
        }
        
        # Initialize an empty list to store the combined sequence
        combined_sequence = []
        
        # Iterate over each column in the matrix
        for col in matrix.T:  # Transpose to iterate over columns
            # Find the index of '1' in the row
            for i, value in enumerate(col):
                if value == 1:
                    # Append the corresponding nucleotide to the sequence
                    combined_sequence.append(nucleotide_map[i + 1])
                    break
        
        # Join the list to form the final sequence
        return ''.join(combined_sequence)

    def seq_to_file(self, num_sequences, seq_length, file):
        with open(file, 'w') as writer:
            samples = self.ddpm.sample(seq_length=seq_length, batch_size=num_sequences, device=self.device)
            for sample in samples:
                sample = self.max_element(sample)
                dna_decode = self.transform_and_combine(sample)
                writer.write(f'{dna_decode}\n')
        print(f"Sequences written to {file}")

    # Function to generate predictions for a batch of data
   
    def generate_predictions(self, test_loader, file_path):
        self.ddpm.eval()  # Set the model to evaluation mode

        with open(file_path, 'w') as file, torch.no_grad():  # No need to calculate gradients for testing
            for batch in test_loader:
                batch_data = batch.to(self.device)  # Assuming the test_loader returns tuples (data, labels)
                generated_batch = self.ddpm.sample(seq_length=batch_data.size(-1), batch_size=batch_data.size(0), device=self.device)

                # Post-process the generated batch
                for sample in generated_batch:
                    sample = self.max_element(sample)
                    dna_decode = self.transform_and_combine(sample)
                    file.write(f"{dna_decode}\n")
