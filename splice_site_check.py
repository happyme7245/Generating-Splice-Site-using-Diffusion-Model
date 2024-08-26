import os


# define path, list, and count
count = 5
seq_list = []
file_path = os.path.join('SYNTHETIC_dataset', f'synthetic_{count}.txt')

with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line[200] == 'A' and line[201] == 'G':
            seq_list.append(line)

# correct splice site: AG in the middle
print(f'Number of DNA sequence with correct splice site: {len(seq_list)}')

file_path_write = os.path.join('SYNTHETIC_dataset', f'only_with_correct_splice_site_{count}')
with open(file_path_write, 'w') as writer:
    for seq in seq_list:
        writer.write(seq)
