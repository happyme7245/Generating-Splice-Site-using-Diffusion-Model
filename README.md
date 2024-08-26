### Generating-Splice-Site-using-Diffusion-Model
#### IRTP project in RC4 related to creating Arabidopsis thaliana DNA sequences with correct splice site using Diffusion Model 

Used model: Diffusion Denoising Probabilistic Model (DDPM)
Generate synthetic DNA sequences using Diffusion Model trained with real dataset.
Dataset overview:
Species: Arabidopsis thaliana

1. SPC acceptor positive (9310)
2. SPC acceptor negative (277255)
3. SPC donor positive (9208)
4. SPC donor negative (263507)

Each sequence length: 402

Evaluation method
1. Train the model using real dataset and test the model with real dataset
2. Train the model using synthetic dataset and test the model with real dataset
3. Train the model using real dataset and test the model with synthetic dataset
4. Train the model using synthetic dataset and test the model with synthetic dataset
