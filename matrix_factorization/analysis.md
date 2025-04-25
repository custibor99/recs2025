## Logistic matric factorization

[Logistic matrix factorization](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf) works on binary interaction matrices and instead of optimizing the mean squared error, it changes the problem to a classification problem and minimized the negative log loss. 

The algorithm was run on the following interaction matrices:
- add_to_cart
- purchase
- remove from cart

All of the above interaction matrices were able to be reconstructed with a high rocauc on the validation dataset (99%) with a small latent dimension size (32). But when using user embeddings for the downstream prediction task such as churn the ROCAUC remained low, reaching a maximum of 51.31%.