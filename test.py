import numpy as np
rng = np.random.RandomState(23455)
embeddings_size = 30
unknown = rng.normal(size=(embeddings_size))
padding = rng.normal(size=(embeddings_size))

print unknown

