
# General

(more documentation will be added)

My approach was to generate the embeddings using SVD. I reuse code from the baseline approach and only change the data loading and embedding generation section.

Generate embeddings like this (only change the path to the script:

~~~
uv run python -m leon.create_embeddings --data-dir data/ --embeddings-dir embeddings/
~~~

Naturally, all the other commands work the same way after generating the embeddings using the SVD approach.

## Notes

There is still some optimization which can be done with this approach, like finding a good value for the EMBEDDING_DIM value. It's also rather slow.

