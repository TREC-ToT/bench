# Code and Baselines for the 2025 edition of TREC-ToT

We provide a set of baselines with pre-computed indices and runs:

- Lexical retrieval:
  - [anserini-bm25-retrieval](anserini-bm25-retrieval): Lexical retrieval with [Anserini](https://github.com/castorini/anserini).
  - [chatnoir-retrieval](chatnoir-retrieval): Still in progress
  - [pyterrier-bm25-retrieval](pyterrier-bm25-retrieval): Lexical retrieval with [PyTerrier](https://github.com/terrier-org/pyterrier).
- Dense retrieval:
  - [lightning-dense-retrieval](lightning-dense-retrieval): Dense retrieval with the model [sbhargav/baseline-distilbert-tot24](https://huggingface.co/sbhargav/baseline-distilbert-tot24) implemented in [Lightning IR](https://github.com/webis-de/lightning-ir).

Indices are available at HuggingFace: [https://huggingface.co/datasets/webis/TREC-ToT-Baselines](https://huggingface.co/datasets/webis/TREC-ToT-Baselines).
