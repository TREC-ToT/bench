# Code and Baselines for the 2025 edition of TREC-ToT

We provide a set of baselines with pre-computed indices and runs:

- Lexical retrieval:
  - [anserini-bm25-retrieval](anserini-bm25-retrieval): Lexical retrieval with [Anserini](https://github.com/castorini/anserini).
  - [chatnoir-retrieval](chatnoir-retrieval): Still in progress
  - [pyterrier-bm25-retrieval](pyterrier-bm25-retrieval): Lexical retrieval with [PyTerrier](https://github.com/terrier-org/pyterrier).
- Dense retrieval:
  - [lightning-dense-retrieval](lightning-dense-retrieval): Dense retrieval with the model [sbhargav/baseline-distilbert-tot24](https://huggingface.co/sbhargav/baseline-distilbert-tot24) implemented in [Lightning IR](https://github.com/webis-de/lightning-ir).

Indices are available at HuggingFace: [https://huggingface.co/datasets/webis/TREC-ToT-Baselines](https://huggingface.co/datasets/webis/TREC-ToT-Baselines).

## Prepared Indices

The indices for our baselines are publicly available, so that you can directly re-use them:

| Index | Framework |Size | md5 |
|-------|-----------|-----|-----|
|[trec-tot-2025-anserini-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-anserini-index.zip) | Anserini |1.7GB | b04afdf33519013bf08857005a6cbd88|
|[trec-tot-2025-pyterrier-index.zip](https://files.webis.de/data-in-progress/trec-tot-2025-indices/trec-tot-2025-pyterrier-index.zip) | PyTerrier | 11GB | a9a22ed35abb6cea842a7c5734987c82 |
|trec-tot-2025-dense-index|Lightning IR|**TODO**|**TODO**|
