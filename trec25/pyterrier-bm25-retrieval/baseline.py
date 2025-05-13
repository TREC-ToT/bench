#!/usr/bin/env python3
from pathlib import Path

import click
import pandas as pd
import pyterrier as pt
import ir_datasets

# We use the tracker to monitor resource consumption etc. of the indexing and retrieval.
# You can remove it if you do not need it.
from tirex_tracker import tracking


def get_index(ir_dataset, index_directory):
    # PyTerrier needs an absolute path
    index_directory = index_directory.resolve().absolute()

    if (
        not index_directory.exists()
        or not (index_directory / "index-ir-metadata.yml").exists()
    ):
        with tracking(export_file_path=index_directory / "index-ir-metadata.yml"):
            # build the index
            indexer = pt.IterDictIndexer(
                str(index_directory), overwrite=True, meta={"docno": 100, "text": 20480}
            )

            # you can do some custom document processing here
            docs = (
                {"docno": i.doc_id, "text": i.default_text()}
                for i in ir_dataset.docs_iter()
            )
            indexer.index(docs)

    return pt.IndexFactory.of(str(index_directory))


def process_dataset(ir_dataset, index_directory, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    index = get_index(ir_dataset, index_directory)
    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        bm25 = pt.terrier.Retriever(index, wmodel="BM25")

        # potentially do some query processing
        topics = pd.DataFrame(
            [
                {"qid": i.query_id, "query": i.default_text()}
                for i in ir_dataset.queries_iter()
            ]
        )

        # PyTerrier needs to use pre-tokenized queries
        tokeniser = pt.java.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()

        topics["query"] = topics["query"].apply(
            lambda i: " ".join(tokeniser.getTokens(i))
        )

        run = bm25(topics)
        pt.io.write_results(run, output_directory / "run.txt.gz")


@click.command()
@click.option("--dataset", type=str, help="The dataset id in ir_datasets (might be from an ir_datasets extension).")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(dataset, output, index):
    ir_dataset = ir_datasets.load(dataset)
    process_dataset(ir_dataset, index, Path(output))


if __name__ == "__main__":
    main()
