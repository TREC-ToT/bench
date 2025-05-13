#!/usr/bin/env python3
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
import click

import ir_datasets
from tqdm import tqdm
import tempfile

# We use the tracker to monitor resource consumption etc. of the indexing and retrieval.
# You can remove it if you do not need it.
from tirex_tracker import tracking

log = logging.getLogger(__name__)

def get_index(ir_dataset, index_directory):
    index_directory = index_directory.resolve().absolute()

    if (
        not index_directory.exists()
        or not (index_directory / "index-ir-metadata.yml").exists()
    ):
        with tempfile.TemporaryDirectory() as docs_dir:
            with open(Path(docs_dir) / "docs.jsonl", "w") as f:
                for raw_doc in tqdm(ir_dataset.docs_iter(), "reformat documents"):
                    f.write(json.dumps({"id": raw_doc.doc_id, "contents": raw_doc.default_text()}) + "\n")

            # call pyserini indexer
            cmd = f"""java -cp /app/anserini.jar io.anserini.index.IndexCollection -collection JsonCollection -input {docs_dir} -index {index_directory} -generator DefaultLuceneDocumentGenerator -threads 1 """.split()

            with tracking(export_file_path=index_directory / "index-ir-metadata.yml"):
                subprocess.check_output(cmd)
    return index_directory


def create_run(index_directory, dataset, output_file):

    if not output_file.exists():
        with tempfile.TemporaryDirectory() as queries_dir:
            with open(Path(queries_dir) / "queries.jsonl", "w") as f:
                for i in dataset.queries_iter():
                    f.write(json.dumps({"id": i.query_id, "query": i.default_text()}) + '\n')

                cmd = f"""java -cp /app/anserini.jar io.anserini.search.SearchCollection -index {index_directory} -bm25 -topics {queries_dir}/queries.jsonl -topicReader JsonString -output {output_file} -threads 1 """.split()

                with tracking(export_file_path=output_file.parent / "retrieval-ir-metadata.yml"):
                    subprocess.check_output(cmd)


@click.command()
@click.option("--dataset", type=str, required=True, help="The dataset id in ir_datasets (might be from an ir_datasets extension).")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(dataset, output, index):
    ir_dataset = ir_datasets.load(dataset)
    index = get_index(ir_dataset, index)
    create_run(index, ir_dataset, output)

if __name__ == "__main__":
    main()

