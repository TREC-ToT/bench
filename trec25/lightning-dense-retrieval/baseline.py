#!/usr/bin/env python3
import click
from pathlib import Path
from tirex_tracker import tracking
from lightning_ir import (
    BiEncoderModule,
    DocDataset,
    IndexCallback,
    LightningIRDataModule,
    LightningIRTrainer,
    TorchDenseIndexConfig,
    QueryDataset,
    SearchCallback,
    TorchDenseSearchConfig,
    DprConfig
)

def get_index(data_module, bi_encoder, index_directory):
    index_directory = index_directory.resolve().absolute()

    if not (index_directory / "index-ir-metadata.yml").exists():        
        with tracking(export_file_path=index_directory / "index-ir-metadata.yml"):
            callback = IndexCallback(index_dir=index_directory, index_config=TorchDenseIndexConfig())
            trainer = LightningIRTrainer(callbacks=[callback], logger=False, enable_checkpointing=False)

            trainer.index(bi_encoder, data_module)

    return index_directory

def create_run(index_directory, data_module, bi_encoder, output_file):
    callback = SearchCallback(index_dir=index_directory, search_config=TorchDenseSearchConfig(k=1000), save_dir=output_file)
    trainer = LightningIRTrainer(callbacks=[callback], logger=False, enable_checkpointing=False)
    if not (output_file / "retrieval-ir-metadata.yml").exists():        
        with tracking(export_file_path=output_file / "retrieval-ir-metadata.yml"):
            trainer.search(bi_encoder, data_module)

@click.command()
@click.option("--dataset", type=str, required=True, help="The dataset id in ir_datasets (might be from an ir_datasets extension).")
@click.option("--model_name_or_path", type=str, default="sbhargav/baseline-distilbert-tot24", required=False, help="The Bi-Encoder model.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(dataset, output, index, model_name_or_path):
    bi_encoder = BiEncoderModule(model_name_or_path=model_name_or_path, config=DprConfig(projection=None, query_pooling_strategy="mean", doc_pooling_strategy="mean", similarity_function="cosine"))

    data_module = LightningIRDataModule(inference_datasets=[DocDataset(dataset)], inference_batch_size=1024)

    index = get_index(data_module, bi_encoder, index)

    data_module = LightningIRDataModule(inference_datasets=[QueryDataset(dataset)], inference_batch_size=32)
    create_run(index, data_module, bi_encoder, output)

if __name__ == "__main__":
    main()
