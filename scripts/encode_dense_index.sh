
# cp corpus.jsonl corpusnew.jsonl
# sed -i 's/doc_id/docid/g' corpusnew.jsonl
# You nned to replace the field doc_id with docid in the corpus file. In order to be compatible with the pyserini encoder.

# # python -m pyserini.encode \
# #   input   --corpus ../../TREC-TOT/corpusnew.jsonl \
# #           --fields text \
# #           --delimiter "\n" \
# #           --shard-id 0 \
# #           --shard-num 1 \
# #   output  --embeddings ../output_dense_enc_2 \
# #           --to-faiss \
# #   encoder --encoder /home/ddo/CMU/PLLM/llms-project/dense_models/baseline_distilbert/model \
# #           --fields text \
# #           --batch 32 \
# #           --fp16  # if inference with autocast()

# echo "Start indexing"


# echo "HNSW"
# python -m pyserini.index.faiss \
#   --input ../output_dense_enc_2 \
#   --output ../dense_index_hnsw \
#   --hnsw

# echo "PQ"
# python -m pyserini.index.faiss \
#   --input ../output_dense_enc_2 \
#   --output ../dense_index_pq \
#   --pq

# echo "HNSW + PQ"
# python -m pyserini.index.faiss \
#   --input ../output_dense_enc_2 \
#   --output ../dense_index_hnswpq \
#   --hnsw \
#   --pq

# echo "Flat"
# python -m pyserini.index.faiss \
#   --input ../output_dense_enc_2 \
#   --output ../dense_index_flat

echo "Start inference Sentence Decoder"

python ../dense_inference_decomp_with_rrf.py \
  --index_path /home/ddo/CMU/PLLM/llms-project/dense_index_hnswpq


python ../dense_inference_decomp_with_rrf.py \
  --index_path /home/ddo/CMU/PLLM/llms-project/dense_index_hnsw


python ../dense_inference_decomp_with_rrf.py \
  --index_path /home/ddo/CMU/PLLM/llms-project/dense_index_pq


python ../dense_inference_decomp_with_rrf.py \
  --index_path /home/ddo/CMU/PLLM/llms-project/dense_index_flat

  