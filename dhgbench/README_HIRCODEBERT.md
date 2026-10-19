# HIR-CodeBERT v1

HIR-CodeBERT keeps the existing H-IR compiler, parser, canonical ordering and
FRAGMENT lowering. It derives semantic relations from each `StructureInput`
view, so masked memberships and masked H-IR fields are never recovered from the
logical program behind that view.

The encoder has two local instruction Transformer layers followed by two
semantic instruction Transformer layers. Visible instruction fragments and
context copies are locally encoded and pooled by canonical edge identity. The
instruction Transformer uses typed biases for REF direction, shared visible
nodes, matching opcodes, and shared DIRECT/ADD/DROP operands. NODE token states
are fused with their updated owning-instruction state before shared attention
pooling.

Full five-graph pretraining and paired Cora scratch/pretrained evaluation:

```bash
cd /home/wh/HGNN/dhgbench && python btest.py --hircodebert --mode=all --device=cuda:3 --dname=cora --pretrain_datasets cora citeseer coauthor_cora NTU2012 house-committees-100 --pretrain_seed=0 --local_samples_per_graph=128 --validation_samples_per_graph=64 --local_min_edges=64 --local_max_edges=128 --local_hops=3 --local_max_nodes=2048 --hir_hidden_size=256 --hir_num_attention_heads=4 --hir_num_hidden_layers=4 --hir_local_num_hidden_layers=2 --hir_semantic_num_hidden_layers=2 --hir_max_seq_length=512 --hir_chunk_batch_size=16 --hir_gradient_checkpointing=true --micro_batch_size=8 --gradient_accumulation_steps=4 --bert_max_batch_tokens=32768 --amp=bf16 --gpu_memory_budget_gb=18 --gpu_target_memory_gb=15 --pretrain_epochs=200 --pretrain_patience=50 --epochs=500 --patience=100 --num_seeds=5 --seed_offset=0 --variants scratch pretrained --output=./results/hircodebert_v1
```
