# H-IR-BERT V0.1

科研原型：确定性超图编译 + 随机初始化的 Hugging Face `BertModel` + 节点分类监督。
不下载预训练权重，不包含自监督预训练。

## H-IR

实现见 `hir.py`，入口为 `compile_hypergraph(N, hyperedge_index)` 和
`parse_hir(tokens, N, M)`。编译不读取节点特征。

节点从 0 编号。语法 Token ID 为：

| 名称 | ID |
| --- | --- |
| PAD / EDGE / END_EDGE | 0 / 1 / 2 |
| DIRECT / EXTEND / PATCH | 3 / 4 / 5 |
| REF / DROP / ADD | 6 / 7 / 8 |
| FRAGMENT（仅供 BERT 分块） | 9 |
| NODE / EDGE_ID / ORDER（BERT 中的数字类型） | 10 / 11 / 12 |

逻辑序列中，节点 `v` 编码为 `13+v`，超边编号 `i` 编码为 `13+N+i`，
阶数 `k` 编码为 `13+N+M+k`。三个数字区间互不重叠，`M` 是超边数。

```text
EDGE edge_id order DIRECT node... END_EDGE
EDGE edge_id order EXTEND REF previous_edge_id ADD node... END_EDGE
EDGE edge_id order PATCH REF previous_edge_id DROP node... ADD node... END_EDGE
```

每条边成员去重并升序排列，边按 `(阶数, 成员字典序)` 排序，包含 0/1 阶边。
重复关联按二元成员关系去重；不同列的相同超边保留多重性。
默认每个出现的源超边 ID 对应一条边，支持非连续 ID；若需要声明空列，传入
`num_hyperedges=M`，此时源 ID 必须在 `[0,M)`，未出现的列作为空边保留。
孤立节点由 `N` 保留。`program.source_edge_ids` 给出规范边到源 ID 的映射。

只搜索已定义且有交集的边。EXTEND 要求严格低阶子集；PATCH 使用准确的集合差。
实际 Token 代价分别为 `5+k`、`8+增加数`、`9+删除数+增加数`。
同代价依次优先 DIRECT、EXTEND、PATCH，再选择最小引用编号；引用必须严格省 Token。
Parser 对非法引用、节点、操作、阶数和不完整语法抛出 `ValueError`。
`parsed.to_hyperedge_index()` 恢复关联索引，`parsed.incidence_matrix()` 恢复稀疏关联矩阵。

## BERT 与长序列

Compiler 始终生成完整逻辑 H-IR。每个 BERT 块最多 `hir_max_seq_length` 个 Token：
块内缺失的引用基边以 DIRECT 定义复制到前面，复制的上下文不额外产生输出边。
若定义和目标无法同时放入一块，目标展开为 DIRECT。超长边拆成若干
`EDGE edge_id full_order FRAGMENT members... END_EDGE`，每个成员恰好进入一个片段；
片段 EDGE 状态按成员数加权平均得到该边表示。没有截断。

输入为结构类型 Embedding、固定正余弦数字身份编码、`Linear(log(1+阶数))`，
再在节点位置加上 `Linear(data.x)`；BERT 添加位置编码。
不为每个节点建立可学习 ID 表。使用 [Hugging Face BERT 的 inputs_embeds 接口](https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/bert#transformers.BertModel.forward)。

EDGE 隐状态形成超边表示，按原始成员关联求均值汇聚到节点，与节点特征投影拼接后
经 MLP 输出 `[N, num_classes]`；接口为 `model(data) -> (logits, None)`。
孤立节点使用特征分支。只缓存编译结果，不缓存神经网络输出。
按 `hir_chunk_batch_size` 批量编码，默认用 PyTorch activation checkpoint 重算
BERT，避免所有块的注意力激活同时驻留。Attention 的单次规模受批大小和块长度限制。

## 依赖、测试与训练

使用原有 DHG-Bench 环境。额外依赖可安装为：

```bash
python -m pip install "transformers==4.44.2" "tokenizers==0.19.1"
```

本次验证环境为 CPU PyTorch 2.1.0 / Python 3.10，使用上述 transformers 版本。
较新的 transformers 可能与旧 PyTorch 不兼容；无需因此升级 PyTorch/CUDA。
此依赖仅在实例化 HIRBERT 时导入，其他方法不需要 transformers。

从仓库根目录执行轻量测试（无需数据集或 GPU）：

```bash
python -m unittest discover -s tests -p "test_hir*.py" -v
```

准备 README 所述 Cora 数据后，从仓库根目录进入 `dhgbench`：

```bash
cd dhgbench
python main.py --method=HIRBERT --task_type=node_cls --dname=cora --is_default=True --num_seeds=1 --seed_offset=0 --device=cuda:0 --epochs=100 --results_dir=./results
```

这里提供 GPU 命令，V0.1 实现验证只运行 CPU 合成小图；也可用 `--device=cpu`。
默认 BERT 为 256 hidden、4 层、4 heads、1024 intermediate、0.2 dropout、256 长度。
配置见 `lib_yamls/node_yamls/config_hirbert.yaml`；`--hir_hidden_size`、
`--hir_num_hidden_layers`、`--hir_num_attention_heads`、`--hir_intermediate_size`、
`--hir_dropout`、`--hir_max_seq_length`、`--hir_chunk_batch_size` 均可覆盖。
HIRBERT 的显式 CLI 参数优先于 YAML，其他方法配置规则不变。

HIRBERT 跳过自动自环增强，真实单节点边保留；使用原有随机种子、masks、NLLLoss 和
节点分类协议。每个种子重新初始化所有模型参数。

## 结果与重载

原有 `results/node_cls_runs.csv` 和 `node_cls_summary.csv` 保留。
另外每个种子保存到 `results/hirbert/<dataset>/<run_id>/seed_<seed>/`：

- `training.csv`：逐 epoch Train/Val Loss、Train/Val/Test Accuracy。
- `metrics.json`：最终分割指标、种子、训练时间、参数量和配置。
- `compilation.json`：真实边数、逻辑 Token 数、引用比例、各阶/模式数量、实际 BERT Token 数、块数、最大块长、展开/片段统计。
- `checkpoint.pt`：最终参数、模型配置、图特征/标签/关联及原训练 masks，可独立重载推理。

Accuracy 与原结果 CSV 一致，使用百分数（0–100）；Loss 为原节点分类 NLLLoss。

在 `dhgbench` 目录中重载：

```python
import torch
from lib_models.HNN.hirbert import HIRBERT

model, data = HIRBERT.from_checkpoint("results/hirbert/.../checkpoint.pt", device="cpu")
with torch.no_grad():
    logits, _ = model(data)
```

`--save_results=False` 关闭 CSV 与上述文件写入。最终 checkpoint 对应最后一个 epoch。
