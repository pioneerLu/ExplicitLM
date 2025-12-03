# SFT 训练指南：使用 OMCQ 数据训练知识融合模块

本文档说明如何使用 OMCQ 数据对 ExplicitLM 的知识融合模块进行监督微调（SFT）。

## 概述

SFT 训练的目标是：
- **冻结 Qwen3 主模型参数**：保持预训练模型的知识
- **只训练知识融合模块**：包括 `memory_gate`、`gated_memory_fusion`、`memory_norm`
- **使用 OMCQ 数据**：约 157 万条多选题数据，训练模型如何利用记忆库回答问题

## 步骤 1：数据转换

将 OMCQ 数据转换为对话格式：

```bash
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

# 转换数据（测试模式，只转换 10 条）
python3 scripts/convert_omcq_to_sft.py \
    --input sft_data/omcq_trex_data.json \
    --output sft_data/omcq_trex_sft.jsonl \
    --max-samples 10

# 转换全部数据（约 157 万条）
python3 scripts/convert_omcq_to_sft.py \
    --input sft_data/omcq_trex_data.json \
    --output sft_data/omcq_trex_sft.jsonl
```

转换后的数据格式：
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "What is Austroasiatic languages an instance of?\nA:language family,B:pteridosperms,C:FIBT World Championships\n请选择正确答案。"
    },
    {
      "role": "assistant",
      "content": "A:language family"
    }
  ]
}
```

## 步骤 2：验证数据加载

测试转换后的数据能否正确加载：

```bash
python3 scripts/test_sft_data.py \
    --data-path sft_data/omcq_trex_sft.jsonl \
    --qwen3-model-path /path/to/Qwen3-4b
```

## 步骤 3：配置训练参数

### 3.1 更新数据集配置

编辑 `config/dataset.py`，设置 SFT 数据路径：

```python
DatasetConf = builds(
    dict,
    # ... 其他配置 ...
    # ---- sft 相关字段 ----
    pretrained_sft_model_path="out/pretrain_latest.pth",  # 预训练模型路径
    sft_dataset_path="sft_data/omcq_trex_sft.jsonl",      # SFT 训练数据
    sft_val_dataset_path="data/benchmarks/eval_data.json", # SFT 验证数据
)
```

### 3.2 更新模型配置

编辑 `config/model.py`，确保：
- 使用预训练的 cache 知识库
- 参数冻结已启用（在 `model_initializer.py` 中自动处理）

```python
ModelConf = builds(
    dict,
    # ... 其他配置 ...
    cache_path="data/cache/knowledge_cache.pt",  # 使用预训练的 cache
    recompute_cache=False,                        # 不重新计算
    use_ema_update=False,                         # SFT 时通常不使用 EMA
)
```

### 3.3 更新训练配置

编辑 `config/training.py`，设置 SFT 训练超参数：

```python
TrainingConf = builds(
    dict,
    batch_size=4,                    # 批次大小
    accumulation_steps=32,           # 梯度累积步数
    epochs=3,                        # 训练轮数
    learning_rate=5e-5,              # 学习率（SFT 通常较小）
    # ... 其他配置 ...
)
```

**推荐配置**：
- **学习率**：`5e-5` 到 `1e-4`（知识融合模块通常需要较小的学习率）
- **批次大小**：根据 GPU 显存调整（4-8）
- **梯度累积**：保持有效批次大小在 128-256
- **训练轮数**：1-3 轮（SFT 通常不需要太多轮次）

## 步骤 4：启动训练

### 4.1 使用命令行参数覆盖配置

```bash
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

python3 2_sft.py \
    model.qwen3_model_path=/path/to/Qwen3-4b \
    model.cache_path=data/cache/knowledge_cache.pt \
    model.recompute_cache=False \
    dataset.sft_dataset_path=sft_data/omcq_trex_sft.jsonl \
    dataset.pretrained_sft_model_path=out/pretrain_latest.pth \
    training.learning_rate=5e-5 \
    training.batch_size=4 \
    training.epochs=3 \
    training.accumulation_steps=32
```

### 4.2 使用配置文件（推荐）

创建配置文件 `config/sft_omcq.yaml`：

```yaml
defaults:
  - model
  - dataset
  - training
  - logging

model:
  qwen3_model_path: /path/to/Qwen3-4b
  cache_path: data/cache/knowledge_cache.pt
  recompute_cache: false

dataset:
  sft_dataset_path: sft_data/omcq_trex_sft.jsonl
  pretrained_sft_model_path: out/pretrain_latest.pth

training:
  learning_rate: 5e-5
  batch_size: 4
  epochs: 3
  accumulation_steps: 32
```

然后运行：
```bash
python3 2_sft.py --config-name sft_omcq
```

## 步骤 5：监控训练

训练过程中会：
1. **自动冻结 Qwen3 主模型参数**：只训练知识融合模块
2. **显示可训练参数统计**：确认只有记忆相关组件在训练
3. **记录训练损失**：通过 SwanLab 可视化（如果启用）

### 参数冻结验证

训练开始时会输出：
```
🔒 冻结Qwen主模型参数...
✅ 参数冻结完成:
  - 冻结参数: XXXX.XXX 百万
  - 可训练参数: XX.XXX 百万
  - 冻结比例: XX.XX%
```

确认可训练参数数量合理（通常只有几百万参数，主要是知识融合模块）。

## 步骤 6：验证训练结果

训练完成后，使用 `examples/quick_start.py` 测试生成效果：

```python
# 加载训练后的模型
args = {
    'qwen3_model_path': '/path/to/Qwen3-4b',
    'cache_path': 'data/cache/knowledge_cache.pt',
    # ... 其他配置 ...
}

model, tokenizer = init_model(args)
# 加载 SFT 后的权重
model.load_state_dict(torch.load('out/sft_latest.pth'))

# 测试生成
# ...
```

## 常见问题

### Q1: 训练时显存不足？

**解决方案**：
- 减小 `batch_size`
- 增加 `accumulation_steps` 保持有效批次大小
- 减小 `max_seq_len`（在 `config/model.py` 中）

### Q2: 训练损失不下降？

**可能原因**：
- 学习率过大或过小：尝试调整 `learning_rate`（1e-5 到 1e-4）
- 数据格式问题：检查数据是否正确转换
- 参数冻结问题：确认只有知识融合模块在训练

### Q3: 如何只训练部分知识融合模块？

修改 `utils/model_initializer.py` 中的 `_freeze_qwen_params` 函数，调整 `is_memory_component` 的判断逻辑。

### Q4: 训练后生成效果没有改善？

**可能原因**：
- 训练轮数不足：尝试增加 `epochs`
- 学习率不合适：尝试不同的学习率
- 数据质量问题：检查数据转换是否正确
- 需要更多训练数据：考虑使用更多数据

## 下一步

训练完成后，可以：
1. 评估模型在验证集上的表现
2. 测试生成质量（使用 `examples/quick_start.py`）
3. 调整超参数并重新训练
4. 使用训练后的模型进行推理

