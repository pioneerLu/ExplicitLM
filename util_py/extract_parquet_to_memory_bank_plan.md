# Parquet数据提取到Memory Bank执行方案

## 一、目录结构

```
data/
├── cache/
│   └── parquet_extract/          # 新建目录，存储中间结果
│       ├── extracted_facts/      # 每个parquet文件的提取结果
│       │   ├── 000001_facts.jsonl
│       │   ├── 000002_facts.jsonl
│       │   └── ...
│       ├── memory_bank_batches/   # 每满1,048,576条保存一次
│       │   ├── memory_bank_batch_0.pt
│       │   ├── memory_bank_batch_0_mapping.json
│       │   ├── memory_bank_batch_1.pt
│       │   └── ...
│       └── stats.json            # 统计信息
└── parquet_data/
    └── 256/
        ├── 000001.parquet
        └── ...
```

## 二、执行流程

### 阶段1：并行提取（4进程/4卡）

**目标**：从所有parquet文件中提取facts并保存

**步骤**：
1. 扫描所有parquet文件，分配给4个进程
2. 每个进程：
   - 加载LLMLingua模型（如果支持GPU，分配到不同GPU）
   - 处理分配的文件批次
   - 对每个text：
     - LLMLingua提取（compression_rate=0.4）
     - 按句子分割
     - Tokenize并过滤（长度检查）
   - 保存到 `extracted_facts/{filename}_facts.jsonl`

**输出格式（每行一个fact）**：
```json
{
    "fact_text": "压缩后的句子",
    "token_ids": [123, 456, ...],  // 已tokenize，长度<=32
    "source_file": "000001.parquet",
    "source_uuid": "<urn:uuid:...>",
    "source_index": 12345,
    "compression_ratio": 0.4
}
```

### 阶段2：合并、去重、采样

**目标**：从所有提取的facts中选择1,048,576条

**步骤**：
1. 读取所有 `extracted_facts/*.jsonl` 文件
2. 去重策略：
   - 使用 `hash(tuple(token_ids))` 作为唯一标识
   - 或使用文本hash（更快）
3. 采样策略：
   - 如果去重后 > 1,048,576：随机采样
   - 如果 < 1,048,576：全部保留
4. 按顺序填充到1,048,576条（不足用pad填充）

### 阶段3：分批保存

**目标**：每满1,048,576条保存一次

**步骤**：
1. 维护一个缓冲区，累积facts
2. 当缓冲区达到1,048,576条时：
   - Tokenize并处理（截断/填充到32）
   - 生成 `memory_bank` tensor `[1048576, 32]`
   - 生成 `valid_mask` tensor `[1048576]`
   - 保存为 `.pt` 文件
   - 保存 `_mapping.json`
   - 清空缓冲区
3. 继续处理，直到所有facts处理完

## 三、技术细节

### 1. 多进程/多卡分配

```python
# 方案A：使用multiprocessing（CPU）
from multiprocessing import Process, Queue
# 4个进程，每个处理一批文件

# 方案B：使用torch.distributed（GPU）
# 4个GPU，每个处理一批文件
```

### 2. 去重方法

**方案A：基于token IDs（精确）**
```python
seen = set()
for fact in facts:
    token_hash = hash(tuple(fact['token_ids']))
    if token_hash not in seen:
        seen.add(token_hash)
        unique_facts.append(fact)
```

**方案B：基于文本hash（快速）**
```python
seen = set()
for fact in facts:
    text_hash = hash(fact['fact_text'].lower().strip())
    if text_hash not in seen:
        seen.add(text_hash)
        unique_facts.append(fact)
```

**推荐**：使用方案B（文本hash），因为：
- 更快（不需要先tokenize）
- 对于相似文本也能去重
- 如果tokenize后相同，文本通常也相同

### 3. 内存优化

- 流式读取parquet文件（不一次性加载全部）
- 分批处理facts（每批10万条）
- 及时释放中间变量

### 4. 错误处理

- LLMLingua失败：使用原始文本的前32个tokens
- 文件读取失败：跳过，记录日志
- Tokenize失败：跳过该fact

## 四、实现脚本结构

```
util_py/
└── extract_parquet_to_memory_bank.py
    ├── extract_facts_from_parquet()  # 阶段1：提取
    ├── merge_and_deduplicate()      # 阶段2：合并去重
    ├── save_memory_bank_batch()      # 阶段3：分批保存
    └── main()                        # 主流程
```

## 五、配置参数

```python
CONFIG = {
    "parquet_dir": "data/parquet_data/256",
    "output_dir": "data/cache/parquet_extract",
    "knowledge_num": 1024 * 1024,  # 1,048,576
    "knowledge_length": 32,
    "compression_rate": 0.4,
    "num_workers": 4,  # 4进程/4卡
    "batch_size": 1048576,  # 每批保存的条数
    "min_fact_length": 10,  # 最小fact长度（字符）
    "deduplication": "text_hash",  # "text_hash" 或 "token_hash"
}
```

## 六、执行命令

```bash
# 单机多进程
python util_py/extract_parquet_to_memory_bank.py \
    --parquet-dir data/parquet_data/256 \
    --output-dir data/cache/parquet_extract \
    --num-workers 4 \
    --compression-rate 0.4

# 或使用torch.distributed（多卡）
torchrun --nproc_per_node=4 util_py/extract_parquet_to_memory_bank.py ...
```

