# ExplicitLM 项目功能总览

本文档整理了 ExplicitLM 项目的所有功能模块。

## 一、训练功能

### 1.1 Pretrain 数据训练（预训练格式）
- **脚本**: `train_pretrain.py`
- **启动脚本**: `scripts/run_fusion_pretrain.sh`
- **功能**: 
  - 基于 Qwen3-4B 预训练模型
  - 只训练 Fusion 组件（GatedMemoryFusion 和 memory_norm）
  - 冻结 MemoryGate、Backbone、MemoryBank
  - 使用预训练格式数据（纯文本，Parquet 格式）
  - 支持 Memory Bank 动态更新（通过 MemoryBankUpdater）
  - 支持分布式训练和混合精度

### 1.2 SFT 数据训练（对话格式）
- **脚本**: `train_sft.py`
- **启动脚本**: `scripts/run_sft.sh`
- **功能**:
  - 基于 Qwen3-4B 预训练模型
  - 只训练 Fusion 组件（GatedMemoryFusion 和 memory_norm）
  - 冻结 MemoryGate（包括 keys）、Backbone、MemoryBank
  - 使用对话格式数据训练（SFT 格式，JSONL 或 Parquet）
  - 支持分布式训练和混合精度（DeepSpeed ZeRO Stage 2）
  - 支持加载预训练的 Router 和 Fusion 权重

## 二、模型转换功能

### 2.1 Checkpoint 转 HuggingFace 格式（独立存储版本）
- **脚本**: `pt2hg_apart.py`
- **功能**:
  - 将训练好的 checkpoint 转换为 HuggingFace 格式
  - Memory Bank 独立存储（`memory_bank.pt`），便于替换/裁剪
  - 生成自包含可分发模型（包含推理脚本和工具）
  - 自动生成 `inference_example.py` 和 `switch_memory_bank.py`
  - 支持指定 Memory Bank 和 Keys 文件
  - **推荐使用**

### 2.2 Checkpoint 转 HuggingFace 格式（同步存储版本）
- **脚本**: `pt2hg_sync.py`
- **功能**:
  - 将训练好的 checkpoint 转换为 HuggingFace 格式
  - Memory Bank 写入权重文件（单目录）
  - 适合单目录部署场景

## 三、推理功能

### 3.1 对话式推理
- **脚本**: `chat_example.py`
- **功能**:
  - 加载训练好的 ExplicitLM 模型
  - 命令行交互式对话
  - 支持多轮对话

### 3.2 推理示例（转换后模型）
- **脚本**: `my_model/inference_example.py` 或 `hf_explicitlm_model/inference_example.py`
- **功能**:
  - 从 HuggingFace 格式模型目录加载模型
  - 命令行推理
  - 支持指定 prompt

## 四、数据预处理功能

### 4.1 对话数据转 SFT 格式
- **脚本**: `convert_conversations_to_labeled.py`
- **功能**:
  - 将对话格式数据转换为带标签的 SFT 格式
  - 生成 keys 文件（通过 K-Means 聚类）
  - 生成 `keys.pt` 和 `meta.json`

### 4.2 Extract 数据转 SFT 格式
- **脚本**: `util_py/convert_extract_data_to_sft.py`
- **功能**:
  - 将 `train_data_with_extract.json` 转换为 SFT 训练格式
  - 生成训练集/验证集（8:2 划分）
  - 将 extract_support 转换为 token IDs，生成 `knowledge_cache.pt`
  - 保存 UUID 映射文件

### 4.3 Parquet 数据提取为 Memory Bank
- **脚本**: `util_py/extract_parquet_to_memory_bank.py`
- **功能**:
  - 从 Parquet 格式数据中提取事实
  - 生成 Memory Bank（.pt 格式）
  - 支持批量处理

### 4.4 JSONL 转 Memory Bank
- **脚本**: `data/pt_factorys/scripts/jsonl_to_pt.py`
- **功能**:
  - 将 JSONL 格式的事实文件转换为 .pt 格式的 Memory Bank
  - 支持 LLMLingua 压缩长文本
  - 自动截断/填充到目标长度
  - 生成 `memory_bank` 和 `valid_mask`

### 4.5 统一数据准备
- **脚本**: `data/pt_factorys/scripts/prepare_unified_data.py`
- **功能**: 准备统一格式的训练数据

### 4.6 通用事实提取器
- **脚本**: `data/pt_factorys/scripts/universal_fact_extractor.py`
- **功能**: 从多种数据源提取事实

### 4.7 添加源 ID
- **脚本**: `data/pt_factorys/scripts/add_source_ids.py`
- **功能**: 为数据添加源标识符

## 五、Keys 生成功能

### 5.1 从 Memory Bank 生成 Keys
- **脚本**: `util_py/generate_keys_from_memory_bank.py`
- **功能**:
  - 从 Memory Bank（.pt 文件）生成 Product Key Memory 的 keys
  - 使用 K-Means 聚类生成 row_keys 和 col_keys
  - 生成 `keys.pt` 文件（包含 row_keys 和 col_keys）

### 5.2 从 Extract 数据生成 Keys
- **脚本**: `util_py/generate_keys_from_extract.py`
- **功能**:
  - 从提取的事实数据生成 keys
  - 支持自定义聚类参数

## 六、模型下载功能

### 6.1 Qwen3 模型下载
- **脚本**: `download_qwen_model.py`
- **功能**:
  - 下载 Qwen3-4B-Instruct 模型
  - 使用 HuggingFace 镜像加速下载
  - 保存到 `Qwen_hg/` 目录

## 七、测试功能

### 7.1 Checkpoint 对比测试
- **脚本**: `test_checkpoint_comparison.py`
- **功能**: 对比原始 checkpoint 和转换后的 HuggingFace 模型的输出

### 7.2 基线推理测试
- **脚本**: `test_inference_baseline.py`
- **功能**: 测试基线模型（无记忆增强）的推理能力

### 7.3 更新后推理测试
- **脚本**: `test_inference_update.py`
- **功能**: 测试 Memory Bank 更新后的推理能力

### 7.4 评估测试
- **脚本**: `test_eval.py`
- **功能**: 模型性能评估

## 八、Memory Bank 管理功能

### 8.1 Memory Bank 切换
- **脚本**: `my_model/switch_memory_bank.py`（转换后生成）
- **功能**:
  - 为 HuggingFace 格式模型切换不同的 Memory Bank
  - 自动重新生成 keys（如果需要）

### 8.2 Memory Bank 更新
- **模块**: `utils/memory_bank_updater.py`
- **功能**:
  - 动态更新 Memory Bank
  - 支持多种更新策略（LRU、FIFO 等）
  - 使用 LLMLingua 压缩新知识

### 8.3 Keys 重聚类
- **模块**: `utils/keys_recluster.py`
- **功能**: 对 Memory Bank 进行重新聚类，更新 keys

## 九、数据集处理功能

### 9.1 OMCQ 数据转换
- **脚本**: `scripts/convert_omcq_to_sft.py`
- **功能**: 将 OMCQ（Object Multiple Choice Question）数据转换为 SFT 格式

### 9.2 数据集处理器
- **模块**: `utils/dataset_processor/`
  - `dataset_processor.py`: 通用数据集处理器
  - `predicate_mcq_generator.py`: 谓词 MCQ 生成器
  - `object_mcq_generator.py`: 对象 MCQ 生成器
  - `judgment_generator.py`: 判断生成器

## 十、工具脚本

### 10.1 打包脚本
- **脚本**: `package_for_pretrain.sh`, `package_for_migration.sh`
- **功能**: 打包模型/数据用于迁移或部署

### 10.2 模型转换脚本（Shell）
- **脚本**: `scripts/convert_checkpoint_to_hf.sh`
- **功能**: 封装 checkpoint 转 HuggingFace 的流程

### 10.3 嵌入模型下载
- **脚本**: `util_py/download_embedding_model.py`
- **功能**: 下载用于 embeddings 的模型（如 BGE）

## 十一、核心模型组件

### 11.1 核心模型
- **模块**: `models/core/ExplicitLM.py`
  - ExplicitLM 主模型（基于 Qwen3）
  - 显式记忆库机制
  - Shortcut 机制确保 backbone 独立工作

### 11.2 Transformer Block
- **模块**: `models/core/Qwen3ExplicitLMBlock.py`
  - 集成 MemoryGate 和 GatedMemoryFusion 的 Transformer Block

### 11.3 Memory 组件
- **模块**: `models/memory_bank/`
  - `MemoryGate.py`: Product Key Memory 检索门控
  - `GatedMemoryFusion.py`: 记忆融合模块
  - `GatedMemoryFusionLoRA.py`: LoRA 版本的融合模块

### 11.4 工具层
- **模块**: `models/layers/RMSNorm.py`: 归一化层

## 十二、配置管理

### 12.1 配置模块
- **模块**: `config/`
  - `__init__.py`: 配置管理主模块（使用 Hydra）
  - `model.py`: 模型配置
  - `dataset.py`: 数据集配置
  - `training.py`: 训练配置
  - `logging.py`: 日志配置
  - `memory_update.py`: Memory Bank 更新配置
  - `qwen3_4b_params.py`: Qwen3-4B 参数配置

## 十三、工具函数

### 13.1 模型初始化
- **模块**: `utils/model_initializer.py`
  - 模型初始化
  - 加载预训练权重（Router、Fusion）
  - 支持多种 checkpoint 格式

### 13.2 数据加载
- **模块**: 
  - `utils/pretrain_datasets.py`: 预训练数据加载器
  - `utils/sft_datasets.py`: SFT 数据加载器

### 13.3 训练循环
- **模块**: `utils/train_loop_sft.py`: SFT 训练循环

### 13.4 日志和监控
- **模块**: `utils/logger.py`: 日志记录器（支持 SwanLab）

### 13.5 其他工具
- `utils/clustering.py`: 聚类工具
- `utils/fact_extractor.py`: 事实提取工具
- `utils/train_utils.py`: 训练工具函数
- `utils/memory_update_tracker.py`: Memory 更新跟踪器
- `utils/dual_path_inference.py`: 双路径推理（实验性）

## 十四、文档

- `README.md`: 项目主文档
- `CONVERT_AND_INFERENCE_GUIDE.md`: 转换与推理指南
- `ENVIRONMENT_SETUP.md`: 环境设置指南
- `MEMORY_BANK_SEPARATION_DESIGN.md`: Memory Bank 分离设计方案
- `GENERATION_FIX_DOCUMENTATION.md`: 生成修复文档
- `QWEN3_VS_EXPLICITLM_COMPARISON.md`: Qwen3 vs ExplicitLM 对比

## 总结

ExplicitLM 项目包含以下主要功能模块：

1. **训练** (2 种): Pretrain 训练、SFT 训练
2. **转换** (2 种): Checkpoint → HuggingFace（独立/同步存储）
3. **推理** (2 种): 对话式推理、模型推理示例
4. **数据预处理** (7+ 种): 多种数据格式转换和处理
5. **Keys 生成** (2 种): 从不同数据源生成 keys
6. **模型管理** (3+ 种): Memory Bank 切换、更新、Keys 重聚类
7. **测试** (4 种): 各种测试和评估脚本
8. **工具** (10+ 种): 打包、下载、数据集处理等

整个项目采用参数高效训练策略，只训练记忆相关组件，保持 Qwen3 backbone 冻结，是一个完整的显式记忆增强语言模型训练和推理框架。


