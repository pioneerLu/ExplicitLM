# Memory Bank独立存储设计方案

## 需求分析

**目标：**
1. Memory Bank独立于模型文件存储
2. 可以在不修改模型文件的情况下更新Memory Bank
3. 支持动态注入不同的Memory Bank
4. 保持向后兼容性（可选）

## 当前实现分析

### 当前存储方式
- Memory Bank通过`register_buffer("memory_bank", ..., persistent=True)`注册
- 保存在`pytorch_model.bin`中的`model.memory_bank`键
- 加载模型时自动加载Memory Bank
- 运行时可以通过`model.memory_bank.data.copy_(new_data)`更新

### 当前更新方式
- 通过`MemoryBankUpdater`类动态更新
- 更新后需要同步到所有进程（多卡训练）
- 更新会直接修改模型中的buffer

## 设计方案

### 方案1：完全分离存储（推荐）

**核心思路：**
- Memory Bank不保存在模型文件中（`persistent=False`）
- 单独保存为独立文件（如`memory_bank.pt`）
- 加载时先加载模型，再单独加载Memory Bank
- 更新时只更新独立文件，不修改模型

**优点：**
- ✅ 完全独立，模型文件不包含Memory Bank
- ✅ 可以轻松切换不同的Memory Bank
- ✅ 模型文件更小，加载更快
- ✅ 支持一个模型使用多个Memory Bank

**缺点：**
- ⚠️ 需要额外的加载步骤
- ⚠️ 需要确保Memory Bank文件与模型匹配

**实现要点：**
1. 修改`ExplicitLM.__init__`：`persistent=False`
2. 转换脚本：保存时排除`memory_bank`，单独保存
3. 加载脚本：提供`load_memory_bank()`函数
4. 更新逻辑：更新独立文件，然后重新加载

**文件结构：**
```
hf_model_path/
├── pytorch_model.bin          # 不包含memory_bank
├── config.json
├── modeling_explicitlm.py
└── memory_bank.pt             # 独立文件（可选）
    └── {
        'memory_bank': tensor,
        'valid_mask': tensor,
        'metadata': {...}
    }
```

### 方案2：配置驱动分离

**核心思路：**
- 在config中添加`memory_bank_path`字段
- 如果路径存在，从独立文件加载
- 如果路径为空，使用模型中的Memory Bank（向后兼容）
- 支持运行时切换Memory Bank路径

**优点：**
- ✅ 向后兼容（可以继续使用模型中的Memory Bank）
- ✅ 灵活：可以选择使用独立文件或模型内Memory Bank
- ✅ 配置清晰：通过config控制

**缺点：**
- ⚠️ 实现稍复杂（需要处理两种加载方式）
- ⚠️ 需要维护两套逻辑

**实现要点：**
1. 在`ExplicitLMConfig`中添加`memory_bank_path: Optional[str] = None`
2. 在`ExplicitLM.__init__`中检查config，决定是否从文件加载
3. 转换脚本：可选择保存方式（独立文件或模型内）
4. 提供`load_memory_bank_from_file()`方法

**Config示例：**
```python
{
    "memory_bank_path": "memory_bank.pt",  # 如果为None，使用模型内的
    "knowledge_num": 1048576,
    "knowledge_length": 32,
    ...
}
```

### 方案3：可选分离（转换时决定）

**核心思路：**
- 转换时通过参数`--separate_memory_bank`决定
- 如果启用：Memory Bank保存为独立文件，模型不包含
- 如果禁用：保持当前行为（保存在模型中）

**优点：**
- ✅ 用户可以选择存储方式
- ✅ 向后兼容（默认行为不变）
- ✅ 实现简单

**缺点：**
- ⚠️ 转换后无法改变存储方式
- ⚠️ 需要重新转换才能切换

**实现要点：**
1. 转换脚本添加`--separate_memory_bank`参数
2. 如果启用：`persistent=False`，单独保存文件
3. 如果禁用：保持`persistent=True`

## 推荐方案：方案1 + 方案2混合

**核心设计：**
1. **默认行为**：Memory Bank独立存储（`persistent=False`）
2. **加载机制**：
   - 优先从`config.memory_bank_path`加载
   - 如果路径不存在，尝试从模型目录的`memory_bank.pt`加载
   - 如果都不存在，使用默认初始化（全pad）
3. **更新机制**：
   - 更新时写入独立文件
   - 可以选择更新模型目录下的文件或指定路径的文件

**文件结构：**
```
hf_model_path/
├── pytorch_model.bin          # 不包含memory_bank
├── config.json                # 包含memory_bank_path（可选）
├── modeling_explicitlm.py
└── memory_bank.pt             # 独立文件（默认位置）
```

**Config结构：**
```python
{
    "memory_bank_path": "memory_bank.pt",  # 相对路径或绝对路径，None表示使用默认
    "knowledge_num": 1048576,
    "knowledge_length": 32,
    ...
}
```

## 实现细节

### 1. 模型初始化修改

```python
# ExplicitLM.__init__
if not use_moe:
    # Memory Bank独立存储
    self.register_buffer(
        "memory_bank",
        memory_bank_tensor,
        persistent=False,  # ✅ 改为False，不保存在模型文件中
    )
    
    # 如果config中有memory_bank_path，加载它
    if hasattr(qwen3_config, 'memory_bank_path') and qwen3_config.memory_bank_path:
        self.load_memory_bank(qwen3_config.memory_bank_path)
    elif hasattr(self, 'config') and hasattr(self.config, 'memory_bank_path'):
        # 从ExplicitLMConfig加载
        if self.config.memory_bank_path:
            self.load_memory_bank(self.config.memory_bank_path)
```

### 2. 添加加载方法

```python
def load_memory_bank(self, path: str):
    """从文件加载Memory Bank"""
    if not os.path.exists(path):
        # 尝试相对路径（相对于模型目录）
        if hasattr(self, 'config') and hasattr(self.config, '_name_or_path'):
            model_dir = os.path.dirname(self.config._name_or_path)
            abs_path = os.path.join(model_dir, path)
            if os.path.exists(abs_path):
                path = abs_path
            else:
                print(f"⚠️ Memory Bank文件不存在: {path}")
                return
    
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict):
        memory_bank = data.get('memory_bank', data.get('processed_tensor'))
        valid_mask = data.get('valid_mask', None)
    else:
        memory_bank = data
        valid_mask = None
    
    if memory_bank is not None:
        self.memory_bank.data.copy_(memory_bank)
        if valid_mask is not None and hasattr(self, 'valid_mask'):
            self.valid_mask.data.copy_(valid_mask)
```

### 3. 转换脚本修改

```python
# 保存时排除memory_bank
def save_pretrained_without_memory_bank(self, output_path):
    """保存模型，但不包含memory_bank"""
    state_dict = self.state_dict()
    # 移除memory_bank相关的keys
    state_dict = {k: v for k, v in state_dict.items() 
                  if 'memory_bank' not in k and 'valid_mask' not in k}
    # 保存...
    
    # 单独保存memory_bank
    if hasattr(self.model, 'memory_bank'):
        memory_bank_data = {
            'memory_bank': self.model.memory_bank.cpu(),
            'valid_mask': self.model.valid_mask.cpu() if hasattr(self.model, 'valid_mask') else None,
            'metadata': {
                'knowledge_num': self.model.memory_bank.shape[0],
                'knowledge_length': self.model.memory_bank.shape[1],
            }
        }
        torch.save(memory_bank_data, os.path.join(output_path, 'memory_bank.pt'))
```

### 4. 更新逻辑修改

```python
# MemoryBankUpdater
def update_and_save(self, memory_bank_path: Optional[str] = None):
    """更新Memory Bank并保存到文件"""
    # 执行更新...
    
    # 保存到文件
    if memory_bank_path is None:
        # 尝试从config获取路径
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'memory_bank_path'):
            memory_bank_path = self.model.config.memory_bank_path
        else:
            # 默认保存到模型目录
            memory_bank_path = 'memory_bank.pt'
    
    memory_bank_data = {
        'memory_bank': self.memory_bank.cpu(),
        'valid_mask': self.valid_mask.cpu() if hasattr(self, 'valid_mask') else None,
    }
    torch.save(memory_bank_data, memory_bank_path)
```

## 使用示例

### 转换时
```bash
uv run python convert_checkpoint_to_hf.py \
    --checkpoint_path checkpoints/... \
    --qwen3_path Qwen_hg/Qwen3-4b \
    --output_path hf_model \
    --memory_bank_path data/memory_bank.pt  # 可选，会单独保存
```

### 加载时
```python
from transformers import AutoModelForCausalLM

# 方式1：自动加载（从模型目录的memory_bank.pt）
model = AutoModelForCausalLM.from_pretrained("hf_model", trust_remote_code=True)
# 会自动从 hf_model/memory_bank.pt 加载

# 方式2：指定路径（通过config）
# 在config.json中设置 "memory_bank_path": "custom_memory_bank.pt"
model = AutoModelForCausalLM.from_pretrained("hf_model", trust_remote_code=True)

# 方式3：手动加载
model = AutoModelForCausalLM.from_pretrained("hf_model", trust_remote_code=True)
model.model.load_memory_bank("custom_memory_bank.pt")
```

### 更新时
```python
# 更新Memory Bank
memory_bank_updater.update_from_text("新的事实...")

# 保存到文件（不修改模型）
memory_bank_updater.update_and_save("memory_bank.pt")

# 或者保存到自定义路径
memory_bank_updater.update_and_save("custom_path/memory_bank.pt")
```

## 兼容性考虑

### 向后兼容
- 如果模型文件中已有`model.memory_bank`（旧格式），加载时检测并迁移
- 提供迁移脚本，将旧模型的Memory Bank提取为独立文件

### 迁移脚本
```python
def migrate_memory_bank_to_separate_file(model_path, output_path=None):
    """将模型中的Memory Bank提取为独立文件"""
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(model.model, 'memory_bank'):
        memory_bank_data = {
            'memory_bank': model.model.memory_bank.cpu(),
            'valid_mask': model.model.valid_mask.cpu() if hasattr(model.model, 'valid_mask') else None,
        }
        output_path = output_path or os.path.join(model_path, 'memory_bank.pt')
        torch.save(memory_bank_data, output_path)
        print(f"✅ Memory Bank已提取到: {output_path}")
```

## 总结

**推荐方案：方案1 + 方案2混合**
- Memory Bank默认独立存储
- 通过config控制加载路径
- 支持运行时切换
- 保持灵活性和兼容性

**关键修改点：**
1. `ExplicitLM.__init__`: `persistent=False`
2. 添加`load_memory_bank()`方法
3. 转换脚本：单独保存Memory Bank
4. `MemoryBankUpdater`: 支持保存到文件
5. Config: 添加`memory_bank_path`字段

