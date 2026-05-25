# WeChat Persona Module

GuppyLM 的 WeChat persona 模块，用于从微信 Linux 桌面版聊天记录中提取数据，训练一个模仿特定人物说话风格的小型 LLM。

## 架构概览

```
微信加密DB
  │
  ▼  decrypt.py (密钥提取 + AES-256-CBC解密)
解密后的SQLite数据库
  │
  ▼  extract.py (动态表发现 + 消息提取)
原始消息JSONL
  │
  ▼  convert.py (ChatML格式转换，单轮/多轮)
ChatML训练样本
  │
  ▼  augment.py (模板式增强 + 可选LLM增强)
增强后的训练数据
  │
  ▼  prepare.py (BPE tokenizer训练，vocab=8192)
tokenizer + 训练数据
  │
  ▼  train.py (GuppyLM模型训练，复用原架构)
模型checkpoint
  │
  ▼  inference.py (交互式对话)
人物风格回复
```

## 模块说明

### `guppylm/wechat/config.py`

5 个 dataclass 配置类：

| 配置类 | 用途 | 关键字段 |
|--------|------|----------|
| `WechatDecryptConfig` | 数据库解密 | `wechat_data_dir`, `wxid`, `decrypted_dir`, `keys_file` |
| `WechatExtractConfig` | 消息提取 | `decrypted_dir`, `target_contacts`, `include_types`, `max_messages` |
| `WechatChatMLConfig` | ChatML转换 | `target_wxid`, `max_turns`, `max_context_messages`, `eval_ratio` |
| `WechatAugmentConfig` | 数据增强 | `llm_enabled`, `template_enabled`, `synonym_replace_prob` |
| `WechatTrainConfig` | 训练配置 | `vocab_size=8192`, `max_seq_len=256`, `batch_size=16`, `max_steps=20000` |

默认数据路径：
- 微信原始数据：`~/Documents/xwechat_files/`（自动检测 wxid 子目录）
- 解密输出：`wechat_data/decrypted/`
- 提取输出：`wechat_data/extracted/`
- ChatML 输出：`wechat_data/chatml/`
- 增强输出：`wechat_data/augmented/`
- Tokenizer：`wechat_data/tokenizer.json`
- 模型检查点：`wechat_checkpoints/`

### `guppylm/wechat/decrypt.py`

**密钥提取**（从微信进程内存）：
1. 扫描 `/proc/` 发现 WeChat 进程（按 RSS 排序，最大进程优先）
2. 解析 `/proc/<pid>/maps` 获取可读内存区域（跳过系统库，保留 wcdb 相关映射）
3. 用正则 `x'([0-9a-fA-F]{64,192})'` 扫描 `/proc/<pid>/mem`
4. 从 `.db` 文件头提取 salt，用 HMAC-SHA512 验证候选密钥
5. 支持交叉验证（同一密钥可能用于多个不同 salt 的数据库）

**数据库解密**（纯 Python，依赖 PyCryptodome）：
- SQLCipher 4 格式：AES-256-CBC 页级解密
- Page 1 特殊处理：前 16 字节是 salt，解密后恢复 SQLite 标准头
- 每页 4096 字节，末尾 80 字节保留区（16 IV + 64 HMAC）
- 解密后用 `sqlite3` 模块验证可正常打开

关键函数：
- `extract_keys(wechat_data_dir, output_path)` — 从进程内存提取密钥
- `decrypt_database(db_path, key_hex, output_path)` — 解密单个数据库
- `decrypt_all(config)` — 完整流程：提取密钥 + 解密所有数据库

权限要求：需要 root 或 `CAP_SYS_PTRACE` 才能读取进程内存。

### `guppylm/wechat/extract.py`

**动态表发现**：不硬编码表名/列名，通过以下方式适配不同微信版本：
1. 查询 `sqlite_master` 获取所有表名
2. 用 `_discover_columns()` 映射标准字段名到实际列名（支持多种命名变体）
3. 消息表按 `Msg_{md5(username)}` 命名规则定位

**联系人提取**：
- 从 `contact.db` 读取好友/群聊列表
- 自动识别群聊（`@chatroom` 后缀）
- 过滤服务号（`gh_` 前缀）

**消息提取**：
- 从 `message_0.db` 的 per-user 表提取消息
- 默认只提取文本消息（type=1）
- 群聊消息自动解析发送者（`wxid_xxx:\ncontent` 格式）
- 使用参数化查询防止 SQL 注入
- 自动检测 self_wxid（从密钥文件的 `_db_dir` 元数据推断）

关键函数：
- `list_contacts(decrypted_dir)` — 列出所有联系人
- `list_sessions(decrypted_dir)` — 列出最近会话
- `extract_messages(decrypted_dir, contact_wxid, ...)` — 提取指定联系人的消息
- `extract_and_save(config)` — 交互式选择 + 保存到 JSONL

### `guppylm/wechat/convert.py`

**单轮转换**：
- 对目标人物的每条消息，收集前面最多 N 条非目标消息作为 "user" 上下文
- 目标人物的消息作为 "assistant" 回复

**多轮转换**：
- 按时间排序，合并连续同一人的消息
- 在时间间隔 >30 分钟处自动分割对话段
- 滑动窗口生成多轮样本
- 确保样本以 "user" 开头、以 "assistant" 结尾

输出格式与原 fish-chat 一致的 JSONL：`{"text": "<ChatML>", "category": "wechat_persona"}`

**sender 判断逻辑**：
- 群聊：`sender` 字段有实际 wxid，与 `target_wxid` 比较
- 一对一：`sender` 可能为空，用 `is_self` 判断（`is_self=False` + `talker=target` = 目标发的）

### `guppylm/wechat/augment.py`

**模板式增强**（本地，不需要 API）：
- 中文同义词替换（50+ 条映射，如 "觉得"→"感觉"、"什么"→"啥"）
- 随机删除虚词（概率可配置，保留标点）
- 短句重排序（拆分中文标点，打乱中间分句顺序）
- 只增强 "assistant" 部分，保持 "user" 输入不变

**LLM 增强**（可选，需要 OpenAI API）：
- 用 LLM 基于原始消息生成风格一致的变体
- 通过 `WechatAugmentConfig.llm_enabled` 控制开关
- 通过 `LLM_API_KEY` 环境变量或 config 设置 API key
- 默认模型 `gpt-4o-mini`，温度 0.8

### `guppylm/wechat/prepare.py`

- 读取增强后的 JSONL 数据
- 训练 ByteLevel BPE tokenizer（vocab_size=8192，适应中文）
- ByteLevel 预分词器通过 UTF-8 字节分解天然处理中文
- 3 个特殊 token：`<pad>`(0), `<|im_start|>`(1), `<|im_end|>`(2)
- 训练后运行中文样本编解码测试

### `guppylm/wechat/train.py`

- 复用 `GuppyLM` 模型架构（vanilla transformer）
- `WechatTrainConfig` 映射为 `GuppyConfig` + `TrainConfig`
- 关键差异：`vocab_size=8192`（中文）、`max_seq_len=256`（更长上下文）、`batch_size=16`
- 调用 `train_with_configs()` 复用核心训练循环

### `guppylm/wechat/inference.py`

- `WechatPersonaInference` 包装 `GuppyInference`
- 默认路径：`wechat_checkpoints/best_model.pt` + `wechat_data/tokenizer.json`
- `max_tokens=128`（中文回复需要更多 token）
- 支持 `--prompt` 单次模式和交互式 REPL

## 使用方式

### 前置条件

- 微信 Linux 桌面版正在运行
- `pycryptodome` 已安装（`pip install pycryptodome`）
- root 或 `CAP_SYS_PTRACE` 权限（仅解密步骤需要）

### 完整流程

```bash
# 1. 解密微信数据库（需要 sudo，微信必须正在运行）
sudo python -m guppylm wechat-decrypt
# 输出：wechat_data/all_keys.json + wechat_data/decrypted/

# 2. 提取聊天记录（交互式选择联系人/群聊）
python -m guppylm wechat-extract
# 输出：wechat_data/extracted/<wxid>.jsonl

# 3. 转换为 ChatML + 数据增强 + 训练 tokenizer
#    需要指定目标人物的 wxid
python -m guppylm wechat-prepare wxid_abc123
# 输出：wechat_data/chatml/ + wechat_data/augmented/ + wechat_data/tokenizer.json

# 4. 训练人物模型
python -m guppylm wechat-train
# 输出：wechat_checkpoints/best_model.pt

# 5. 与人物模型对话
python -m guppylm wechat-chat
# 或单次提问：
python -m guppylm wechat-chat -p "最近怎么样？"
```

### 自定义配置

各命令使用默认配置，如需调整可直接修改 `guppylm/wechat/config.py` 中的 dataclass 默认值，或在 Python 中传入自定义配置对象：

```python
from guppylm.wechat.config import WechatChatMLConfig
from guppylm.wechat.convert import convert_to_chatml

config = WechatChatMLConfig(
    target_wxid="wxid_abc123",
    max_turns=3,            # 多轮对话
    max_msg_length=300,     # 允许更长消息
    eval_ratio=0.1,         # 10% 评估集
)
convert_to_chatml(config)
```

### 启用 LLM 增强

```bash
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.openai.com/v1"  # 可选，默认用 OpenAI

# 在 Python 中启用
from guppylm.wechat.config import WechatAugmentConfig
from guppylm.wechat.augment import augment

config = WechatAugmentConfig(
    llm_enabled=True,
    llm_model="gpt-4o-mini",
    llm_augment_ratio=2,    # 每条原始样本生成 2 条变体
)
augment(config)
```

## 模型参数

默认 WeChat persona 配置（`WechatTrainConfig`）：

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 8192 | 中文需要更大词表 |
| max_seq_len | 256 | 中文对话需要更长上下文 |
| d_model | 384 | 与 fish-chat 相同 |
| n_layers | 6 | 与 fish-chat 相同 |
| n_heads | 6 | 与 fish-chat 相同 |
| batch_size | 16 | 小批量适配中文序列 |
| max_steps | 20000 | 更多训练步数 |
| 参数量 | ~15M | vocab 增大导致 embedding 层更大 |

## 数据安全

- 密钥文件 `all_keys.json` 权限为 `0600`（仅所有者可读写）
- `wechat_data/`、`wechat_checkpoints/`、`all_keys.json` 已加入 `.gitignore`
- 解密后的数据库包含完整聊天记录，不应提交到版本控制
- LLM 增强会将消息内容发送到第三方 API，注意隐私风险

## 与原有 fish-chat 模块的关系

- `guppylm/wechat/` 是完全独立的子包，不修改也不依赖 fish-chat 的数据生成代码
- 共享模块：`GuppyLM` 模型架构、`GuppyInference` 推理引擎、`GuppyDataset` 数据加载
- `train.py` 已重构为 `train_with_configs(mc, tc)` 可复用函数，原 `train()` 行为不变
- 两条 pipeline 的数据和检查点完全隔离（`data/` vs `wechat_data/`，`checkpoints/` vs `wechat_checkpoints/`）
