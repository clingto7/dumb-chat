# Commit Plan (Atomic Commits)

按依赖顺序，从底层到上层，每个 commit 是一个不可分割的逻辑单元。

---

## Commit 1: refactor: extract train_with_configs() from train()

**文件变更：**
- `guppylm/train.py` — 将训练循环核心提取为 `train_with_configs(mc, tc)`，原 `train()` 变为薄包装

**理由：** 后续 wechat/train.py 需要复用训练循环，必须先有此可复用入口。这是纯粹的内部重构，不改变任何外部行为。

```
guppylm/train.py
  - train(): 原函数体 → 移入 train_with_configs()
  + train_with_configs(mc: GuppyConfig, tc: TrainConfig) -> None: 新函数
  + train(): 简化为 train_with_configs(GuppyConfig(), TrainConfig())
```

---

## Commit 2: feat: add wechat sub-package with config and decrypt modules

**文件变更：**
- `guppylm/wechat/__init__.py` — 新建，导出配置类
- `guppylm/wechat/config.py` — 新建，5 个 dataclass 配置
- `guppylm/wechat/decrypt.py` — 新建，密钥提取 + 数据库解密

**理由：** 这是 wechat pipeline 的基础层——配置定义和数据库解密。没有解密就没有后续所有步骤。

```
guppylm/wechat/__init__.py        (新建, 9行)
guppylm/wechat/config.py          (新建, 106行)
guppylm/wechat/decrypt.py         (新建, 622行)
```

---

## Commit 3: feat: add wechat message extraction with dynamic schema discovery

**文件变更：**
- `guppylm/wechat/extract.py` — 新建，联系人/会话/消息提取

**理由：** 依赖 commit 2 的 config 定义。解密后的数据库需要提取消息才能使用。动态表发现是此模块的核心特性。

```
guppylm/wechat/extract.py         (新建, 593行)
```

---

## Commit 4: feat: add wechat ChatML conversion module

**文件变更：**
- `guppylm/wechat/convert.py` — 新建，原始消息 → ChatML 训练样本

**理由：** 依赖 commit 3 的提取结果（WechatMessage 数据结构）。将原始消息转换为模型可训练的 ChatML 格式。

```
guppylm/wechat/convert.py         (新建, 307行)
```

---

## Commit 5: feat: add wechat data augmentation (template + LLM)

**文件变更：**
- `guppylm/wechat/augment.py` — 新建，模板式增强 + 可选 LLM 增强

**理由：** 依赖 commit 4 的 ChatMLSample 结构。增强模块扩充训练数据量。

```
guppylm/wechat/augment.py         (新建, 357行)
```

---

## Commit 6: feat: add wechat tokenizer training and pipeline orchestration

**文件变更：**
- `guppylm/wechat/prepare.py` — 新建，BPE tokenizer 训练 + 数据准备编排

**理由：** 依赖 commit 4/5 的数据格式。训练 tokenizer 是模型训练前的最后一步数据准备。

```
guppylm/wechat/prepare.py         (新建, 123行)
```

---

## Commit 7: feat: add wechat training and inference modules

**文件变更：**
- `guppylm/wechat/train.py` — 新建，WechatTrainConfig → GuppyConfig+TrainConfig 映射
- `guppylm/wechat/inference.py` — 新建，WechatPersonaInference 交互式对话

**理由：** 依赖 commit 1 的 train_with_configs() 和 commit 6 的 tokenizer。训练和推理是 pipeline 的终点。

```
guppylm/wechat/train.py           (新建, 59行)
guppylm/wechat/inference.py       (新建, 93行)
```

---

## Commit 8: feat: add wechat CLI commands and project configuration

**文件变更：**
- `guppylm/__main__.py` — 添加 5 个 wechat-* 子命令
- `.gitignore` — 添加 wechat_data/、wechat_checkpoints/、all_keys.json
- `requirements.txt` — 添加 pycryptodome

**理由：** CLI 集成和项目配置是最后一层粘合。所有模块就绪后才能添加入口。.gitignore 和 requirements 是项目基础设施。

```
guppylm/__main__.py               (修改, +38行)
.gitignore                        (修改, +5行)
requirements.txt                  (修改, +1行)
```

---

## Commit 9: docs: add wechat persona module documentation

**文件变更：**
- `docs/wechat-persona.md` — 新建，完整的模块说明和使用方式

**理由：** 文档独立于代码变更，单独提交。

```
docs/wechat-persona.md            (新建)
```

---

## 依赖关系图

```
commit 1 (train refactor)
    │
    ├─→ commit 2 (config + decrypt)
    │       │
    │       └─→ commit 3 (extract)
    │               │
    │               └─→ commit 4 (convert)
    │                       │
    │                       └─→ commit 5 (augment)
    │                               │
    │                               └─→ commit 6 (prepare/tokenizer)
    │                                       │
    │                                       └─→ commit 7 (train + inference)
    │                                               │
    ├───────────────────────────────────────────────┘
    │
    └─→ commit 8 (CLI + config files)
            │
            └─→ commit 9 (docs)
```
