# GuppyLM 完全初学者详细指南

这份文档面向**完全没有 LLM 背景**的读者。它不会只说“这是一个小模型”，而是会把这个仓库里的每一块代码怎么连起来、每个关键概念在代码里对应哪里、怎么把项目跑起来，都讲清楚。

GuppyLM 的目标很明确：它不是要做 ChatGPT 那种通用大模型，而是用一个很小的 PyTorch Transformer，训练出一个“像小鱼 Guppy 一样说话”的语言模型。你可以把它理解成一个教学项目：从造数据、训练 tokenizer、搭建模型、训练参数、保存 checkpoint、加载模型聊天，到导出到浏览器运行，整条链路都在这个仓库里。

---

## 1. 这个项目到底是什么？

GuppyLM 是一个约 8.7M 参数的小型语言模型。它被训练成一个鱼缸里的小鱼角色：说话短、基本小写、总是围绕水、食物、光、气泡、鱼缸、猫、温度这些主题。

从机器学习角度看，它做的是一件非常标准的事：

> 给模型一段文本的前半部分，让它预测下一个 token 是什么。

比如训练样本可能长这样：

```text
<|im_start|>user
hi guppy<|im_end|>
<|im_start|>assistant
hello. the water is nice today.<|im_end|>
```

模型训练时并不知道“聊天”是什么复杂概念。它只是在学习：当看到 `<|im_start|>user\nhi guppy...` 之后，后面很可能出现 `<|im_start|>assistant\nhello...`。大量类似样本训练之后，它就学会了这种问答格式和 Guppy 的说话风格。

这个项目的核心流程是：

```text
生成鱼类聊天数据
        ↓
把文本切成 token，训练 BPE tokenizer
        ↓
把 token 序列做成训练数据 x/y
        ↓
用 Transformer 学习预测下一个 token
        ↓
保存训练好的参数 checkpoint
        ↓
加载 checkpoint，根据用户输入逐 token 生成回复
        ↓
可选：导出 HuggingFace 格式或 ONNX 浏览器版本
```

---

## 2. 如何把项目跑起来

### 2.1 安装依赖

仓库没有 `pyproject.toml` 或 `setup.py`，也没有安装成一个正式 Python 包。你通常在仓库根目录直接运行它。

推荐先安装依赖：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 里只有核心运行/训练依赖：

```text
torch>=2.0.0
tokenizers>=0.19.0
tqdm>=4.65.0
numpy>=1.24.0
datasets>=2.14.0
```

这里最重要的是：

- `torch`：PyTorch，负责神经网络、张量计算、反向传播、训练。
- `tokenizers`：HuggingFace 的高性能 tokenizer 库，用来训练和加载 BPE tokenizer。
- `datasets`：主要用于 Colab 或导出场景里从 HuggingFace 下载数据集。
- `numpy`、`tqdm`：辅助数值处理和进度显示。

如果你要运行导出脚本，可能还需要额外安装：

```bash
python -m pip install huggingface_hub onnx onnxruntime onnxscript
```

因为这些导出依赖没有放进基础 `requirements.txt`。

### 2.2 直接下载已经训练好的模型并聊天

如果你只是想先体验，不想自己训练，运行：

```bash
python -m guppylm download
python -m guppylm chat --prompt "tell me a joke"
```

第一条命令会下载：

```text
checkpoints/best_model.pt
data/tokenizer.json
checkpoints/config.json
```

第二条命令会加载模型和 tokenizer，然后对单个 prompt 生成回复。

也可以进入交互式聊天：

```bash
python -m guppylm chat
```

但 README 里特别说明：这个模型上下文长度只有 128 tokens，多轮聊天很快会退化。所以更推荐单轮：

```bash
python -m guppylm chat --prompt "hi guppy"
```

### 2.3 从零生成数据并训练模型

如果你想完整体验训练流程：

```bash
python -m guppylm prepare
python -m guppylm train
```

`prepare` 会做两件事：

1. 生成 60,000 条鱼类聊天样本。
2. 用这些文本训练 BPE tokenizer，并保存到 `data/tokenizer.json`。

训练数据会写入：

```text
data/train.jsonl
data/eval.jsonl
data/train_openai.jsonl
data/eval_openai.jsonl
data/tokenizer.json
```

`train` 会读取：

```text
data/train.jsonl
data/eval.jsonl
data/tokenizer.json
```

然后训练模型，输出 checkpoint：

```text
checkpoints/config.json
checkpoints/best_model.pt
checkpoints/step_<step>.pt
checkpoints/final_model.pt
```

训练默认配置在 `guppylm/config.py`：

```python
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_interval: int = 200
    save_interval: int = 500
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "checkpoints"
```

如果你在普通 CPU 上训练，会比较慢。README 里的定位是用 Colab T4 GPU 快速演示。

### 2.4 浏览器运行版本

项目还提供一个静态浏览器 demo：

```text
docs/index.html
docs/model.onnx
docs/tokenizer.json
```

可以本地启动静态服务：

```bash
python3 -m http.server 8080 --directory docs
```

然后浏览器打开：

```text
http://localhost:8080
```

浏览器版本用的是 ONNX Runtime Web。它会从当前目录加载：

```javascript
const MODEL_BASE = ".";
fetch(`${MODEL_BASE}/tokenizer.json`)
fetch(`${MODEL_BASE}/model.onnx`)
```

也就是说，`model.onnx` 和 `tokenizer.json` 必须和 `index.html` 放在同一个 `docs/` 目录里。

---

## 3. 仓库结构总览

项目主要目录如下：

```text
guppylm/
├── guppylm/              # 核心 Python 包
├── tools/                # 导出脚本、Colab notebook 生成脚本
├── docs/                 # 浏览器 demo 和 ONNX/tokenizer 文件
├── assets/               # 图片资源
├── train_guppylm.ipynb   # 训练用 Colab notebook
├── use_guppylm.ipynb     # 使用模型聊天的 Colab notebook
├── requirements.txt      # 基础依赖
└── Makefile              # 目前只有 notebook 生成命令
```

核心代码在 `guppylm/` 目录：

```text
guppylm/config.py         # 模型和训练超参数
guppylm/generate_data.py  # 合成训练数据
guppylm/prepare_data.py   # 生成数据 + 训练 tokenizer
guppylm/dataset.py        # 把 JSONL 文本变成 PyTorch batch
guppylm/model.py          # Transformer 模型本体
guppylm/train.py          # 训练循环
guppylm/inference.py      # 加载模型并聊天
guppylm/__main__.py       # python -m guppylm 的命令入口
guppylm/eval_cases.py     # 手写评估样例
```

---

## 4. LLM 初学者必须先理解的几个概念

### 4.1 token：模型真正看到的不是文字，而是数字

人类看到的是：

```text
hi guppy
```

模型看到的是一串整数，比如概念上类似：

```text
[88, 64, 779, 278]
```

这些整数叫 token id。token 可以是一个字符、一个单词、一个子词、一个空格加单词，具体取决于 tokenizer。

在本项目里，tokenizer 使用 BPE，也就是 Byte Pair Encoding。代码在 `prepare_data.py`：

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
```

它会学习常见文本片段。例如，如果训练语料里经常出现 `water`、`guppy`、`food`，这些片段可能会成为比较短的 token 组合。

### 4.2 special tokens：聊天格式的边界标记

`prepare_data.py` 定义了三个特殊 token：

```python
SPECIAL_TOKENS = [
    "<pad>",         # 0
    "<|im_start|>",  # 1
    "<|im_end|>",    # 2
]
```

它们非常重要：

- `<pad>`：填充用。一个 batch 里不同句子长度不同，需要补齐到同样长度。
- `<|im_start|>`：一段消息开始。
- `<|im_end|>`：一段消息结束。

训练样本被格式化成：

```python
f"<|im_start|>user\n{s['input']}<|im_end|>\n"
f"<|im_start|>assistant\n{s['output']}<|im_end|>"
```

这告诉模型：哪里是用户说的话，哪里是助手说的话，哪里该停止。

### 4.3 embedding：把 token id 变成向量

token id 只是数字编号，本身没有意义。模型第一步要把 id 变成向量。

在 `model.py` 里：

```python
self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
```

这里：

- `vocab_size = 4096`：词表里有 4096 个 token。
- `d_model = 384`：每个 token 会变成 384 维向量。
- `tok_emb`：表示“这个 token 是什么”。
- `pos_emb`：表示“这个 token 在第几个位置”。

为什么需要位置 embedding？因为 Transformer 本身不像 RNN 那样天然知道顺序。`hi guppy` 和 `guppy hi` 的 token 一样但顺序不同，所以必须加入位置信息。

代码里把两者相加：

```python
pos = torch.arange(T, device=idx.device)
x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
```

`idx` 的形状是 `[B, T]`：

- `B` 是 batch size。
- `T` 是序列长度。

经过 embedding 后，`x` 变成 `[B, T, C]`：

- `C = d_model = 384`。

### 4.4 logits：模型输出的是每个 token 的分数

语言模型不是直接输出文字。它对词表里每个 token 给一个分数。

在 `model.py`：

```python
logits = self.lm_head(self.norm(x))
```

`logits` 的形状是：

```text
[batch_size, sequence_length, vocab_size]
```

本项目里 `vocab_size=4096`，所以每个位置都会输出 4096 个分数。分数最高的 token 就是模型认为最可能的下一个 token。但推理时不一定永远选最高分，因为那样回复会很死板，所以项目使用采样。

### 4.5 next-token prediction：训练目标就是预测下一个 token

`dataset.py` 里有一个很关键的逻辑：

```python
ids = self.samples[idx]
x = ids[:-1]
y = ids[1:]
```

假设原始 token 是：

```text
[10, 20, 30, 40]
```

那么：

```text
x = [10, 20, 30]
y = [20, 30, 40]
```

这表示：

- 看到 `10`，预测 `20`。
- 看到 `10,20`，预测 `30`。
- 看到 `10,20,30`，预测 `40`。

这就是 GPT 类模型的基本训练方式。

### 4.6 loss：模型错得有多离谱

训练时模型输出 `logits`，真实答案是 `targets`。代码用 cross entropy 计算损失：

```python
loss = F.cross_entropy(
    logits.view(-1, self.config.vocab_size),
    targets.view(-1),
    ignore_index=0,
)
```

`ignore_index=0` 的意思是忽略 `<pad>` token。因为 padding 只是为了补齐 batch 长度，不是真正要学习的文字。

loss 越低，说明模型越能预测训练文本里的下一个 token。

---

## 5. `config.py`：所有关键超参数放在哪里

`guppylm/config.py` 很短，但非常关键。

```python
@dataclass
class GuppyConfig:
    vocab_size: int = 4096
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1
```

这些字段决定模型大小：

- `vocab_size=4096`：模型能识别 4096 种 token。
- `max_seq_len=128`：最多看 128 个 token 的上下文。
- `d_model=384`：每个 token 的内部向量维度。
- `n_layers=6`：堆叠 6 个 Transformer block。
- `n_heads=6`：每层 attention 分成 6 个头。
- `ffn_hidden=768`：前馈网络中间层维度。
- `dropout=0.1`：训练时随机丢弃一部分激活，减少过拟合。

特殊 token id 也在这里：

```python
pad_id: int = 0
bos_id: int = 1           # <|im_start|>
eos_id: int = 2           # <|im_end|>
```

训练配置：

```python
@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_interval: int = 200
    save_interval: int = 500
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "checkpoints"
```

这些字段决定训练怎么跑：

- `batch_size=32`：每次训练同时看 32 条样本。
- `learning_rate=3e-4`：参数每次更新的步子大小。
- `warmup_steps=200`：前 200 步学习率从 0 慢慢升上去。
- `max_steps=10000`：训练 10000 步。
- `eval_interval=200`：每 200 步评估一次。
- `save_interval=500`：每 500 步保存一次中间 checkpoint。
- `grad_clip=1.0`：梯度裁剪，防止训练不稳定。
- `device="auto"`：自动选择 CUDA、MPS 或 CPU。

---

## 6. `generate_data.py`：没有真实语料，如何造出 60K 条鱼类聊天数据

这个文件是项目里最大的文件，约 1700 行。它的任务是生成合成数据。

为什么要合成数据？因为项目不是训练通用智能，而是训练一个稳定角色：小鱼 Guppy。与其收集大量真实聊天，不如用模板明确规定它该怎么说话。

文件开头说明了角色设定：

```python
Guppy speaks in short, lowercase sentences. It experiences the world through
water, temperature, light, vibrations, and food. It doesn't understand
human abstractions. It's friendly, curious, and a little dumb.
```

### 6.1 基础工具函数

```python
def pick(lst):
    return random.choice(lst)
```

从列表里随机选一个。

```python
def pick_n(lst, n):
    return random.sample(lst, min(n, len(lst)))
```

从列表里随机选多个。

```python
def maybe(text, p=0.5):
    return text if random.random() < p else ""
```

以一定概率加入一段文本。

```python
def join_sentences(*parts):
    return " ".join(p.strip() for p in parts if p.strip()).strip()
```

把非空句子拼起来，去掉多余空格。

这些函数让模板生成更灵活，不会每条数据都一模一样。

### 6.2 词汇池

代码定义了很多列表，比如：

```python
TANK_OBJECTS = ["rock", "plant", "castle", ...]
FOOD_TYPES = ["flakes", "pellets", "bloodworms", ...]
WATER_DESCRIPTIONS = ["clear", "fresh", "cool", ...]
ACTIVITIES = ["swimming in circles", "following a bubble", ...]
```

这些列表相当于 Guppy 的世界观素材库。模型之所以总说水、食物、气泡、石头，是因为训练数据就是围绕这些词生成的。

这也解释了 README 里说的：

> personality is baked into the weights

也就是“性格被烤进权重里”。不是靠系统提示词临时命令模型装成鱼，而是训练数据本身就让模型学成了鱼。

### 6.3 主题生成器

一个主题通常分成用户问题和 Guppy 回答。比如 greeting：

```python
def gen_greeting():
    return _make_sample(_user_greeting(), _guppy_greeting(), "greeting")
```

它返回一个 dict：

```python
{
    "input": user_msg,
    "output": guppy_msg,
    "category": category,
}
```

主题很多，包括：

- greeting
- feeling
- temp_hot / temp_cold
- food
- light
- water
- confused
- tank
- noise
- night
- lonely
- bubbles
- glass
- reflection
- breathing
- swimming
- colors
- cat
- rain
- dreams
- joke
- love
- smart
- doctor
- tv

这些主题一起组成约 60 类对话。

### 6.4 格式化成 ChatML-like 文本

核心函数：

```python
def format_sample(s):
    return (
        f"<|im_start|>user\n{s['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{s['output']}<|im_end|>"
    )
```

原始样本：

```python
{
    "input": "hi guppy",
    "output": "hello. the water is nice today.",
    "category": "greeting",
}
```

会变成：

```text
<|im_start|>user
hi guppy<|im_end|>
<|im_start|>assistant
hello. the water is nice today.<|im_end|>
```

这一步非常重要，因为模型训练和推理时必须使用同一种格式。

### 6.5 生成数据集

```python
def generate_dataset(n_samples=60000, eval_ratio=0.05):
```

这个函数会：

1. 把所有主题函数放进 `topics`。
2. 每个主题分配相同权重。
3. 调用每个生成器多次。
4. 打乱样本。
5. 按 `eval_ratio=0.05` 拆成训练集和验证集。
6. 写入 JSONL 文件。

输出文件：

```python
data/train.jsonl
data/eval.jsonl
data/train_openai.jsonl
data/eval_openai.jsonl
```

`train.jsonl` 的每一行类似：

```json
{"text": "<|im_start|>user\nhi guppy<|im_end|>...", "category": "greeting"}
```

`train_openai.jsonl` 则是 OpenAI message 格式：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## 7. `prepare_data.py`：训练 tokenizer，把文本变成 token id

`prepare_data.py` 做两件事：

1. 调用 `generate_dataset()` 生成数据。
2. 用生成的数据训练 BPE tokenizer。

主函数：

```python
def prepare(data_dir=DATA_DIR, n_samples=60000, eval_ratio=0.05):
    os.makedirs(data_dir, exist_ok=True)

    print(f"Generating {n_samples} samples...")
    from .generate_data import generate_dataset
    generate_dataset(n_samples, eval_ratio)
```

然后读取训练和验证文本：

```python
texts = []
for name in ["data/train.jsonl", "data/eval.jsonl"]:
    if os.path.exists(name):
        with open(name) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
```

注意：它训练 tokenizer 时用的是 `text` 字段，也就是完整 ChatML-like 文本。

训练 tokenizer：

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
```

`ByteLevel` 的好处是对各种字符比较稳健。即使遇到生僻字符，也能退回到字节级表示。

训练器：

```python
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    min_frequency=2,
)
```

关键点：

- `vocab_size=4096`：最多学出 4096 个 token。
- `special_tokens`：确保 `<pad>`、`<|im_start|>`、`<|im_end|>` 有固定 id。
- `min_frequency=2`：出现太少的片段不进入词表。

最后保存：

```python
tokenizer.save(save_path)
```

默认路径：

```text
data/tokenizer.json
```

---

## 8. `dataset.py`：把 JSONL 文本变成模型训练用的 batch

PyTorch 训练通常需要 `Dataset` 和 `DataLoader`。

### 8.1 `GuppyDataset`

```python
class GuppyDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: str, max_len: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_len = max_len
        self.samples = []
```

初始化时加载 tokenizer，并准备保存样本。

读取 JSONL：

```python
with open(path) as f:
    for line in f:
        data = json.loads(line)
        ids = self.tokenizer.encode(data["text"]).ids
```

这里把每一条文本变成 token id 列表。

截断：

```python
if len(ids) > max_len:
    ids = ids[:max_len]
```

如果太长，就只保留前 `max_len` 个 token。训练里传入的是 `mc.max_seq_len`，也就是 128。

过滤太短样本：

```python
if len(ids) >= 2:
    self.samples.append(ids)
```

因为训练 next-token 至少需要两个 token：一个输入，一个目标。

### 8.2 `__getitem__`：制造 x 和 y

```python
def __getitem__(self, idx):
    ids = self.samples[idx]
    x = ids[:-1]
    y = ids[1:]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
```

这是语言模型训练的核心。模型输入 `x`，目标是 `y`。

比如：

```text
ids = [1, 88, 64, 779, 278, 2]
x   = [1, 88, 64, 779, 278]
y   = [88, 64, 779, 278, 2]
```

也就是让模型在每个位置预测下一个 token。

### 8.3 `collate_fn`：把不同长度样本补齐

一个 batch 里的样本长度可能不同。PyTorch tensor 需要矩形，所以要 padding。

```python
max_len = max(len(x) for x in xs)
padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
padded_y = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
```

先创建全是 pad id 的矩阵，再把真实 token 填进去：

```python
for i, (x, y) in enumerate(zip(xs, ys)):
    padded_x[i, :len(x)] = x
    padded_y[i, :len(y)] = y
```

模型训练 loss 里 `ignore_index=0`，所以这些 pad 位置不会参与训练。

---

## 9. `model.py`：GuppyLM 的 Transformer 是怎么写的

这是项目最重要的文件。它实现了一个非常朴素的 GPT-like causal Transformer。

### 9.1 总体结构

`GuppyLM` 包含：

```python
self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
self.drop = nn.Dropout(config.dropout)
self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
self.norm = nn.LayerNorm(config.d_model)
self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
self.lm_head.weight = self.tok_emb.weight
```

一步步解释：

1. `tok_emb`：token id → token 向量。
2. `pos_emb`：位置 id → 位置向量。
3. `drop`：训练时 dropout。
4. `blocks`：6 层 Transformer block。
5. `norm`：最终 LayerNorm。
6. `lm_head`：把 384 维隐藏状态映射回 4096 个 token 分数。
7. `lm_head.weight = self.tok_emb.weight`：权重共享。

权重共享的意思是：输入 embedding 和输出分类头使用同一套权重。这在语言模型中很常见，可以减少参数，也让输入/输出 token 空间更一致。

### 9.2 Attention：模型如何看上下文

`Attention` 类：

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
```

`d_model=384`，`n_heads=6`，所以：

```text
head_dim = 384 / 6 = 64
```

多头 attention 的直觉是：不同 head 可以关注不同关系。有的 head 可能更关注消息边界，有的更关注食物词，有的更关注温度词。这个项目没有显式规定 head 学什么，这是训练中自动学出来的。

前向传播：

```python
qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
```

这一步把每个 token 的向量变成三种向量：

- Query：我现在这个位置想找什么信息？
- Key：我这个位置能提供什么标签？
- Value：如果别人关注我，我实际提供什么内容？

attention 分数：

```python
attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

这是 Query 和 Key 的相似度。除以 `sqrt(head_dim)` 是为了稳定数值。

### 9.3 causal mask：为什么模型不能偷看未来

训练语言模型时，第 3 个 token 只能看前 1、2、3 个 token，不能看第 4 个 token。否则模型训练时就作弊了。

`GuppyLM.forward()` 里创建 mask：

```python
mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
```

`torch.tril` 会生成下三角矩阵。概念上：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

第 1 行只能看第 1 个位置；第 2 行能看第 1、2 个位置；以此类推。

在 attention 里应用：

```python
attn = attn.masked_fill(mask == 0, float("-inf"))
attn = self.dropout(F.softmax(attn, dim=-1))
```

被 mask 的位置变成负无穷，softmax 后概率接近 0。这样模型就不能关注未来 token。

### 9.4 FFN：每个位置单独做非线性变换

```python
class FFN(nn.Module):
    def __init__(self, config):
        self.up = nn.Linear(config.d_model, config.ffn_hidden)
        self.down = nn.Linear(config.ffn_hidden, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.relu(self.up(x))))
```

FFN 的作用是增强每个位置的表示能力。attention 负责“从上下文拿信息”，FFN 负责“对拿到的信息做加工”。

这里是：

```text
384 维 → 768 维 → ReLU → 384 维
```

### 9.5 Block：Attention + FFN + 残差连接 + LayerNorm

```python
class Block(nn.Module):
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

这是一个 pre-norm Transformer block。

关键点：

- `self.norm1(x)`：先归一化，再做 attention。
- `x + ...`：残差连接，帮助深层网络训练稳定。
- `self.norm2(x)`：再归一化。
- `x + self.ffn(...)`：再加一次残差。

模型堆叠了 6 个这样的 block：

```python
self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
```

### 9.6 `forward()`：从 token id 到 loss

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
```

输入 `idx` 是 token id。先加 token embedding 和 position embedding。

然后过 6 层 block：

```python
for block in self.blocks:
    x = block(x, mask)
```

输出 logits：

```python
logits = self.lm_head(self.norm(x))
```

如果传入了 `targets`，就计算 loss：

```python
if targets is not None:
    loss = F.cross_entropy(...)
```

训练时会传 `targets`，推理时不传。

### 9.7 `generate()`：逐 token 生成回复

推理生成在 `GuppyLM.generate()`：

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -self.config.max_seq_len:]
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :] / temperature
```

每次循环：

1. 只保留最后 `max_seq_len` 个 token。
2. 跑一次模型。
3. 取最后一个位置的 logits。
4. 根据 logits 采样下一个 token。

temperature：

- 小于 1：分布更尖锐，输出更保守。
- 大于 1：分布更平，输出更随机。
- 本项目默认 `0.7`。

top-k：

```python
if top_k > 0:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = float("-inf")
```

意思是只从概率最高的前 `k` 个 token 里采样。默认 `top_k=50`。

采样：

```python
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)
idx = torch.cat([idx, next_id], dim=1)
```

如果生成了结束 token：

```python
if next_id.item() == self.config.eos_id:
    break
```

就停止。

---

## 10. `train.py`：模型是怎么被训练出来的

训练入口：

```python
def train():
    mc = GuppyConfig()
    tc = TrainConfig()
```

`mc` 是模型配置，`tc` 是训练配置。

### 10.1 选择设备

```python
def get_device(config):
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
```

优先顺序：

1. NVIDIA GPU：`cuda`
2. Apple Silicon：`mps`
3. CPU

### 10.2 学习率调度

```python
def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * coeff
```

这叫 warmup + cosine decay。

- warmup：刚开始训练不稳定，学习率从小慢慢升到 `3e-4`。
- cosine decay：后期逐渐降到 `3e-5`，让训练更细致。

### 10.3 加载数据

```python
train_loader = get_dataloader(
    os.path.join(tc.data_dir, "train.jsonl"), tokenizer_path,
    mc.max_seq_len, tc.batch_size, shuffle=True,
)
```

训练集 shuffle，验证集不 shuffle：

```python
eval_loader = get_dataloader(..., shuffle=False)
```

### 10.4 优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=tc.learning_rate,
    weight_decay=tc.weight_decay, betas=(0.9, 0.95),
)
```

AdamW 是训练 Transformer 常见优化器。它会根据梯度的一阶、二阶统计自适应更新参数。`weight_decay=0.1` 是权重衰减，帮助减少过拟合。

### 10.5 AMP 混合精度

```python
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda") if use_amp else None
```

如果是 CUDA GPU，就用自动混合精度：

```python
with torch.amp.autocast("cuda"):
    _, loss = model(x, y)
```

混合精度可以让 GPU 训练更快、更省显存。CPU 和 MPS 路径不启用这里的 AMP。

### 10.6 训练循环

核心循环：

```python
while step < tc.max_steps:
    for x, y in train_loader:
        if step >= tc.max_steps:
            break
```

每个 batch：

1. 把 `x` 和 `y` 放到设备上。
2. 计算当前学习率。
3. 前向计算 loss。
4. 反向传播。
5. 梯度裁剪。
6. optimizer 更新参数。
7. 清空梯度。

CPU/MPS 路径：

```python
_, loss = model(x, y)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
optimizer.step()
```

CUDA AMP 路径：

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
scaler.step(optimizer)
scaler.update()
```

### 10.7 评估和保存

每 200 步评估一次：

```python
if step > 0 and step % tc.eval_interval == 0:
    el = evaluate(model, eval_loader, device)
```

如果验证 loss 更低，就保存 best model：

```python
if el < best_eval:
    best_eval = el
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": vars(mc),
        "eval_loss": el,
    }, os.path.join(tc.output_dir, "best_model.pt"))
```

这里保存的不是整个 Python 对象，而是：

- 当前 step
- 模型参数 `state_dict`
- 模型配置 `config`
- 验证 loss

最后还会保存：

```python
final_model.pt
```

里面额外包含 `train_losses`。

---

## 11. `inference.py`：如何从用户输入生成鱼的回复

推理类：

```python
class GuppyInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
```

### 11.1 加载 tokenizer

```python
self.tokenizer = Tokenizer.from_file(tokenizer_path)
```

默认路径是：

```text
data/tokenizer.json
```

### 11.2 加载 checkpoint

```python
ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
```

默认路径是：

```text
checkpoints/best_model.pt
```

然后兼容两种格式：

```python
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
else:
    state_dict = ckpt
```

第一种是本项目训练保存的格式。第二种是 HuggingFace 导出的纯 state dict 格式。

### 11.3 加载配置

它优先找 checkpoint 同目录下的 `config.json`：

```python
config_path = os.path.join(config_dir, "config.json")
```

如果存在，就支持 HF 风格字段：

```python
vocab_size=cfg.get("vocab_size", 4096)
max_seq_len=cfg.get("max_position_embeddings", cfg.get("max_seq_len", 128))
d_model=cfg.get("hidden_size", cfg.get("d_model", 384))
```

如果没有 `config.json`，但 checkpoint 里有 `config`，就从 checkpoint 读取。

这里有一个项目细节：`train.py` 写出的 `checkpoints/config.json` 是嵌套结构：

```json
{"model": {...}, "train": {...}}
```

而 `inference.py` 读取 `config.json` 时期待的是扁平字段或 HF 字段。默认配置下没问题，因为 fallback 默认值正好相同；但如果你改了模型尺寸，最好确保读取逻辑也同步更新，或者使用 checkpoint 内嵌的 `config`。

### 11.4 格式化 prompt

```python
def _format_prompt(self, messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
```

用户输入：

```python
[{"role": "user", "content": "hi guppy"}]
```

会变成：

```text
<|im_start|>user
hi guppy<|im_end|>
<|im_start|>assistant
```

注意最后没有 assistant 的 `<|im_end|>`。因为这是让模型从这里开始续写。

### 11.5 生成回复

```python
input_ids = self.tokenizer.encode(prompt).ids
input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
output_t, _ = self.model.generate(input_t, max_tokens, temperature, top_k)
```

模型生成的是完整序列：输入 prompt + 新生成 token。所以代码只解码新生成部分：

```python
output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])
```

然后做边界截断：

```python
if "<|im_end|>" in output_text:
    output_text = output_text.split("<|im_end|>")[0]
if "<|im_start|>" in output_text:
    output_text = output_text.split("<|im_start|>")[0]
```

这一步很重要。小模型有时可能会继续生成下一轮对话标记，比如又冒出一个 `<|im_start|>user`。截断可以防止它“泄漏到下一轮”。

返回格式：

```python
return {
    "choices": [{
        "message": {"role": "assistant", "content": resp_text},
    }],
}
```

这是一个类似 OpenAI Chat Completion 的结构。

---

## 12. `__main__.py`：命令行入口如何分发

当你运行：

```bash
python -m guppylm chat
```

Python 会执行 `guppylm/__main__.py`。

它支持四个命令：

```python
python -m guppylm train
python -m guppylm prepare
python -m guppylm chat
python -m guppylm download
```

分发逻辑很直接：

```python
cmd = sys.argv[1]
sys.argv = sys.argv[1:]
```

然后：

```python
if cmd == "prepare":
    from .prepare_data import prepare
    prepare()
elif cmd == "train":
    from .train import train
    train()
elif cmd == "download":
    download_model()
elif cmd == "chat":
    from .inference import main as inference_main
    inference_main()
```

`download_model()` 会从 HuggingFace 下载：

```python
files = [
    (f"{HF_BASE}/pytorch_model.bin", CHECKPOINT_PATH),
    (f"{HF_BASE}/tokenizer.json", TOKENIZER_PATH),
    (f"{HF_BASE}/config.json", "checkpoints/config.json"),
]
```

这里把 HF 的 `pytorch_model.bin` 保存成本地 `checkpoints/best_model.pt`，这样后续 `chat` 仍然使用统一默认路径。

---

## 13. `eval_cases.py`：这个项目如何“评估”模型

这个仓库没有 pytest/unittest 自动测试。它有一个手写的评估样例文件：

```python
EVAL_CASES = [
    {
        "id": "greeting_basic",
        "category": "greeting",
        "prompt": "hi guppy",
        "expect_keywords": ["hello", "hi", "water", "swim", "bubble"],
        "expect_style": "lowercase, short, friendly",
    },
]
```

这些不是严格的单元测试，而是人工检查标准：

- 回复是否包含合理关键词？
- 是否保持小写、短句、友好？
- 是否符合鱼的世界观？
- 遇到抽象问题是否表现出“不懂人类概念”？

对生成模型来说，这类评估很常见。因为同一个 prompt 可以有多个合理答案，不能只用字符串完全相等来判断。

---

## 14. `tools/`：导出和发布脚本

### 14.1 `tools/export_model.py`：导出 HuggingFace 模型格式

这个脚本把训练 checkpoint 转成 HuggingFace 常见布局：

```text
hf_export/pytorch_model.bin
hf_export/config.json
hf_export/tokenizer.json
hf_export/README.md
hf_export/assets/guppy.png
```

核心逻辑：

```python
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = ckpt["config"]
state_dict = ckpt["model_state_dict"]
```

注意它期待的是本项目训练出的 legacy checkpoint，也就是必须有：

```text
model_state_dict
config
```

它保存纯参数：

```python
torch.save(state_dict, model_path)
```

并写 HF 风格 `config.json`：

```python
hf_config = {
    "model_type": "guppylm",
    "architectures": ["GuppyLM"],
    "vocab_size": config["vocab_size"],
    "max_position_embeddings": config["max_seq_len"],
    "hidden_size": config["d_model"],
    ...
}
```

如果传了 token 和 repo，它还会上传到 HuggingFace。

### 14.2 `tools/export_dataset.py`：导出数据集

这个脚本生成数据集并保存到本地：

```python
save_local(train_data, test_data, args.output_dir)
```

默认输出：

```text
dataset/train.jsonl
dataset/test.jsonl
```

如果不是 `--local-only`，还会推送到 HuggingFace Dataset repo。

一个细节：docstring 里写的是 `HF_REPO`，但代码实际读取：

```python
repo = args.repo or os.environ.get("HF_DATASET")
```

所以数据集导出要用 `HF_DATASET`。

### 14.3 `tools/export_onnx.py`：导出浏览器模型

浏览器不能直接运行 PyTorch checkpoint。这个脚本把模型导出成 ONNX：

```python
torch.onnx.export(
    model,
    (dummy_input,),
    output_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    },
    opset_version=17,
)
```

它只导出模型 forward，也就是：

```text
input_ids → logits
```

采样循环是在浏览器 JavaScript 里重新实现的。

默认还会量化：

```python
quantize_dynamic(fp32_path, output_path, weight_type=QuantType.QUInt8)
```

量化的直觉是：把权重从 float32 压缩成更小的整数表示，文件变小，浏览器下载更快。README 里说大约从 35MB 到 9-10MB。

---

## 15. `docs/index.html`：浏览器里怎么跑小模型

浏览器 demo 是一个单文件应用。它会加载：

```javascript
import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.mjs")
```

然后加载 tokenizer 和模型：

```javascript
const tokResp = await fetch(`${MODEL_BASE}/tokenizer.json`);
const modelResp = await fetch(`${MODEL_BASE}/model.onnx`);
session = await ort.InferenceSession.create(modelBuf, {
  executionProviders: ["wasm"],
});
```

### 15.1 JavaScript tokenizer

浏览器不能直接调用 Python tokenizer，所以 `index.html` 里手写了一个 ByteLevel BPE tokenizer 的 JS 版本：

```javascript
function buildTokenizer(json) {
  const vocab = json.model.vocab;
  const merges = json.model.merges;
  ...
  return { encode, decode };
}
```

它读取 `tokenizer.json` 里的：

- vocab：token 到 id 的映射。
- merges：BPE 合并规则。
- added_tokens：特殊 token。

### 15.2 浏览器生成循环

浏览器推理函数：

```javascript
async function generate(inputIds) {
  let ids = inputIds.slice();

  for (let i = 0; i < GEN.max_tokens; i++) {
    const seq = ids.slice(-CONFIG.max_seq_len);
    const tensor = new ort.Tensor("int64",
      BigInt64Array.from(seq.map(BigInt)), [1, seq.length]);
    const out = await session.run({ input_ids: tensor });
    const logits = out.logits.data;
    ...
  }
}
```

这和 Python 的 `GuppyLM.generate()` 思路一样：

1. 保留最后 128 token。
2. 跑 ONNX 模型得到 logits。
3. 取最后一个位置的 logits。
4. temperature 调整。
5. top-k 截断。
6. softmax 得到概率。
7. 随机采样下一个 token。
8. 遇到 eos 停止。

浏览器里的 prompt 格式也和 Python 一致：

```javascript
const prompt = `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
```

生成后同样截断：

```javascript
if (reply.includes("<|im_end|>")) reply = reply.split("<|im_end|>")[0];
if (reply.includes("<|im_start|>")) reply = reply.split("<|im_start|>")[0];
```

---

## 16. 为什么这个模型很小，但仍然能“像鱼说话”

你可以从三个层面理解：

### 16.1 任务非常窄

它不是要回答所有问题，而是只要像鱼一样回应常见话题。训练分布很集中：水、食物、鱼缸、光、气泡、猫、天气、梦、爱、笑话。

任务越窄，需要的模型能力越少。

### 16.2 数据风格非常稳定

训练样本都遵守类似风格：

- 短句。
- 小写。
- 鱼的视角。
- 不理解复杂人类概念。
- 经常回到水和食物。

模型不需要从混乱数据里抽象出复杂人格，而是在大量一致模板里学习稳定模式。

### 16.3 模型结构虽小但完整

GuppyLM 仍然具备 GPT 类模型的关键部件：

- token embedding
- position embedding
- causal self-attention
- feed-forward network
- residual connection
- LayerNorm
- next-token cross entropy training
- autoregressive generation

所以它是“小”，不是“假”。它确实是一套完整语言模型训练/推理流程。

---

## 17. 初学者最容易误解的地方

### 17.1 “模型是不是理解了鱼？”

严格说，它不是像人一样理解鱼。它学到的是：在这种文本上下文里，哪些 token 接下来更可能出现。

但如果训练数据非常一致，这种 next-token prediction 会表现得像“角色理解”。例如用户说 `are you hungry`，训练集中这类问题后面经常出现 `food`、`flakes`、`yes`，于是模型也会生成类似回复。

### 17.2 “为什么不用 system prompt？”

大模型可以通过 system prompt 临时改变行为。但这个模型只有约 9M 参数，上下文也只有 128 tokens。README 里明确说：它不能很好地条件化遵循复杂指令。

所以项目选择把性格写进训练数据，而不是每次推理靠 prompt 告诉它“你是一条鱼”。

### 17.3 “为什么多轮聊天会变差？”

上下文只有 128 tokens。多轮聊天会把历史都塞进 prompt，很快占满上下文。模型能看到的有效信息变少，而且训练主要是单轮样本，所以多轮分布和训练分布不一致。

因此单轮更可靠。

### 17.4 “为什么 tokenizer 要自己训练？”

因为这是从零训练的小模型。tokenizer 决定文本如何拆成 token。如果直接用别的大模型 tokenizer，也能做，但会引入额外复杂性。

本项目训练自己的 4096 BPE vocab，更符合教学目标：从原始文本到 tokenizer 到模型都自己做一遍。

### 17.5 “为什么训练目标不是直接学习问答？”

它实际上就是通过 next-token prediction 学问答。只要训练文本格式是：

```text
user 问题 + assistant 答案
```

模型学习下一个 token 时，自然会学到“问题后面应该接答案”。

---

## 18. 如果你想修改这个项目，应该从哪里开始

### 18.1 想改 Guppy 的性格

主要改：

```text
guppylm/generate_data.py
```

比如增加新的词汇池、改回答模板、增加主题生成器。

改完后需要重新运行：

```bash
python -m guppylm prepare
python -m guppylm train
```

因为性格来自训练数据，不是来自推理时的一句提示词。

### 18.2 想改模型大小

主要改：

```text
guppylm/config.py
```

例如：

- `d_model`
- `n_layers`
- `n_heads`
- `ffn_hidden`
- `max_seq_len`

但要注意：`docs/index.html` 里也写死了一份浏览器配置：

```javascript
const CONFIG = {
  vocab_size: 4096, max_seq_len: 128, d_model: 384,
  n_layers: 6, n_heads: 6, ffn_hidden: 768,
  pad_id: 0, bos_id: 1, eos_id: 2,
};
```

如果你要导出浏览器版本，必须保持这些配置和模型一致。

### 18.3 想改训练方式

主要看：

```text
guppylm/train.py
```

可以调整：

- 学习率调度。
- optimizer。
- eval/save interval。
- AMP 策略。
- checkpoint 保存内容。

### 18.4 想改聊天接口

主要看：

```text
guppylm/inference.py
```

尤其是：

- `_format_prompt()`：prompt 格式。
- `chat_completion()`：生成参数、截断逻辑、返回格式。

不要轻易删除 `<|im_end|>` 和 `<|im_start|>` 截断逻辑，否则小模型可能生成越界内容。

### 18.5 想改浏览器 demo

主要看：

```text
docs/index.html
```

这个文件同时包含：

- 页面样式。
- 主题切换。
- tokenizer JS 实现。
- ONNX Runtime 加载。
- 自回归生成。
- 聊天 UI。
- topic hint 弹窗。

---

## 19. 常用命令速查

安装：

```bash
python -m pip install -r requirements.txt
```

下载预训练模型：

```bash
python -m guppylm download
```

单轮聊天：

```bash
python -m guppylm chat --prompt "tell me a joke"
```

生成数据和 tokenizer：

```bash
python -m guppylm prepare
```

训练：

```bash
python -m guppylm train
```

重新生成 Colab notebook：

```bash
make notebook
```

导出 HuggingFace 模型格式到本地：

```bash
python tools/export_model.py --local-only
```

导出数据集到本地：

```bash
python tools/export_dataset.py --local-only
```

导出 ONNX 浏览器模型：

```bash
python tools/export_onnx.py
```

本地预览浏览器 demo：

```bash
python3 -m http.server 8080 --directory docs
```

---

## 20. 一句话总结整条链路

GuppyLM 的本质是：用模板生成大量“用户问小鱼、小鱼回答”的文本，把文本切成 token，用一个小型 causal Transformer 学习 next-token prediction，训练好后通过同样的 ChatML-like prompt 格式逐 token 生成回复；因为训练数据风格高度一致，所以模型虽然很小，也能稳定表现出“鱼缸小鱼”的角色感。
