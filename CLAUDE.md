# hftbacktest 子项目规范

## 一、项目背景

### 来源
Fork 自 [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest)，维护于 [AaronJckson/hftbacktest](https://github.com/AaronJckson/hftbacktest)。

### 用途
作为 HFT-Arbitrage 项目的回测引擎底层，替代 pip 安装的公开版本。修改后的版本通过 `maturin develop --release` 编译安装，供上层策略代码直接 `from hftbacktest import ...` 使用，上层调用方式无需改动。

### 核心优化方向
**撮合逻辑优化** — 针对加密货币高频回测场景，修改 Rust 实现的撮合引擎，使其更贴近真实市场行为。

---

## 二、仓库结构

```
hftbacktest/
├── hftbacktest/src/              # 核心 Rust library crate（撮合引擎主体）
│   ├── backtest/
│   │   ├── proc/                 # 撮合处理器（核心修改区域）
│   │   │   ├── nopartialfillexchange.rs   # 不允许部分成交的交易所模型
│   │   │   ├── partialfillexchange.rs     # 允许部分成交的交易所模型
│   │   │   ├── local.rs                   # 本地订单管理（策略侧视角）
│   │   │   ├── l3_local.rs                # L3 行情下的本地订单管理
│   │   │   └── l3_nopartialfillexchange.rs # L3 行情下的交易所模型
│   │   ├── models/
│   │   │   ├── queue.rs          # 队列位置估算模型（QueueModel trait + 各实现）
│   │   │   ├── latency.rs        # 延迟模型
│   │   │   └── fee.rs            # 手续费模型
│   │   ├── data/                 # 行情数据读取（npy/npz 格式解析）
│   │   ├── state.rs              # 账户状态（仓位、资金）
│   │   └── recorder.rs           # 回测过程记录
│   ├── depth/                    # 订单簿数据结构
│   │   ├── hashmapmarketdepth.rs # HashMap 实现的 L2 订单簿
│   │   ├── btreemarketdepth.rs   # BTreeMap 实现的 L2 订单簿
│   │   └── roivectormarketdepth.rs # ROI Vector 实现的订单簿
│   ├── types.rs                  # 核心类型定义（Order, Event, Side 等）
│   └── lib.rs                    # crate 入口
└── py-hftbacktest/               # Python 绑定层（maturin 构建）
    ├── src/                      # PyO3 绑定代码（infra，非业务修改区）
    ├── hftbacktest/              # Python 层封装（__init__.py, recorder.py 等）
    └── pyproject.toml            # maturin 构建配置
```

---

## 三、撮合引擎核心逻辑

### 撮合处理器（`backtest/proc/`）

两个主要实现，区别在于是否支持部分成交：

| 文件 | 说明 |
|------|------|
| `nopartialfillexchange.rs` | 不支持部分成交，订单要么全成交要么不成交 |
| `partialfillexchange.rs` | 支持部分成交，更接近真实交易所行为 |

**`NoPartialFillExchange` 成交条件：**
- 买单：`order.price >= best_ask`，或 `order.price > trade_price`，或 `order.price == trade_price` 且在队列最前
- 卖单：`order.price <= best_bid`，或 `order.price < trade_price`，或 `order.price == trade_price` 且在队列最前
- 市价单/流动性吃单：直接以 best price 全量成交（不考虑 best 档位的数量）

### 队列位置模型（`backtest/models/queue.rs`）

`QueueModel` trait 定义队列估算接口，核心方法：

| 方法 | 触发时机 | 作用 |
|------|----------|------|
| `new_order` | 订单进入订单簿时 | 初始化队列位置 |
| `trade` | 同价位发生成交时 | 减少队列前方数量 |
| `depth` | 同价位深度变化时 | 更新队列位置估算 |
| `is_filled` | 每个 tick 检查时 | 判断订单是否应成交，返回成交量 |

内置实现：
- `RiskAdverseQueueModel` — 保守模型，仅当同价位有成交时队列才前进
- `ProbabilisticSizeClassQueueModel` — 基于概率的队列模型
- `SquareProbQueueModel` / `PowerProbQueueModel` — 不同概率函数形状的变体

---

## 四、开发工作流

### 修改 Rust 代码后的本地编译

```bash
cd /data/Hftbacktest_strategy/hftbacktest/py-hftbacktest
maturin develop --release -m pyproject.toml
```

编译完成后，`from hftbacktest import ...` 自动使用本地修改版本，无需改动上层代码。

### 提交到 GitHub

```bash
cd /data/Hftbacktest_strategy/hftbacktest
git add <修改文件>
git commit -m "..."
git push origin main
```

### 服务器侧更新

```bash
# 服务器上 pull 并重新编译
cd /home/yhh/Alibaba_HFT/Hftbacktest_strategy/hftbacktest/py-hftbacktest
git pull
/home/yhh/miniconda3/envs/py312/bin/python3.12 -m maturin develop --release -m pyproject.toml
```

---

## 五、修改注意事项

1. **`hftbacktest/` crate 是业务修改区**，`py-hftbacktest/src/` 是 PyO3 绑定层（infra），原则上不动
2. 修改撮合逻辑后须验证：正常 limit order 成交、队列位置估算、部分成交场景（如适用）
3. Rust 编译耗时较长，修改前先在逻辑层面确认思路，再动手
4. 禁止在未与 Aaron 确认修改计划前直接改动框架性结构
