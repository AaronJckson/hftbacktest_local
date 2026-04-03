"""
test_partialfillexchange.py — PartialFillExchange bug 修复回归测试

覆盖范围:
  TC1 — Bug 1  (local.rs  #299): PartiallyFilled 订单须触发本地 apply_fill
  TC2 — Bug 2-5 (exchange #301): lot-size 取整判断防止已 Filled 订单残留 book（僵尸订单）
  TC3 — Bug 7-8 (exchange #273): FOK 浮点边界不 panic
  TC4 — 回归:  NoPartialFillExchange 行为不受影响
  TC5 — FOK exec_qty 合并修复: 多档扫单时 apply_fill 须使用累积总量而非最后一档

测试日期范围: 20250901 ~ 20250902（2 天）

运行方式（服务器，需先 maturin develop --release 编译）:
    cd /home/yhh/Alibaba_HFT/Hftbacktest_strategy
    /home/yhh/miniconda3/envs/py312/bin/python3.12 -m pytest \
        hftbacktest/py-hftbacktest/tests/test_partialfillexchange.py -v
"""

import os
import sys
import unittest

import numpy as np
from numba import njit

# 确保 hftbacktest 包可被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hftbacktest import (
    BacktestAsset,
    HashMapMarketDepthBacktest,
    ALL_ASSETS,
    BUY, SELL,
    GTC, GTX,
    LIMIT,
    FILLED,
)
from hftbacktest.order import PARTIALLY_FILLED, FOK

# ---------------------------------------------------------------------------
# 数据与资产参数（与 cross_arbitrage/config.py 保持一致）
# ---------------------------------------------------------------------------

_DATA_ROOT   = '/Volumes/newhomes/NLshare/market_data/incremental_l2_npz'
_COIN_DIR    = os.path.join(_DATA_ROOT, 'DOGEUSDT')

# 测试日期范围：含首尾两端
_START_DATE  = '20250901'
_END_DATE    = '20250902'

_TICK_SIZE     = 0.00001
_LOT_SIZE      = 1.0
_MAKER_FEE     = -0.00005
_TAKER_FEE     =  0.00013
_ROI_LB        =  0.2
_ROI_UB        =  0.31
_FEED_LATENCY  = 2_000_000   # 2 ms
_ORDER_LATENCY = 2_000_000   # 2 ms
_TRADES_CAP    = 50_000


def _find_data_files() -> list[str]:
    """
    返回 [_START_DATE, _END_DATE] 范围内所有 npz 文件路径（有序列表）。
    文件命名约定与 data/loader.py 一致：{YYYYMMDD}_{symbol}.npz。
    """
    if not os.path.isdir(_COIN_DIR):
        return []
    files = []
    for fname in sorted(os.listdir(_COIN_DIR)):
        if not fname.endswith('.npz'):
            continue
        # 取下划线前的日期部分，与 loader.py 解析逻辑一致
        file_date = fname.split('_')[0]
        if _START_DATE <= file_date <= _END_DATE:
            files.append(os.path.join(_COIN_DIR, fname))
    return files


def _make_asset(partial_fill: bool, data_files: list[str]) -> BacktestAsset:
    """构造 BacktestAsset，接受多日文件列表，根据 partial_fill 选择撮合模型。"""
    asset = (
        BacktestAsset()
        .data(data_files)
        .linear_asset(1.0)
        .risk_adverse_queue_model()
        .trading_value_fee_model(_MAKER_FEE, _TAKER_FEE)
        .tick_size(_TICK_SIZE)
        .lot_size(_LOT_SIZE)
        .roi_lb(_ROI_LB)
        .roi_ub(_ROI_UB)
        .last_trades_capacity(_TRADES_CAP)
        .latency_offset(_FEED_LATENCY)
        .constant_order_latency(_ORDER_LATENCY, _ORDER_LATENCY)
    )
    if partial_fill:
        asset = asset.partial_fill_exchange()
    else:
        asset = asset.no_partial_fill_exchange()
    return asset


# ===========================================================================
# TC1 策略：持续提交 GTC crossing 大单，追踪 leaves_qty 验证 position 更新
# ===========================================================================

@njit
def _tc1_run(hbt, results):
    """
    策略逻辑（持续 6 天）：
      - 每当无活跃订单时，提交 GTC buy（价格高于 best_ask），qty=5000 lots
      - GTC crossing：立即以 taker 身份消耗 ask 深度 → PartiallyFilled
      - 剩余量挂回 book，后续 sell trade 继续部分成交
      - 订单完全成交后清理并立即提交下一笔，覆盖全部 6 天数据
      - 每步通过 leaves_qty 变化量累积手动仓位，与 state.position 对比

    results[0] = state.position（本地状态，由 apply_fill 更新）
    results[1] = 手动累积仓位（由 leaves_qty delta 计算）
    results[2] = 检测到成交变化的步数
    """
    ORDER_QTY      = 5000.0
    order_id_counter = 1      # 每笔新订单分配唯一 ID，避免 ID 复用导致查找混乱
    current_order_id = 0
    placed           = False
    manual_position  = 0.0
    fill_steps       = 0
    prev_leaves_qty  = 0.0

    while hbt.elapse(1_000_000_000) == 0:   # 1-second steps，跑满全部 6 天数据
        depth = hbt.depth(0)

        # 无有效盘口时跳过，等待数据就绪
        if depth.best_ask_tick <= 0:
            continue

        # 无活跃订单时，提交新的 GTC buy（价格高于 best_ask 2 tick，触发 crossing）
        if not placed:
            current_order_id  = order_id_counter
            order_id_counter += 1
            entry_price = depth.best_ask + 2.0 * _TICK_SIZE
            hbt.submit_buy_order(0, current_order_id, entry_price, ORDER_QTY, GTC, LIMIT, False)
            placed          = True
            prev_leaves_qty = ORDER_QTY

        # 遍历本地订单，通过 leaves_qty delta 检测每步的成交量
        orders = hbt.orders(0)
        values = orders.values()
        found  = False
        while True:
            order = values.next()
            if order is None:
                break
            if order.order_id == current_order_id:
                found      = True
                cur_leaves = order.leaves_qty
                # 阈值 half-lot 过滤 FP 噪声，真实成交量均为 lot_size 整数倍
                if cur_leaves < prev_leaves_qty - _LOT_SIZE / 2.0:
                    manual_position += prev_leaves_qty - cur_leaves
                    fill_steps      += 1
                    prev_leaves_qty  = cur_leaves
                # 完全成交后清理，下一步提交新订单
                if cur_leaves < _LOT_SIZE / 2.0:
                    placed = False
                    hbt.clear_inactive_orders(ALL_ASSETS)
                break

        # 订单不在 orders 中（外部意外清除）时，同样重置，防止卡死
        if not found and placed:
            placed = False

    # 循环结束后补捉：数据末尾最后一次 elapse() 内处理的成交在 loop body 执行前已到达，
    # apply_fill 已更新 state.position，但 manual_position 未累积，需在此补齐。
    if placed:
        orders = hbt.orders(0)
        values = orders.values()
        while True:
            order = values.next()
            if order is None:
                break
            if order.order_id == current_order_id:
                cur_leaves = order.leaves_qty
                if cur_leaves < prev_leaves_qty - _LOT_SIZE / 2.0:
                    manual_position += prev_leaves_qty - cur_leaves
                    fill_steps      += 1
                break

    state = hbt.state_values(0)
    results[0] = state.position
    results[1] = manual_position
    results[2] = float(fill_steps)


# ===========================================================================
# TC2 策略：激进挂单压力测试，验证无僵尸订单（Bugs 2-5）
# ===========================================================================

@njit
def _tc2_run(hbt):
    """
    轮流提交在 best_ask/best_bid 的 GTC limit 单，覆盖：
      - Ordering::Greater/Less（crossing 路径，Bugs 2-3）
      - Ordering::Equal（同价位队列路径，Bugs 4-5）

    若有僵尸订单，下一笔 trade 事件会对已 Filled 的订单调用 fill()，
    返回 InvalidOrderStatus 错误，导致 hbt.elapse() 抛出异常。
    测试通过 = 全程无异常。
    """
    order_id = 0
    step = 0

    while hbt.elapse(5_000_000_000) == 0:   # 5-second steps，跑满全部 6 天数据
        step += 1
        depth = hbt.depth(0)

        if depth.best_bid_tick <= 0 or depth.best_ask_tick <= 0:
            continue

        # 交替挂 buy/sell，使用不同价格覆盖三条成交路径
        mod = step % 6
        if mod == 1:
            # GTC buy @ best_ask → crossing，触发 Ordering::Greater 路径后挂 book
            hbt.submit_buy_order(0, order_id, depth.best_ask, 10.0, GTC, LIMIT, False)
            order_id += 1
        elif mod == 2:
            # GTC sell @ best_bid → crossing，触发 Ordering::Less 路径后挂 book
            hbt.submit_sell_order(0, order_id, depth.best_bid, 10.0, GTC, LIMIT, False)
            order_id += 1
        elif mod == 4:
            # GTC buy @ best_bid - 1 tick → 纯 maker 挂单，等 Ordering::Equal 队列成交
            hbt.submit_buy_order(
                0, order_id,
                depth.best_bid - _TICK_SIZE,
                10.0, GTC, LIMIT, False,
            )
            order_id += 1
        elif mod == 5:
            # GTC sell @ best_ask + 1 tick → 纯 maker 挂单，等 Ordering::Equal 队列成交
            hbt.submit_sell_order(
                0, order_id,
                depth.best_ask + _TICK_SIZE,
                10.0, GTC, LIMIT, False,
            )
            order_id += 1

        hbt.clear_inactive_orders(ALL_ASSETS)


# ===========================================================================
# TC3 策略：周期提交 FOK 订单，验证浮点边界不 panic（Bugs 7-8）
# ===========================================================================

@njit
def _tc3_run(hbt):
    """
    每 30 秒提交一次 FOK buy @ best_ask。
    - 若 ask 深度充足 → 全量成交（Filled）
    - 若不足         → Expired（FOK 语义）
    测试通过 = 全程无异常（正常 FOK 路径回归验证）。

    注：Bugs 7-8 的核心触发条件是 lot_size 为非整数时 fill 循环因 FP 累积
    产生 sub-lot 残差，导致原始 unreachable!() panic。当前数据（DOGEUSDT，
    lot_size=1.0，book qty 为整数）无法触发该路径；FP 安全兜底逻辑的专项
    验证需在 Rust 单元测试层以合成非整数 depth 数据覆盖，超出本集成测试范围。
    """
    order_id = 0
    step = 0

    while hbt.elapse(10_000_000_000) == 0:   # 10-second steps，跑满全部 6 天数据
        step += 1
        depth = hbt.depth(0)

        if depth.best_ask_tick <= 0:
            continue

        if step % 3 == 0:
            # 提交 FOK，qty=1 lot，价格在 best_ask
            hbt.submit_buy_order(0, order_id, depth.best_ask, 1.0, FOK, LIMIT, False)
            order_id += 1

        hbt.clear_inactive_orders(ALL_ASSETS)


# ===========================================================================
# TC4 策略：NoPartialFillExchange 持续下单回归验证
# ===========================================================================

@njit
def _tc4_run(hbt, results):
    """
    使用与 TC1 相同的持续下单逻辑，但底层为 NoPartialFillExchange。
    NoPartialFill 语义：订单要么全量成交（leaves_qty → 0），要么不成交。
    一笔成交对应 leaves_qty 一步降至 0，fill_steps 每次 +1。

    验证：
      - 回测正常完成（无异常）
      - state.position 与手动累积量完全一致
    """
    ORDER_QTY        = 500.0   # NoPartialFill 全量成交，用较小 qty 控制单笔仓位增量
    order_id_counter = 1
    current_order_id = 0
    placed           = False
    manual_position  = 0.0
    fill_steps       = 0
    prev_leaves_qty  = 0.0

    while hbt.elapse(1_000_000_000) == 0:   # 1-second steps，跑满全部 6 天数据
        depth = hbt.depth(0)

        if depth.best_ask_tick <= 0:
            continue

        if not placed:
            current_order_id  = order_id_counter
            order_id_counter += 1
            entry_price = depth.best_ask + 2.0 * _TICK_SIZE
            hbt.submit_buy_order(0, current_order_id, entry_price, ORDER_QTY, GTC, LIMIT, False)
            placed          = True
            prev_leaves_qty = ORDER_QTY

        orders = hbt.orders(0)
        values = orders.values()
        found  = False
        while True:
            order = values.next()
            if order is None:
                break
            if order.order_id == current_order_id:
                found      = True
                cur_leaves = order.leaves_qty
                if cur_leaves < prev_leaves_qty - _LOT_SIZE / 2.0:
                    manual_position += prev_leaves_qty - cur_leaves
                    fill_steps      += 1
                    prev_leaves_qty  = cur_leaves
                if cur_leaves < _LOT_SIZE / 2.0:
                    placed = False
                    hbt.clear_inactive_orders(ALL_ASSETS)
                break

        if not found and placed:
            placed = False

    # 循环结束后补捉：数据末尾最后一次 elapse() 内处理的成交在 loop body 执行前已到达
    if placed:
        orders = hbt.orders(0)
        values = orders.values()
        while True:
            order = values.next()
            if order is None:
                break
            if order.order_id == current_order_id:
                cur_leaves = order.leaves_qty
                if cur_leaves < prev_leaves_qty - _LOT_SIZE / 2.0:
                    manual_position += prev_leaves_qty - cur_leaves
                    fill_steps      += 1
                break

    state = hbt.state_values(0)
    results[0] = state.position
    results[1] = manual_position
    results[2] = float(fill_steps)


# ===========================================================================
# TC5 策略：FOK 多档扫单，验证 exec_qty 正确合并
# ===========================================================================

@njit
def _tc5_run(hbt, results):
    """
    策略逻辑（持续 2 天）：
      - 每步提交 FOK buy（price = best_ask + 2 tick，qty = ORDER_QTY）
      - 提交当步不检查（回报尚未到达），下一步起检查结果
      - FOK 全量成交（status == FILLED）：累积 manual_position，计入 fill_steps
      - FOK 取消（status == Expired）：直接重置，不计入
      - FOK 扫单跨多档时（best_ask 单档深度 < ORDER_QTY），exec_qty 修复前
        只计最后一档，修复后为全部档位的累积总量

    results[0] = state.position（本地状态，由 apply_fill 更新）
    results[1] = 手动累积仓位（由 leaves_qty → 0 时计 ORDER_QTY）
    results[2] = 实际成交的 FOK 笔数
    """
    ORDER_QTY        = 5000.0
    order_id_counter = 1
    current_order_id = 0
    placed           = False
    submitted_at_step = -2   # 初始化为负数，保证首次检查不误触发
    manual_position  = 0.0
    fill_steps       = 0
    step_counter     = 0

    while hbt.elapse(1_000_000_000) == 0:   # 1-second steps
        step_counter += 1
        depth = hbt.depth(0)

        if depth.best_ask_tick <= 0:
            continue

        # 未持有活跃订单时提交新 FOK
        if not placed:
            current_order_id  = order_id_counter
            order_id_counter += 1
            entry_price = depth.best_ask + 2.0 * _TICK_SIZE
            hbt.submit_buy_order(0, current_order_id, entry_price, ORDER_QTY, FOK, LIMIT, False)
            placed            = True
            submitted_at_step = step_counter

        # 提交当步回报尚未到达，下一步起检查
        # FOK 在回报到达后必然终结（Filled 或 Expired），一步内处理完毕
        if placed and step_counter > submitted_at_step:
            orders = hbt.orders(0)
            values = orders.values()
            found  = False
            while True:
                order = values.next()
                if order is None:
                    break
                if order.order_id == current_order_id:
                    found = True
                    if order.status == FILLED:
                        # FOK 全量成交（status == FILLED）：用 status 而非 leaves_qty 判断，
                        # 避免 FOK safety-fallback Expired 但 leaves_qty 接近 0 时的误判。
                        # 修复前 exec_qty = 最后一档，apply_fill 少算；修复后 = 总量
                        manual_position += ORDER_QTY
                        fill_steps      += 1
                    # 无论 Filled 还是 Expired，FOK 均已终结，立即重置
                    placed = False
                    hbt.clear_inactive_orders(ALL_ASSETS)
                    break
            # 订单不在 orders 中（异常情况），同样重置防止卡死
            if not found:
                placed = False

    # 循环结束后补捉：数据末尾最后一次 elapse() 内处理的 FOK 成交在 loop body 执行前已到达
    if placed:
        orders = hbt.orders(0)
        values = orders.values()
        while True:
            order = values.next()
            if order is None:
                break
            if order.order_id == current_order_id:
                if order.status == FILLED:
                    manual_position += ORDER_QTY
                    fill_steps      += 1
                break

    state = hbt.state_values(0)
    results[0] = state.position
    results[1] = manual_position
    results[2] = float(fill_steps)


# ===========================================================================
# 测试类
# ===========================================================================

class TestPartialFillExchange(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_files = _find_data_files()

    def _require_data(self):
        if not self.data_files:
            self.skipTest(
                f'DOGEUSDT data for {_START_DATE}~{_END_DATE} not found at {_COIN_DIR}. '
                'This test must run on the server where the data is mounted.'
            )

    # ------------------------------------------------------------------
    # TC1 — Bug 1: PartiallyFilled 须触发 apply_fill（持续 6 天验证）
    # ------------------------------------------------------------------
    def test_tc1_partial_fill_updates_local_position(self):
        """
        Bug 1 (local.rs #299):
        state.position 须与手动累积的成交量完全一致。
        修复前：PartiallyFilled 回报不调用 apply_fill → state.position 恒为 0。
        持续 6 天下单，保证足够的部分成交事件覆盖整个测试窗口。
        """
        self._require_data()

        asset = _make_asset(partial_fill=True, data_files=self.data_files)
        hbt = HashMapMarketDepthBacktest([asset])
        results = np.zeros(3, dtype=np.float64)
        _tc1_run(hbt, results)

        state_pos   = results[0]
        manual_pos  = results[1]
        fill_steps  = int(results[2])

        if fill_steps == 0:
            self.skipTest(
                'TC1: 6 天内未检测到任何成交，无法验证 Bug 1 修复（数据或盘口异常）。'
            )

        self.assertGreater(fill_steps, 0,
            f'预期有部分成交，实际 fill_steps={fill_steps}')
        self.assertGreater(state_pos, 0.0,
            f'state.position 应为正值（有 buy 成交），实际={state_pos}')
        self.assertAlmostEqual(
            state_pos, manual_pos, places=6,
            msg=(
                f'state.position ({state_pos:.6f}) 与手动累积仓位 ({manual_pos:.6f}) '
                f'不一致（经历 {fill_steps} 步成交）。'
                'Bug 1 (PartiallyFilled 不触发 apply_fill) 可能未修复。'
            ),
        )

    # ------------------------------------------------------------------
    # TC2 — Bugs 2-5: 无僵尸订单（6 天压力测试）
    # ------------------------------------------------------------------
    def test_tc2_no_zombie_orders_full_day(self):
        """
        Bugs 2-5 (exchange #301):
        激进挂单压力测试跑满 6 天数据，若有僵尸订单会触发 InvalidOrderStatus 异常。
        测试通过 = 全程无异常。
        """
        self._require_data()

        asset = _make_asset(partial_fill=True, data_files=self.data_files)
        hbt = HashMapMarketDepthBacktest([asset])
        try:
            _tc2_run(hbt)
        except Exception as exc:
            self.fail(
                f'TC2: 回测抛出异常，可能存在僵尸订单 → {type(exc).__name__}: {exc}'
            )

    # ------------------------------------------------------------------
    # TC3 — Bugs 7-8: FOK 浮点边界不 panic（6 天验证）
    # ------------------------------------------------------------------
    def test_tc3_fok_no_panic(self):
        """
        Bugs 7-8 (exchange #273):
        FOK 订单在浮点精度边界下不应 panic（原始 unreachable!() 已替换为安全兜底）。
        测试通过 = 全程无异常。
        """
        self._require_data()

        asset = _make_asset(partial_fill=True, data_files=self.data_files)
        hbt = HashMapMarketDepthBacktest([asset])
        try:
            _tc3_run(hbt)
        except Exception as exc:
            self.fail(
                f'TC3: FOK 策略抛出异常 → {type(exc).__name__}: {exc}'
            )

    # ------------------------------------------------------------------
    # TC4 — 回归: NoPartialFillExchange 行为不受影响（持续 6 天验证）
    # ------------------------------------------------------------------
    def test_tc4_no_partial_fill_exchange_regression(self):
        """
        回归测试：NoPartialFillExchange 的全量成交流程不受本次改动影响。
        本次修改只涉及 partialfillexchange.rs 和 local.rs，NoPartialFill 路径不变。
        持续 6 天下单，验证：回测正常完成，且 state.position 与手动累积量一致。
        """
        self._require_data()

        asset = _make_asset(partial_fill=False, data_files=self.data_files)
        hbt = HashMapMarketDepthBacktest([asset])
        results = np.zeros(3, dtype=np.float64)

        try:
            _tc4_run(hbt, results)
        except Exception as exc:
            self.fail(
                f'TC4: NoPartialFillExchange 回归测试抛出异常 → {type(exc).__name__}: {exc}'
            )

        state_pos  = results[0]
        manual_pos = results[1]
        fill_steps = int(results[2])

        if fill_steps == 0:
            self.skipTest('TC4: 6 天内未检测到任何成交，跳过仓位一致性校验。')

        self.assertAlmostEqual(
            state_pos, manual_pos, places=6,
            msg=(
                f'TC4: NoPartialFillExchange state.position ({state_pos:.6f}) '
                f'与手动累积量 ({manual_pos:.6f}) 不一致（{fill_steps} 步成交）。'
            ),
        )


    # ------------------------------------------------------------------
    # TC5 — FOK exec_qty 合并修复: 多档扫单 apply_fill 须使用累积总量
    # ------------------------------------------------------------------
    def test_tc5_fok_exec_qty_consolidated(self):
        """
        FOK exec_qty 合并修复验证：
        FOK 扫单覆盖多个价格档位时，local 收到的 exec_qty 须为全部档位的
        累积总量，而非最后一档的单档成交量。
        修复前：state.position < manual_position（差值 = 前 N-1 档成交量之和）。
        修复后：state.position == manual_position。

        注：当 best_ask 单档深度 >= ORDER_QTY 时，FOK 单档即可成交，
        exec_qty 本就等于总量，bug 不会显现。测试依赖数据中存在跨档成交场景。
        fill_steps 为 0 时跳过（所有 FOK 均因深度不足而 Expired）。
        """
        self._require_data()

        asset = _make_asset(partial_fill=True, data_files=self.data_files)
        hbt = HashMapMarketDepthBacktest([asset])
        results = np.zeros(3, dtype=np.float64)
        _tc5_run(hbt, results)

        state_pos  = results[0]
        manual_pos = results[1]
        fill_steps = int(results[2])

        if fill_steps == 0:
            self.skipTest(
                'TC5: 2 天内无 FOK 成交（ask 深度始终不足 ORDER_QTY），跳过验证。'
            )

        self.assertGreater(state_pos, 0.0,
            f'TC5: state.position 应为正值，实际={state_pos}')
        self.assertAlmostEqual(
            state_pos, manual_pos, places=6,
            msg=(
                f'TC5: state.position ({state_pos:.2f}) 与手动累积量 ({manual_pos:.2f}) '
                f'不一致（{fill_steps} 笔 FOK 成交）。'
                'FOK exec_qty 合并修复可能未生效。'
            ),
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
