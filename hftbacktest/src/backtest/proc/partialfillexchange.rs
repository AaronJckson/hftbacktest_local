use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::{
    backtest::{
        BacktestError,
        assettype::AssetType,
        models::{FeeModel, LatencyModel, QueueModel},
        order::ExchToLocal,
        proc::Processor,
        state::State,
    },
    depth::{L2MarketDepth, MarketDepth},
    prelude::OrdType,
    types::{
        EXCH_ASK_DEPTH_CLEAR_EVENT,
        EXCH_ASK_DEPTH_EVENT,
        EXCH_ASK_DEPTH_SNAPSHOT_EVENT,
        EXCH_BID_DEPTH_CLEAR_EVENT,
        EXCH_BID_DEPTH_EVENT,
        EXCH_BID_DEPTH_SNAPSHOT_EVENT,
        EXCH_BUY_TRADE_EVENT,
        EXCH_DEPTH_CLEAR_EVENT,
        EXCH_EVENT,
        EXCH_SELL_TRADE_EVENT,
        Event,
        Order,
        OrderId,
        Side,
        Status,
        TimeInForce,
    },
};

/// The exchange model with partial fills.
///
/// * Support order types: [OrdType::Limit](crate::types::OrdType::Limit)
/// * Support time-in-force: [`TimeInForce::GTC`], [`TimeInForce::FOK`], [`TimeInForce::IOC`],
///   [`TimeInForce::GTX`]
///
/// **Conditions for Full Execution**
/// Buy order in the order book
///
/// - Your order price >= the best ask price
/// - Your order price > sell trade price
///
/// Sell order in the order book
///
/// - Your order price <= the best bid price
/// - Your order price < buy trade price
///
/// **Conditions for Partial Execution**
/// Buy order in the order book
///
/// - Filled by (remaining) sell trade quantity: your order is at the front of the queue && your
///   order price == sell trade price
///
/// Sell order in the order book
///
/// - Filled by (remaining) buy trade quantity: your order is at the front of the queue && your
///   order price == buy trade price
///
/// **Liquidity-Taking Order**
/// Liquidity-taking orders will be executed based on the quantity of the order book, even though
/// the best price and quantity do not change due to your execution. Be aware that this may cause
/// unrealistic fill simulations if you attempt to execute a large quantity.
///
/// **General Comment**
/// Simulating partial fills accurately can be challenging, as they may indicate potential market
/// impact. The rule of thumb is to ensure that your backtesting results align with your live
/// results.
/// (more comment will be added...)
///
pub struct PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth,
    FM: FeeModel,
{
    // key: order_id, value: Order
    orders: Rc<RefCell<HashMap<OrderId, Order>>>,
    // key: order's price tick, value: order_ids
    buy_orders: HashMap<i64, HashSet<OrderId>>,
    sell_orders: HashMap<i64, HashSet<OrderId>>,

    order_e2l: ExchToLocal<LM>,

    depth: MD,
    state: State<AT, FM>,
    queue_model: QM,

    filled_orders: Vec<OrderId>,
}

impl<AT, LM, QM, MD, FM> PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth,
    FM: FeeModel,
{
    /// Constructs an instance of `PartialFillExchange`.
    pub fn new(
        depth: MD,
        state: State<AT, FM>,
        queue_model: QM,
        order_e2l: ExchToLocal<LM>,
    ) -> Self {
        Self {
            orders: Default::default(),
            buy_orders: Default::default(),
            sell_orders: Default::default(),
            order_e2l,
            depth,
            state,
            queue_model,
            filled_orders: Default::default(),
        }
    }

    fn check_if_sell_filled(
        &mut self,
        order: &mut Order,
        price_tick: i64,
        qty: f64,
        timestamp: i64,
        remaining_qty: &mut f64,
    ) -> Result<(), BacktestError> {
        match order.price_tick.cmp(&price_tick) {
            Ordering::Greater => {}
            Ordering::Less => {
                // The trade price has crossed below this sell order's price, so it should fill.
                // PartialFill semantics: fill up to min(leaves_qty, remaining_qty). If
                // remaining_qty is exhausted the order stays in the book for future trades.
                let exec_qty = order.leaves_qty.min(*remaining_qty);
                if exec_qty <= self.depth.lot_size() {
                    return Ok(());
                }
                *remaining_qty -= exec_qty;
                // Use lot-size rounding (matching fill()'s own Status::Filled check) rather than
                // a raw float comparison, which would miss sub-lot FP residuals and leave a
                // Filled order stranded in the book as a zombie.
                if ((order.leaves_qty - exec_qty) / self.depth.lot_size()).round() <= 0f64 {
                    self.filled_orders.push(order.order_id);
                }
                return self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty);
            }
            Ordering::Equal => {
                // Same-price fill is gated by queue position, not by remaining_qty, because
                // the queue model already accounts for the trade qty at this exact price level.
                self.queue_model.trade(order, qty, &self.depth);
                let filled_qty = self.queue_model.is_filled(order, &self.depth);
                if filled_qty > 0.0 {
                    // Cap at leaves_qty in case is_filled returns a larger value.
                    // Use lot-size rounding to detect full completion: the original
                    // filled_qty > leaves_qty missed the exact-equal case and FP residuals,
                    // both of which would leave a Filled order stranded in the book.
                    let exec_qty = filled_qty.min(order.leaves_qty);
                    if ((order.leaves_qty - exec_qty) / self.depth.lot_size()).round() <= 0f64 {
                        self.filled_orders.push(order.order_id);
                    }
                    return self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty);
                }
            }
        }
        Ok(())
    }

    fn check_if_buy_filled(
        &mut self,
        order: &mut Order,
        price_tick: i64,
        qty: f64,
        timestamp: i64,
        remaining_qty: &mut f64,
    ) -> Result<(), BacktestError> {
        match order.price_tick.cmp(&price_tick) {
            Ordering::Greater => {
                // The trade price has crossed above this buy order's price, so it should fill.
                // PartialFill semantics: fill up to min(leaves_qty, remaining_qty). If
                // remaining_qty is exhausted the order stays in the book for future trades.
                let exec_qty = order.leaves_qty.min(*remaining_qty);
                if exec_qty <= self.depth.lot_size() {
                    return Ok(());
                }
                *remaining_qty -= exec_qty;
                // Use lot-size rounding (matching fill()'s own Status::Filled check) rather than
                // a raw float comparison, which would miss sub-lot FP residuals and leave a
                // Filled order stranded in the book as a zombie.
                if ((order.leaves_qty - exec_qty) / self.depth.lot_size()).round() <= 0f64 {
                    self.filled_orders.push(order.order_id);
                }
                return self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty);
            }
            Ordering::Less => {}
            Ordering::Equal => {
                // Same-price fill is gated by queue position, not by remaining_qty, because
                // the queue model already accounts for the trade qty at this exact price level.
                self.queue_model.trade(order, qty, &self.depth);
                let filled_qty = self.queue_model.is_filled(order, &self.depth);
                if filled_qty > 0.0 {
                    // Cap at leaves_qty in case is_filled returns a larger value.
                    // Use lot-size rounding to detect full completion: the original
                    // filled_qty > leaves_qty missed the exact-equal case and FP residuals,
                    // both of which would leave a Filled order stranded in the book.
                    let exec_qty = filled_qty.min(order.leaves_qty);
                    if ((order.leaves_qty - exec_qty) / self.depth.lot_size()).round() <= 0f64 {
                        self.filled_orders.push(order.order_id);
                    }
                    return self.fill::<true>(order, timestamp, true, order.price_tick, exec_qty);
                }
            }
        }
        Ok(())
    }

    fn fill<const MAKE_RESPONSE: bool>(
        &mut self,
        order: &mut Order,
        timestamp: i64,
        maker: bool,
        exec_price_tick: i64,
        exec_qty: f64,
    ) -> Result<(), BacktestError> {
        if order.status == Status::Expired
            || order.status == Status::Canceled
            || order.status == Status::Filled
        {
            return Err(BacktestError::InvalidOrderStatus);
        }

        order.maker = maker;
        if maker {
            order.exec_price_tick = order.price_tick;
        } else {
            order.exec_price_tick = exec_price_tick;
        }

        order.exec_qty = exec_qty;
        order.leaves_qty -= exec_qty;
        if (order.leaves_qty / self.depth.lot_size()).round() > 0f64 {
            order.status = Status::PartiallyFilled;
        } else {
            order.status = Status::Filled;
        }
        order.exch_timestamp = timestamp;

        self.state.apply_fill(order);

        if MAKE_RESPONSE {
            self.order_e2l.respond(order.clone());
        }
        Ok(())
    }

    fn remove_filled_orders(&mut self) {
        if !self.filled_orders.is_empty() {
            let mut orders = self.orders.borrow_mut();
            for order_id in self.filled_orders.drain(..) {
                let Some(order) = orders.remove(&order_id) else {
                    // Already removed (e.g. duplicate order_id in filled_orders); skip.
                    continue;
                };
                if order.side == Side::Buy {
                    if let Some(set) = self.buy_orders.get_mut(&order.price_tick) {
                        set.remove(&order_id);
                    }
                } else {
                    if let Some(set) = self.sell_orders.get_mut(&order.price_tick) {
                        set.remove(&order_id);
                    }
                }
            }
        }
    }

    fn on_bid_qty_chg(&mut self, price_tick: i64, prev_qty: f64, new_qty: f64) {
        let orders = self.orders.clone();
        if let Some(order_ids) = self.buy_orders.get(&price_tick) {
            for order_id in order_ids.iter() {
                let mut orders_borrowed = orders.borrow_mut();
                let order = orders_borrowed.get_mut(order_id).unwrap();
                self.queue_model
                    .depth(order, prev_qty, new_qty, &self.depth);
            }
        }
    }

    fn on_ask_qty_chg(&mut self, price_tick: i64, prev_qty: f64, new_qty: f64) {
        let orders = self.orders.clone();
        if let Some(order_ids) = self.sell_orders.get(&price_tick) {
            for order_id in order_ids.iter() {
                let mut orders_borrowed = orders.borrow_mut();
                let order = orders_borrowed.get_mut(order_id).unwrap();
                self.queue_model
                    .depth(order, prev_qty, new_qty, &self.depth);
            }
        }
    }

    fn on_best_bid_update(
        &mut self,
        _prev_best_tick: i64,
        _new_best_tick: i64,
        _timestamp: i64,
    ) -> Result<(), BacktestError> {
        // Depth-only event: the bid move may be caused entirely by cancellations with no actual
        // trade occurring. Triggering fills here would overestimate fill probability (liquidity
        // vacuum bug). Orders crossed by the bid update remain in the book and will only be
        // filled when a real trade event arrives.
        Ok(())
    }

    fn on_best_ask_update(
        &mut self,
        _prev_best_tick: i64,
        _new_best_tick: i64,
        _timestamp: i64,
    ) -> Result<(), BacktestError> {
        // Depth-only event: the ask move may be caused entirely by cancellations with no actual
        // trade occurring. Triggering fills here would overestimate fill probability (liquidity
        // vacuum bug). Orders crossed by the ask update remain in the book and will only be
        // filled when a real trade event arrives.
        Ok(())
    }

    fn ack_new(&mut self, order: &mut Order, timestamp: i64) -> Result<(), BacktestError> {
        if self.orders.borrow().contains_key(&order.order_id) {
            return Err(BacktestError::OrderIdExist);
        }

        if order.side == Side::Buy {
            match order.order_type {
                OrdType::Limit => {
                    // Checks if the buy order price is greater than or equal to the current best ask.
                    if order.price_tick >= self.depth.best_ask_tick() {
                        match order.time_in_force {
                            TimeInForce::GTX => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::FOK => {
                                // The order must be executed immediately in its entirety; otherwise, the
                                // entire order will be cancelled.
                                let mut execute = false;
                                let mut cum_qty = 0f64;
                                for t in self.depth.best_ask_tick()..=order.price_tick {
                                    cum_qty += self.depth.ask_qty_at_tick(t);
                                    if (cum_qty / self.depth.lot_size()).round()
                                        >= (order.qty / self.depth.lot_size()).round()
                                    {
                                        execute = true;
                                        break;
                                    }
                                }
                                if execute {
                                    for t in self.depth.best_ask_tick()..=order.price_tick {
                                        let qty = self.depth.ask_qty_at_tick(t);
                                        if qty > 0.0 {
                                            let exec_qty = qty.min(order.leaves_qty);
                                            self.fill::<false>(
                                                order, timestamp, false, t, exec_qty,
                                            )?;
                                            if order.status == Status::Filled {
                                                return Ok(());
                                            }
                                        }
                                    }
                                    // Floating-point rounding may leave a sub-lot residual that
                                    // fill() rounds to zero without setting Status::Filled.
                                    // Expire the order rather than panic.
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    return Ok(());
                                } else {
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    Ok(())
                                }
                            }
                            TimeInForce::IOC => {
                                // The order must be executed immediately.
                                for t in self.depth.best_ask_tick()..=order.price_tick {
                                    let qty = self.depth.ask_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<false>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::GTC => {
                                // Sweep [best_ask, order.price_tick] as taker, inclusive on both ends.
                                for t in self.depth.best_ask_tick()..=order.price_tick {
                                    let qty = self.depth.ask_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<false>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                // Remaining qty transitions from taker to maker.
                                // order.price_tick ask qty is exhausted; queue position starts at 0.
                                self.queue_model.new_order_at_front(order, &self.depth);
                                order.status = Status::New;
                                self.buy_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);
                                order.exch_timestamp = timestamp;
                                self.orders.borrow_mut().insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    } else {
                        match order.time_in_force {
                            TimeInForce::GTC | TimeInForce::GTX => {
                                // Initializes the order's queue position.
                                self.queue_model.new_order(order, &self.depth);
                                order.status = Status::New;
                                // The exchange accepts this order.
                                self.buy_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);

                                order.exch_timestamp = timestamp;
                                self.orders
                                    .borrow_mut()
                                    .insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::FOK | TimeInForce::IOC => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    }
                }
                OrdType::Market => {
                    // todo: set the proper upper bound.
                    for t in self.depth.best_ask_tick()..(self.depth.best_ask_tick() + 100) {
                        let qty = self.depth.ask_qty_at_tick(t);
                        if qty > 0.0 {
                            let exec_qty = qty.min(order.leaves_qty);
                            self.fill::<false>(order, timestamp, false, t, exec_qty)?;
                        }
                        if order.status == Status::Filled {
                            return Ok(());
                        }
                    }
                    order.status = Status::Expired;
                    order.exch_timestamp = timestamp;
                    Ok(())
                }
                OrdType::Unsupported => Err(BacktestError::InvalidOrderRequest),
            }
        } else {
            match order.order_type {
                OrdType::Limit => {
                    // Checks if the sell order price is less than or equal to the current best bid.
                    if order.price_tick <= self.depth.best_bid_tick() {
                        match order.time_in_force {
                            TimeInForce::GTX => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::FOK => {
                                // The order must be executed immediately in its entirety; otherwise, the
                                // entire order will be cancelled.
                                let mut execute = false;
                                let mut cum_qty = 0f64;
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    cum_qty += self.depth.bid_qty_at_tick(t);
                                    if (cum_qty / self.depth.lot_size()).round()
                                        >= (order.qty / self.depth.lot_size()).round()
                                    {
                                        execute = true;
                                        break;
                                    }
                                }
                                if execute {
                                    for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                        let qty = self.depth.bid_qty_at_tick(t);
                                        if qty > 0.0 {
                                            let exec_qty = qty.min(order.leaves_qty);
                                            self.fill::<false>(
                                                order, timestamp, false, t, exec_qty,
                                            )?;
                                            if order.status == Status::Filled {
                                                return Ok(());
                                            }
                                        }
                                    }
                                    // Floating-point rounding may leave a sub-lot residual that
                                    // fill() rounds to zero without setting Status::Filled.
                                    // Expire the order rather than panic.
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    return Ok(());
                                } else {
                                    order.status = Status::Expired;
                                    order.exch_timestamp = timestamp;
                                    Ok(())
                                }
                            }
                            TimeInForce::IOC => {
                                // The order must be executed immediately.
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    let qty = self.depth.bid_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<false>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::GTC => {
                                // Sweep [order.price_tick, best_bid] as taker, inclusive on both ends.
                                for t in (order.price_tick..=self.depth.best_bid_tick()).rev() {
                                    let qty = self.depth.bid_qty_at_tick(t);
                                    if qty > 0.0 {
                                        let exec_qty = qty.min(order.leaves_qty);
                                        self.fill::<false>(order, timestamp, false, t, exec_qty)?;
                                    }
                                    if order.status == Status::Filled {
                                        return Ok(());
                                    }
                                }
                                // Remaining qty transitions from taker to maker.
                                // order.price_tick bid qty is exhausted; queue position starts at 0.
                                self.queue_model.new_order_at_front(order, &self.depth);
                                order.status = Status::New;
                                self.sell_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);
                                order.exch_timestamp = timestamp;
                                self.orders.borrow_mut().insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::Unsupported => {
                                Err(BacktestError::InvalidOrderRequest)
                            }
                        }
                    } else {
                        match order.time_in_force {
                            TimeInForce::GTC | TimeInForce::GTX => {
                                // Initializes the order's queue position.
                                self.queue_model.new_order(order, &self.depth);
                                order.status = Status::New;
                                // The exchange accepts this order.
                                self.sell_orders
                                    .entry(order.price_tick)
                                    .or_default()
                                    .insert(order.order_id);

                                order.exch_timestamp = timestamp;
                                self.orders
                                    .borrow_mut()
                                    .insert(order.order_id, order.clone());
                                Ok(())
                            }
                            TimeInForce::FOK | TimeInForce::IOC => {
                                order.status = Status::Expired;
                                order.exch_timestamp = timestamp;
                                Ok(())
                            }
                            TimeInForce::Unsupported => Err(BacktestError::InvalidOrderRequest),
                        }
                    }
                }
                OrdType::Market => {
                    // todo: set the proper lower bound.
                    for t in ((self.depth.best_bid_tick() - 100)..=self.depth.best_bid_tick()).rev()
                    {
                        let qty = self.depth.bid_qty_at_tick(t);
                        if qty > 0.0 {
                            let exec_qty = qty.min(order.leaves_qty);
                            self.fill::<false>(order, timestamp, false, t, exec_qty)?;
                        }
                        if order.status == Status::Filled {
                            return Ok(());
                        }
                    }
                    order.status = Status::Expired;
                    order.exch_timestamp = timestamp;
                    Ok(())
                }
                OrdType::Unsupported => Err(BacktestError::InvalidOrderRequest),
            }
        }
    }

    fn ack_cancel(&mut self, order: &mut Order, timestamp: i64) -> Result<(), BacktestError> {
        let exch_order = {
            let mut order_borrowed = self.orders.borrow_mut();
            order_borrowed.remove(&order.order_id)
        };

        if exch_order.is_none() {
            order.req = Status::Rejected;
            order.exch_timestamp = timestamp;
            return Ok(());
        }

        let exch_order = exch_order.unwrap();
        let _ = std::mem::replace(order, exch_order);

        // Deletes the order.
        if order.side == Side::Buy {
            self.buy_orders
                .get_mut(&order.price_tick)
                .unwrap()
                .remove(&order.order_id);
        } else {
            self.sell_orders
                .get_mut(&order.price_tick)
                .unwrap()
                .remove(&order.order_id);
        }
        order.status = Status::Canceled;
        order.exch_timestamp = timestamp;
        Ok(())
    }

    fn ack_modify<const RESET_QUEUE_POS: bool>(
        &mut self,
        order: &mut Order,
        timestamp: i64,
    ) -> Result<(), BacktestError> {
        let (prev_order_price_tick, prev_leaves_qty) = {
            let order_borrowed = self.orders.borrow();
            let exch_order = order_borrowed.get(&order.order_id);

            // The order can be already deleted due to fill or expiration.
            if exch_order.is_none() {
                order.req = Status::Rejected;
                order.exch_timestamp = timestamp;
                return Ok(());
            }

            let exch_order = exch_order.unwrap();
            (exch_order.price_tick, exch_order.leaves_qty)
        };

        // The initialization of the order queue position may not occur when the modified quantity
        // is smaller than the previous quantity, depending on the exchanges. It may need to
        // implement exchange-specific specialization.
        if RESET_QUEUE_POS
            || prev_order_price_tick != order.price_tick
            || order.qty > prev_leaves_qty
        {
            let mut cancel_order = order.clone();
            self.ack_cancel(&mut cancel_order, timestamp)?;
            self.ack_new(order, timestamp)?;
            // todo: Status::Replaced or Status::New?
        } else {
            let mut order_borrowed = self.orders.borrow_mut();
            let exch_order = order_borrowed.get_mut(&order.order_id);
            let exch_order = exch_order.unwrap();

            exch_order.qty = order.qty;
            exch_order.leaves_qty = order.qty;
            exch_order.exch_timestamp = timestamp;
            // todo: Status::Replaced or Status::New?
            exch_order.status = Status::New;
            order.leaves_qty = order.qty;
            order.exch_timestamp = timestamp;
            order.status = Status::New;
        }
        Ok(())
    }
}

impl<AT, LM, QM, MD, FM> Processor for PartialFillExchange<AT, LM, QM, MD, FM>
where
    AT: AssetType,
    LM: LatencyModel,
    QM: QueueModel<MD>,
    MD: MarketDepth + L2MarketDepth,
    FM: FeeModel,
{
    fn event_seen_timestamp(&self, event: &Event) -> Option<i64> {
        event.is(EXCH_EVENT).then_some(event.exch_ts)
    }

    fn process(&mut self, event: &Event) -> Result<(), BacktestError> {
        if event.is(EXCH_BID_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::Buy, event.px);
        } else if event.is(EXCH_ASK_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::Sell, event.px);
        } else if event.is(EXCH_DEPTH_CLEAR_EVENT) {
            self.depth.clear_depth(Side::None, 0.0);
        } else if event.is(EXCH_BID_DEPTH_EVENT) || event.is(EXCH_BID_DEPTH_SNAPSHOT_EVENT) {
            let (price_tick, prev_best_bid_tick, best_bid_tick, prev_qty, new_qty, timestamp) =
                self.depth
                    .update_bid_depth(event.px, event.qty, event.exch_ts);
            self.on_bid_qty_chg(price_tick, prev_qty, new_qty);
            if best_bid_tick > prev_best_bid_tick {
                self.on_best_bid_update(prev_best_bid_tick, best_bid_tick, timestamp)?;
            }
        } else if event.is(EXCH_ASK_DEPTH_EVENT) || event.is(EXCH_ASK_DEPTH_SNAPSHOT_EVENT) {
            let (price_tick, prev_best_ask_tick, best_ask_tick, prev_qty, new_qty, timestamp) =
                self.depth
                    .update_ask_depth(event.px, event.qty, event.exch_ts);
            self.on_ask_qty_chg(price_tick, prev_qty, new_qty);
            if best_ask_tick < prev_best_ask_tick {
                self.on_best_ask_update(prev_best_ask_tick, best_ask_tick, timestamp)?;
            }
        } else if event.is(EXCH_BUY_TRADE_EVENT) {
            let price_tick = (event.px / self.depth.tick_size()).round() as i64;
            let qty = event.qty;
            // Total trade qty available to allocate across all filled orders. Once exhausted,
            // no further fills can be triggered by this single trade event.
            let mut remaining_qty = qty;
            {
                let orders = self.orders.clone();
                let mut orders_borrowed = orders.borrow_mut();
                // Use the actual lowest sell order price as the loop lower bound instead of
                // best_bid_tick+1. After a liquidity vacuum, sell orders may sit below the
                // current best_bid, and best_bid-based bounds would miss them entirely.
                let min_sell_price_tick = self
                    .sell_orders
                    .keys()
                    .min()
                    .copied()
                    .unwrap_or(price_tick);
                // Choose between iterating all orders vs. iterating by tick based on which is
                // cheaper. The tick-range span is price_tick - min_sell_price_tick + 1.
                if (orders_borrowed.len() as i64) < price_tick - min_sell_price_tick + 1 {
                    for (_, order) in orders_borrowed.iter_mut() {
                        if order.side == Side::Sell {
                            self.check_if_sell_filled(
                                order,
                                price_tick,
                                qty,
                                event.exch_ts,
                                &mut remaining_qty,
                            )?;
                            if remaining_qty <= 0.0 {
                                break;
                            }
                        }
                    }
                } else {
                    'outer: for t in min_sell_price_tick..=price_tick {
                        if let Some(order_ids) = self.sell_orders.get(&t) {
                            for order_id in order_ids.clone().iter() {
                                let order = orders_borrowed.get_mut(order_id).unwrap();
                                self.check_if_sell_filled(
                                    order,
                                    price_tick,
                                    qty,
                                    event.exch_ts,
                                    &mut remaining_qty,
                                )?;
                                if remaining_qty <= 0.0 {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
            self.remove_filled_orders();
        } else if event.is(EXCH_SELL_TRADE_EVENT) {
            let price_tick = (event.px / self.depth.tick_size()).round() as i64;
            let qty = event.qty;
            // Total trade qty available to allocate across all filled orders. Once exhausted,
            // no further fills can be triggered by this single trade event.
            let mut remaining_qty = qty;
            {
                let orders = self.orders.clone();
                let mut orders_borrowed = orders.borrow_mut();
                // Use the actual highest buy order price as the loop upper bound instead of
                // best_ask_tick-1. After a liquidity vacuum, buy orders may sit above the
                // current best_ask, and best_ask-based bounds would miss them entirely.
                let max_buy_price_tick = self
                    .buy_orders
                    .keys()
                    .max()
                    .copied()
                    .unwrap_or(price_tick);
                // Choose between iterating all orders vs. iterating by tick based on which is
                // cheaper. The tick-range span is max_buy_price_tick - price_tick + 1.
                if (orders_borrowed.len() as i64) < max_buy_price_tick - price_tick + 1 {
                    for (_, order) in orders_borrowed.iter_mut() {
                        if order.side == Side::Buy {
                            self.check_if_buy_filled(
                                order,
                                price_tick,
                                qty,
                                event.exch_ts,
                                &mut remaining_qty,
                            )?;
                            if remaining_qty <= 0.0 {
                                break;
                            }
                        }
                    }
                } else {
                    // Iterate from highest buy price downward so that the most price-aggressive
                    // orders are filled first, consistent with real exchange priority.
                    'outer: for t in (price_tick..=max_buy_price_tick).rev() {
                        if let Some(order_ids) = self.buy_orders.get(&t) {
                            for order_id in order_ids.clone().iter() {
                                let order = orders_borrowed.get_mut(order_id).unwrap();
                                self.check_if_buy_filled(
                                    order,
                                    price_tick,
                                    qty,
                                    event.exch_ts,
                                    &mut remaining_qty,
                                )?;
                                if remaining_qty <= 0.0 {
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
            self.remove_filled_orders();
        }

        Ok(())
    }

    fn process_recv_order(
        &mut self,
        timestamp: i64,
        _wait_resp_order_id: Option<OrderId>,
    ) -> Result<bool, BacktestError> {
        while let Some(mut order) = self.order_e2l.receive(timestamp) {
            // Processes a new order.
            if order.req == Status::New {
                order.req = Status::None;
                self.ack_new(&mut order, timestamp)?;
            }
            // Processes a cancel order.
            else if order.req == Status::Canceled {
                order.req = Status::None;
                self.ack_cancel(&mut order, timestamp)?;
            }
            // Processes a modify order.
            else if order.req == Status::Replaced {
                order.req = Status::None;
                self.ack_modify::<false>(&mut order, timestamp)?;
            } else {
                return Err(BacktestError::InvalidOrderRequest);
            }
            // Makes the response.
            self.order_e2l.respond(order);
        }
        Ok(false)
    }

    fn earliest_recv_order_timestamp(&self) -> i64 {
        self.order_e2l
            .earliest_recv_order_timestamp()
            .unwrap_or(i64::MAX)
    }

    fn earliest_send_order_timestamp(&self) -> i64 {
        self.order_e2l
            .earliest_send_order_timestamp()
            .unwrap_or(i64::MAX)
    }
}
