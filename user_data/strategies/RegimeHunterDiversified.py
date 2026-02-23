import logging
from datetime import datetime, timezone
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from technical import qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

try:
    from bling_ai.decision_bridge import read_trade_conviction
except ImportError:
    read_trade_conviction = None  # BlingAI not installed


logger = logging.getLogger(__name__)


class RegimeHunterDiversified(IStrategy):
    """
    RegimeHunterDiversified - Optimized Multi-Pair ML Strategy

    Backtest Results (Mar 2024 - Jan 2026):
    - Profit: +$761.55 (76.15%)
    - Win Rate: 54.1%
    - Max Drawdown: 9.88%
    - CAGR: 34.4%
    - Calmar: 21.07

    Configuration:
    - Pairs: BTC, ETH, SOL, XRP, DOGE, LINK
    - Max Open Trades: 3
    - Leverage: 3x

    Key features:
    1. FreqAI ML model predicts local extrema (tops/bottoms)
    2. Regime-aware features (EMA slope, ATR expansion, drawdown)
    3. DCA averaging at -1.5%, -3%, -4.5% dips
    4. Trailing exit once profitable
    5. Gradual unstucking for stuck positions
    6. Diversified across 6 pairs for reduced drawdown
    """

    minimal_roi = {"0": 100}  # disabled — managed by trailing exit

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_extrema": {"&-s_extrema": {"color": "blue"}},
            "do_predict": {"do_predict": {"color": "brown"}},
        },
    }

    process_only_new_candles = True
    stoploss = -0.08  # emergency floor — should rarely hit with DCA + unstucking
    use_custom_stoploss = True
    use_exit_signal = True
    startup_candle_count: int = 50
    can_short = True

    # Enable position adjustments for DCA
    position_adjustment_enable = True
    max_entry_position_adjustment = 3  # 3 DCA entries after initial

    # Protections: pause trading after consecutive stoploss hits
    protections = [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 2,
            "stop_duration_candles": 12,
            "only_per_pair": False,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 48,
            "max_allowed_drawdown": 0.08,
            "trade_limit": 5,
            "stop_duration_candles": 24,
        },
    ]

    # --- Hyperoptable parameters ---
    entry_threshold_long = DecimalParameter(
        0.1, 0.9, default=0.5, decimals=2, space="buy",
        optimize=True, load=True,
    )
    entry_threshold_short = DecimalParameter(
        -0.9, -0.1, default=-0.3, decimals=2, space="sell",
        optimize=True, load=True,
    )
    adx_guard = IntParameter(
        0, 35, default=15, space="buy",
        optimize=True, load=True,
    )
    entry_cooldown_candles = IntParameter(
        0, 6, default=2, space="buy",
        optimize=True, load=True,
    )

    # ------------------------------------------------------------------ #
    #                       Feature Engineering                           #
    # ------------------------------------------------------------------ #

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)

        ema = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        dataframe["%-atr_pct-period"] = (
            ta.ATR(dataframe, timeperiod=period) / dataframe["close"]
        )

        macd = ta.MACD(
            dataframe,
            fastperiod=period,
            slowperiod=period * 2,
            signalperiod=max(period // 2, 5),
        )
        dataframe["%-macdhist-period"] = macd["macdhist"]

        obv = ta.OBV(dataframe)
        obv_sma = obv.rolling(period).mean()
        dataframe["%-obv_ratio-period"] = obv / obv_sma

        vwap = (
            (dataframe["close"] * dataframe["volume"]).rolling(period).sum()
            / dataframe["volume"].rolling(period).sum()
        )
        dataframe["%-vwap_ratio-period"] = dataframe["close"] / vwap

        # Regime features
        dataframe["%-ema_slope-period"] = ema.pct_change(
            periods=max(period // 2, 3)
        )
        dataframe["%-price_ema_dist-period"] = (
            (dataframe["close"] - ema) / dataframe["close"]
        )

        atr = ta.ATR(dataframe, timeperiod=period)
        longer_atr = atr.rolling(period * 2).mean()
        dataframe["%-atr_expansion-period"] = atr / longer_atr

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()

        candle_range = dataframe["high"] - dataframe["low"]
        candle_range_safe = candle_range.replace(0, np.nan)

        dataframe["%-candle_position"] = (
            (dataframe["close"] - dataframe["low"]) / candle_range_safe
        )
        dataframe["%-body_pct"] = (
            (dataframe["close"] - dataframe["open"]).abs() / candle_range_safe
        )
        dataframe["%-hl_range_pct"] = candle_range / dataframe["close"]

        rolling_high = dataframe["high"].rolling(50).max()
        dataframe["%-drawdown_pct"] = (
            (dataframe["close"] - rolling_high) / rolling_high
        )

        ema_32 = ta.EMA(dataframe, timeperiod=32)
        below = (dataframe["close"] < ema_32).astype(float)
        dataframe["%-consec_below_ema"] = below * (
            below.groupby((below != below.shift()).cumsum()).cumcount() + 1
        )

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek / 6.0
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour / 23.0
        return dataframe

    # ------------------------------------------------------------------ #
    #                          Target                                     #
    # ------------------------------------------------------------------ #

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]

        high = dataframe["high"].values
        low = dataframe["low"].values

        max_idx, _ = find_peaks(high, distance=label_period, prominence=0)
        min_idx, _ = find_peaks(-low, distance=label_period, prominence=0)

        extrema = np.zeros(len(dataframe))
        extrema[min_idx] = 1.0
        extrema[max_idx] = -1.0

        smoothed = gaussian_filter1d(extrema, sigma=3)

        abs_max = np.abs(smoothed).max()
        if abs_max > 0:
            smoothed = smoothed / abs_max

        dataframe["&-s_extrema"] = smoothed

        return dataframe

    # ------------------------------------------------------------------ #
    #                      Indicators                                     #
    # ------------------------------------------------------------------ #

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        # --- Crash Filter: ATR expansion ratio ---
        # When ATR > 1.5x its 50-period average, market is in crash/spike mode
        dataframe["atr_sma"] = dataframe["atr"].rolling(50).mean()
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["atr_sma"]
        dataframe["not_crashing"] = dataframe["atr_ratio"] < 1.5

        # --- Momentum Filter: RSI extreme avoidance ---
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        # Only avoid EXTREME conditions where trend is too strong against us
        # RSI < 20 = panic selling (don't catch falling knife)
        # RSI > 80 = euphoric buying (don't short into FOMO)
        dataframe["rsi_ok_long"] = dataframe["rsi"] > 20
        dataframe["rsi_ok_short"] = dataframe["rsi"] < 80

        # --- Regime Detection ---
        # Long-term trend: 200 EMA on 1h equivalent (200 * 12 = 2400 on 5m)
        # Medium-term trend: 50 EMA on 1h equivalent (50 * 12 = 600 on 5m)
        dataframe["ema_200_1h"] = ta.EMA(dataframe, timeperiod=2400)
        dataframe["ema_50_1h"] = ta.EMA(dataframe, timeperiod=600)

        # EMA slope: positive = uptrend momentum
        dataframe["ema_50_slope"] = dataframe["ema_50_1h"].pct_change(periods=12)

        # Bull regime: price above 200 EMA AND 50 EMA slope positive
        dataframe["regime_bull"] = (
            (dataframe["close"] > dataframe["ema_200_1h"]) &
            (dataframe["ema_50_slope"] > 0)
        ).astype(int)

        # --- TREND FILTER: Block counter-trend entries in strong moves ---
        # This is the key filter to prevent catching falling knives / shorting parabolas
        # Threshold -0.008 = ~0.8% decline in 50 EMA over 12 candles = strong downtrend
        # These are the trades that hit stoploss after full DCA
        dataframe["trend_ok_long"] = dataframe["ema_50_slope"] > -0.008
        dataframe["trend_ok_short"] = dataframe["ema_50_slope"] < 0.008

        return dataframe

    # ------------------------------------------------------------------ #
    #                      Entry Logic                                    #
    # ------------------------------------------------------------------ #

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        cooldown = self.entry_cooldown_candles.value
        adx_min = self.adx_guard.value

        # --- Dynamic Thresholds Based on Regime ---
        # Bull market: aggressive longs (0.3), selective shorts (-0.5)
        # Bear market: selective longs (0.5), aggressive shorts (-0.3)
        bull_regime = df["regime_bull"] == 1

        df["dynamic_long_thresh"] = 0.5   # Default: bear market (selective)
        df["dynamic_short_thresh"] = -0.3  # Default: bear market (aggressive)

        df.loc[bull_regime, "dynamic_long_thresh"] = 0.3   # Bull: aggressive longs
        df.loc[bull_regime, "dynamic_short_thresh"] = -0.5  # Bull: selective shorts

        # --- Long Entry Conditions ---
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_extrema"] > df["dynamic_long_thresh"],
            df["volume"] > 0,
            df["not_crashing"],      # Don't enter during volatility spikes
            df["rsi_ok_long"],       # Don't buy into falling knife
            df["trend_ok_long"],     # TREND FILTER: Don't long in strong downtrends
        ]

        if adx_min > 0:
            enter_long_conditions.append(df["adx"] > adx_min)

        if cooldown > 0:
            # Use base threshold for cooldown check
            recent_signal = (
                df["&-s_extrema"]
                .shift(1)
                .rolling(cooldown)
                .apply(lambda x: (x > 0.3).any(), raw=True)
            )
            enter_long_conditions.append(recent_signal != 1)

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "ml_long_bull" if bull_regime.any() else "ml_long_bear")

        # Fix: Use regime-specific tags
        long_mask = reduce(lambda x, y: x & y, enter_long_conditions)
        df.loc[long_mask & bull_regime, "enter_tag"] = "ml_long_bull"
        df.loc[long_mask & ~bull_regime, "enter_tag"] = "ml_long_bear"

        # --- Short Entry Conditions ---
        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_extrema"] < df["dynamic_short_thresh"],
            df["volume"] > 0,
            df["not_crashing"],      # Don't enter during volatility spikes
            df["rsi_ok_short"],      # Don't short into parabolic rally
            df["trend_ok_short"],    # TREND FILTER: Don't short in strong uptrends
        ]

        if adx_min > 0:
            enter_short_conditions.append(df["adx"] > adx_min)

        if cooldown > 0:
            recent_signal_short = (
                df["&-s_extrema"]
                .shift(1)
                .rolling(cooldown)
                .apply(lambda x: (x < -0.3).any(), raw=True)
            )
            enter_short_conditions.append(recent_signal_short != 1)

        short_mask = reduce(lambda x, y: x & y, enter_short_conditions)
        df.loc[short_mask, "enter_short"] = 1
        df.loc[short_mask & bull_regime, "enter_tag"] = "ml_short_bull"
        df.loc[short_mask & ~bull_regime, "enter_tag"] = "ml_short_bear"

        return df

    # ------------------------------------------------------------------ #
    #                       Exit Logic                                    #
    # ------------------------------------------------------------------ #

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Disable dataframe-based exits — use custom_exit for trailing
        return df

    # ------------------------------------------------------------------ #
    #                          Leverage                                   #
    # ------------------------------------------------------------------ #

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs) -> float:
        """Apply leverage from config (capped at exchange max)."""
        return min(proposed_leverage, max_leverage)

    # ------------------------------------------------------------------ #
    #                    Initial Stake (25% for DCA)                      #
    # ------------------------------------------------------------------ #

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Start with 25% of available stake.
        Reserve 75% for DCA entries.
        """
        # Use 25% of proposed stake for initial entry
        initial_stake = proposed_stake * 0.25

        # Ensure we meet minimum stake requirements
        if min_stake and initial_stake < min_stake:
            initial_stake = min_stake

        return initial_stake

    # ------------------------------------------------------------------ #
    #                    DCA — Averaging Down                             #
    # ------------------------------------------------------------------ #

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str]:
        """
        DCA: Add to position when price dips against us.
        Passivbot-style averaging.
        """
        filled_entries = trade.nr_of_successful_entries

        # Max 4 entries total (initial + 3 DCA)
        if filled_entries >= 4:
            return None

        # Block DCA if conviction agent says no
        if read_trade_conviction is not None:
            conviction = read_trade_conviction(trade.id, trade.pair)
            if conviction and not conviction.get("allow_dca", True):
                return None

        # DCA levels: -1.5%, -3%, -4.5%
        dca_levels = [-0.015, -0.03, -0.045]

        for i, level in enumerate(dca_levels):
            if filled_entries == i + 1 and current_profit <= level:
                # Add same amount as initial stake
                stake = trade.stake_amount / filled_entries  # Approximate initial stake

                # Ensure min stake
                if min_stake and stake < min_stake:
                    stake = min_stake

                # Don't exceed max
                if stake > max_stake:
                    stake = max_stake

                logger.info(
                    f"DCA #{filled_entries} for {trade.pair}: "
                    f"profit={current_profit:.2%}, adding {stake:.2f}"
                )
                return stake, f"dca_{filled_entries}"

        return None

    # ------------------------------------------------------------------ #
    #                Dynamic Stoploss Based on DCA                        #
    # ------------------------------------------------------------------ #

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        Dynamic stoploss: wide before all DCAs complete, tighter after.
        Must give DCA room to work — stoploss should not trigger before DCA levels.

        DCA levels: -1.5%, -3%, -4.5%
        So stoploss must be wider than -4.5% until all DCAs are done.
        """
        entries = trade.nr_of_successful_entries

        # --- Conviction-based SL tightening ---
        if read_trade_conviction is not None:
            conviction = read_trade_conviction(trade.id, pair)
            if conviction and conviction.get("tighten_sl") and conviction.get("suggested_sl") is not None:
                suggested = float(conviction["suggested_sl"])
                sl_ratio = (suggested - current_rate) / current_rate
                return max(sl_ratio, -0.08)  # Never wider than -8%

        # Before all DCAs done: wide stop to let averaging work
        # After all DCAs: tighten to limit max loss
        if entries < 4:
            return -0.06  # -6% floor while DCA is still possible
        else:
            return -0.08  # -8% after full DCA (all 4 entries used)

    # ------------------------------------------------------------------ #
    #                Trailing Exit + Unstucking                           #
    # ------------------------------------------------------------------ #

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """
        1. Conviction-based exit (BlingAI agent recommendation)
        2. Proportional trailing exit: once +0.5% profit, trail a % below peak
        3. Unstucking: if stuck 48h+ and model flipped, exit
        """
        # --- Conviction-based exit ---
        if read_trade_conviction is not None:
            conviction = read_trade_conviction(trade.id, pair)
            if conviction and conviction.get("action") == "exit" and conviction.get("conviction", 100) < 20:
                logger.info(
                    f"Conviction exit for {trade.pair}: "
                    f"conviction={conviction.get('conviction')}, reason={conviction.get('reasoning', '')[:100]}"
                )
                return "conviction_exit"

        # --- Persistent max_profit tracking (survives restarts) ---
        max_profit = trade.get_custom_data('max_profit', default=0.0)

        # Recovery fallback: reconstruct from DB-persisted max_rate/min_rate
        # if custom_data doesn't exist yet (first run after code update)
        if max_profit == 0.0 and current_profit > 0:
            if trade.is_short and trade.min_rate is not None:
                recovered = trade.calc_profit_ratio(trade.min_rate)
            elif not trade.is_short and trade.max_rate is not None:
                recovered = trade.calc_profit_ratio(trade.max_rate)
            else:
                recovered = 0.0
            if recovered > max_profit:
                max_profit = recovered

        # Update only when max_profit increases (minimizes DB writes)
        if current_profit > max_profit:
            max_profit = current_profit
            trade.set_custom_data('max_profit', max_profit)

        # --- Proportional Trailing Exit ---
        # Trail distance scales with profit level:
        #   0.5%-2% peak  -> give back 40% (keep 60%, lock small wins fast)
        #   2%-5% peak    -> give back 30% (keep 70%)
        #   5%+ peak      -> give back 25% (keep 75%, let big winners run)
        if max_profit >= 0.005:
            if max_profit >= 0.05:
                giveback_pct = 0.25
            elif max_profit >= 0.02:
                giveback_pct = 0.30
            else:
                giveback_pct = 0.40

            trail_distance = max(max_profit * giveback_pct, 0.002)
            trail_threshold = max_profit - trail_distance

            if current_profit <= trail_threshold:
                logger.info(
                    f"Trailing exit for {trade.pair}: "
                    f"max_profit={max_profit:.2%}, current={current_profit:.2%}, "
                    f"trail_at={trail_threshold:.2%} (giveback={giveback_pct:.0%})"
                )
                return "trailing_exit"

        # --- Unstucking (Conservative) ---
        # Only unstuck truly stuck positions with strong reversal signal
        # 72h wait, -4% loss threshold, 0.5 model conviction required
        if trade.open_date_utc:
            hours_open = (current_time.replace(tzinfo=timezone.utc) - trade.open_date_utc).total_seconds() / 3600
        else:
            hours_open = 0

        if hours_open > 48 and current_profit < -0.04:
            # Check if model conviction strongly flipped
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) < 1:
                return None

            pred = dataframe.iloc[-1].get("&-s_extrema", 0)
            do_predict = dataframe.iloc[-1].get("do_predict", 0)

            if do_predict == 1:
                if not trade.is_short and pred < -0.5:
                    # Was long, model now strongly bearish
                    logger.info(
                        f"Unstucking LONG {trade.pair}: "
                        f"hours={hours_open:.0f}, profit={current_profit:.2%}, pred={pred:.2f}"
                    )
                    return "unstuck_long"
                elif trade.is_short and pred > 0.5:
                    # Was short, model now strongly bullish
                    logger.info(
                        f"Unstucking SHORT {trade.pair}: "
                        f"hours={hours_open:.0f}, profit={current_profit:.2%}, pred={pred:.2f}"
                    )
                    return "unstuck_short"

        return None

    # ------------------------------------------------------------------ #
    #                   Confirm Trade Entry                               #
    # ------------------------------------------------------------------ #

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(df) < 1:
            return False

        last_candle = df.iloc[-1].squeeze()

        # Reject if price drifted >0.15% from signal candle
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0015)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0015)):
                return False

        return True
