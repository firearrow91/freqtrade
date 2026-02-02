import logging
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from technical import qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


logger = logging.getLogger(__name__)


class FreqaiMLFirstStrategy(IStrategy):
    """
    ML-first FreqAI strategy: the model prediction is the primary entry/exit
    signal. Uses ATR-based hard stoploss (no trailing) and relies on ML exit
    signal to manage trades.
    """

    minimal_roi = {"0": 100}  # disabled — exits via ML signal + stoploss only

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_extrema": {"&-s_extrema": {"color": "blue"}},
            "do_predict": {"do_predict": {"color": "brown"}},
        },
    }

    process_only_new_candles = True
    stoploss = -0.04  # hard floor — ML custom_exit tries to cut before this
    use_custom_stoploss = False
    use_exit_signal = True
    startup_candle_count: int = 50
    can_short = True

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
            "max_allowed_drawdown": 0.05,
            "trade_limit": 5,
            "stop_duration_candles": 24,
        },
    ]

    # --- Hyperoptable parameters ---
    # Extrema scale: +1 = local bottom (go long), -1 = local top (go short)
    entry_threshold_long = DecimalParameter(
        0.1, 0.7, default=0.4, decimals=2, space="buy",
        optimize=True, load=True,
    )
    entry_threshold_short = DecimalParameter(
        -0.7, -0.1, default=-0.5, decimals=2, space="sell",
        optimize=True, load=True,
    )
    adx_guard = IntParameter(
        0, 35, default=0, space="buy",
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
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

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

        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr_pct-period"] = (
            dataframe["%-atr-period"] / dataframe["close"]
        )

        # MACD histogram only (removed raw macd and signal — redundant)
        macd = ta.MACD(
            dataframe,
            fastperiod=period,
            slowperiod=period * 2,
            signalperiod=max(period // 2, 5),
        )
        dataframe["%-macdhist-period"] = macd["macdhist"]

        # Volume flow
        obv = ta.OBV(dataframe)
        obv_sma = obv.rolling(period).mean()
        dataframe["%-obv_ratio-period"] = obv / obv_sma

        # VWAP ratio
        vwap = (
            (dataframe["close"] * dataframe["volume"]).rolling(period).sum()
            / dataframe["volume"].rolling(period).sum()
        )
        dataframe["%-vwap_ratio-period"] = dataframe["close"] / vwap

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()

        # Candle structure
        candle_range = dataframe["high"] - dataframe["low"]
        candle_range_safe = candle_range.replace(0, np.nan)

        dataframe["%-candle_position"] = (
            (dataframe["close"] - dataframe["low"]) / candle_range_safe
        )
        dataframe["%-body_pct"] = (
            (dataframe["close"] - dataframe["open"]).abs() / candle_range_safe
        )
        dataframe["%-hl_range_pct"] = candle_range / dataframe["close"]

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

        # Detect significant peaks (maxima on highs, minima on lows)
        max_idx, max_props = find_peaks(high, distance=label_period, prominence=0)
        min_idx, min_props = find_peaks(-low, distance=label_period, prominence=0)

        # Create sparse extrema signal: +1 at bottoms, -1 at tops
        extrema = np.zeros(len(dataframe))
        extrema[min_idx] = 1.0   # local bottom = buy opportunity
        extrema[max_idx] = -1.0  # local top = sell opportunity

        # Gaussian smoothing for smooth target (replaces linear interpolation)
        smoothed = gaussian_filter1d(extrema, sigma=3)

        # Normalize to [-1, 1] range
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

        # ADX for optional trend guard
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ATR for dynamic stoploss and position sizing
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        return dataframe

    # ------------------------------------------------------------------ #
    #                      Entry Logic                                    #
    # ------------------------------------------------------------------ #

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        threshold_long = self.entry_threshold_long.value
        threshold_short = self.entry_threshold_short.value
        cooldown = self.entry_cooldown_candles.value
        adx_min = self.adx_guard.value

        # --- Long --- (extrema > threshold means "near a bottom")
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_extrema"] > threshold_long,
            df["volume"] > 0,
        ]

        if adx_min > 0:
            enter_long_conditions.append(df["adx"] > adx_min)

        if cooldown > 0:
            recent_signal = (
                df["&-s_extrema"]
                .shift(1)
                .rolling(cooldown)
                .apply(lambda x: (x > threshold_long).any(), raw=True)
            )
            enter_long_conditions.append(recent_signal != 1)

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"],
            ] = (1, "ml_long")

        # --- Short --- (extrema < threshold means "near a top")
        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_extrema"] < threshold_short,
            df["volume"] > 0,
        ]

        if adx_min > 0:
            enter_short_conditions.append(df["adx"] > adx_min)

        if cooldown > 0:
            recent_signal_short = (
                df["&-s_extrema"]
                .shift(1)
                .rolling(cooldown)
                .apply(lambda x: (x < threshold_short).any(), raw=True)
            )
            enter_short_conditions.append(recent_signal_short != 1)

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"],
            ] = (1, "ml_short")

        return df

    # ------------------------------------------------------------------ #
    #                       Exit Logic                                    #
    # ------------------------------------------------------------------ #

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when model predicts we're near a top (require strong signal)
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_extrema"] < -0.2,
        ]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        # Exit short when model predicts we're near a bottom (require strong signal)
        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_extrema"] > 0.2,
        ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    # ------------------------------------------------------------------ #
    #                    ML-Driven Early Exit                             #
    # ------------------------------------------------------------------ #

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None

        last_candle = dataframe.iloc[-1]
        prediction = last_candle.get("&-s_extrema", 0)
        do_predict = last_candle.get("do_predict", 0)

        if do_predict != 1:
            return None

        if not trade.is_short:
            # LONG: entered near bottom (extrema > +0.5). Cut if model now
            # says we're near a top AND trade is losing.
            if current_profit < -0.003 and prediction < -0.2:
                return "ml_cut_long"
        else:
            # SHORT: entered near top (extrema < -0.5). Cut if model now
            # says we're near a bottom AND trade is losing.
            if current_profit < -0.003 and prediction > 0.2:
                return "ml_cut_short"

        return None

    # ------------------------------------------------------------------ #
    #              Volatility-Based Position Sizing                       #
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
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return proposed_stake

        atr_pct = dataframe.iloc[-1].get("atr_pct", 0)

        if atr_pct > 0.02:
            return proposed_stake * 0.5
        elif atr_pct > 0.015:
            return proposed_stake * 0.75

        return proposed_stake

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
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        # Reject if price drifted >0.15% from signal candle
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0015)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0015)):
                return False

        return True
