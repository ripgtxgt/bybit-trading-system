import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import talib
from functools import partial

from ..utils.logger import Logger
from ..utils.config import Config
from ..storage.data_storage import DataStorage

class DataProcessor:
    """
    数据处理管道，负责数据清洗、特征工程和技术分析
    """
    def __init__(self):
        """
        初始化数据处理器
        """
        self.logger = Logger.get_logger("data_processor")
        self.config = Config()
        self.storage = DataStorage()
        
        # 注册技术指标计算函数
        self.indicators = {
            # 趋势指标
            "sma": self._calculate_sma,
            "ema": self._calculate_ema,
            "macd": self._calculate_macd,
            "adx": self._calculate_adx,
            
            # 动量指标
            "rsi": self._calculate_rsi,
            "stoch": self._calculate_stoch,
            "cci": self._calculate_cci,
            "mfi": self._calculate_mfi,
            
            # 波动率指标
            "bbands": self._calculate_bbands,
            "atr": self._calculate_atr,
            
            # 成交量指标
            "obv": self._calculate_obv,
            "ad": self._calculate_ad,
            "adosc": self._calculate_adosc,
            
            # 自定义指标
            "price_channels": self._calculate_price_channels,
            "vwap": self._calculate_vwap,
            "zigzag": self._calculate_zigzag,
            "market_regime": self._calculate_market_regime
        }
    
    async def process_kline_data(self, df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """
        处理K线数据，包括数据清洗和特征工程
        
        Args:
            df: 原始K线数据DataFrame
            indicators: 需要计算的技术指标列表，默认为None（计算所有指标）
            
        Returns:
            处理后的DataFrame
        """
        if df.empty:
            self.logger.warning("输入数据为空，无法处理")
            return df
        
        try:
            # 数据清洗
            df_cleaned = await self.clean_data(df)
            
            # 特征工程
            df_features = await self.engineer_features(df_cleaned)
            
            # 计算技术指标
            if indicators is None:
                # 默认计算一组基本指标
                indicators = ["sma", "ema", "macd", "rsi", "bbands", "atr"]
            
            df_with_indicators = await self.calculate_indicators(df_features, indicators)
            
            self.logger.info(f"数据处理完成，原始数据: {len(df)} 行，处理后: {len(df_with_indicators)} 行")
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"处理K线数据时出错: {e}")
            return df
    
    async def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗原始数据，处理缺失值、异常值等
        
        Args:
            df: 原始K线数据DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        if df.empty:
            return df
        
        try:
            # 创建副本，避免修改原始数据
            df_cleaned = df.copy()
            
            # 确保必要的列存在
            required_columns = ["open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df_cleaned.columns:
                    self.logger.error(f"缺少必要的列: {col}")
                    return df
            
            # 确保时间列存在
            if "datetime" not in df_cleaned.columns and "timestamp" in df_cleaned.columns:
                df_cleaned["datetime"] = pd.to_datetime(df_cleaned["timestamp"], unit="ms")
            
            # 排序
            if "datetime" in df_cleaned.columns:
                df_cleaned = df_cleaned.sort_values("datetime")
            
            # 删除重复行
            df_cleaned = df_cleaned.drop_duplicates(subset=["datetime"], keep="first")
            
            # 检查并处理缺失值
            missing_values = df_cleaned[required_columns].isnull().sum()
            if missing_values.sum() > 0:
                self.logger.warning(f"检测到缺失值: {missing_values}")
                
                # 对于少量缺失值，使用前向填充
                if missing_values.sum() / len(df_cleaned) < 0.05:  # 少于5%的缺失
                    df_cleaned[required_columns] = df_cleaned[required_columns].fillna(method="ffill")
                    self.logger.info("已使用前向填充处理缺失值")
            
            # 检查并处理异常值
            for col in ["open", "high", "low", "close"]:
                # 使用3倍标准差检测异常值
                mean = df_cleaned[col].mean()
                std = df_cleaned[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
                if not outliers.empty:
                    self.logger.warning(f"检测到 {len(outliers)} 个异常值在列 {col}")
                    
                    # 对于少量异常值，使用中位数替换
                    if len(outliers) / len(df_cleaned) < 0.01:  # 少于1%的异常
                        median = df_cleaned[col].median()
                        df_cleaned.loc[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound), col] = median
                        self.logger.info(f"已使用中位数 {median} 替换列 {col} 中的异常值")
            
            # 检查价格逻辑
            invalid_rows = df_cleaned[(df_cleaned["high"] < df_cleaned["low"]) | 
                                     (df_cleaned["open"] > df_cleaned["high"]) | 
                                     (df_cleaned["open"] < df_cleaned["low"]) | 
                                     (df_cleaned["close"] > df_cleaned["high"]) | 
                                     (df_cleaned["close"] < df_cleaned["low"])]
            
            if not invalid_rows.empty:
                self.logger.warning(f"检测到 {len(invalid_rows)} 行价格逻辑错误")
                
                # 对于少量逻辑错误，修复它们
                if len(invalid_rows) / len(df_cleaned) < 0.01:  # 少于1%的错误
                    for idx in invalid_rows.index:
                        row = df_cleaned.loc[idx]
                        # 修复high和low
                        if row["high"] < row["low"]:
                            df_cleaned.loc[idx, "high"], df_cleaned.loc[idx, "low"] = row["low"], row["high"]
                        
                        # 确保open在high和low之间
                        df_cleaned.loc[idx, "open"] = min(max(row["open"], df_cleaned.loc[idx, "low"]), df_cleaned.loc[idx, "high"])
                        
                        # 确保close在high和low之间
                        df_cleaned.loc[idx, "close"] = min(max(row["close"], df_cleaned.loc[idx, "low"]), df_cleaned.loc[idx, "high"])
                    
                    self.logger.info("已修复价格逻辑错误")
                else:
                    # 如果错误太多，可能是数据源问题，记录警告但不修改
                    self.logger.warning("价格逻辑错误过多，可能是数据源问题")
            
            # 检查时间连续性
            if "datetime" in df_cleaned.columns:
                df_cleaned = df_cleaned.sort_values("datetime")
                time_diff = df_cleaned["datetime"].diff()
                
                # 对于1分钟K线，预期的时间差是1分钟
                expected_diff = pd.Timedelta(minutes=1)
                
                gaps = time_diff[time_diff > expected_diff]
                if not gaps.empty:
                    self.logger.warning(f"检测到 {len(gaps)} 个时间间隙")
                    
                    # 记录大的时间间隙
                    for i, (idx, diff) in enumerate(gaps.items()):
                        if i < 5:  # 只记录前5个间隙
                            prev_time = df_cleaned.loc[idx-1, "datetime"] if idx > 0 else "开始"
                            curr_time = df_cleaned.loc[idx, "datetime"]
                            self.logger.warning(f"时间间隙: {prev_time} 到 {curr_time}, 差值: {diff}")
                        elif i == 5:
                            self.logger.warning("更多时间间隙省略...")
            
            self.logger.info(f"数据清洗完成，原始数据: {len(df)} 行，清洗后: {len(df_cleaned)} 行")
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"清洗数据时出错: {e}")
            return df
    
    async def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程，从原始数据中提取和构建特征
        
        Args:
            df: 清洗后的K线数据DataFrame
            
        Returns:
            添加了特征的DataFrame
        """
        if df.empty:
            return df
        
        try:
            # 创建副本，避免修改原始数据
            df_features = df.copy()
            
            # 计算价格变化
            df_features["price_change"] = df_features["close"].diff()
            df_features["price_change_pct"] = df_features["close"].pct_change() * 100
            
            # 计算价格范围
            df_features["price_range"] = df_features["high"] - df_features["low"]
            df_features["price_range_pct"] = df_features["price_range"] / df_features["low"] * 100
            
            # 计算成交量变化
            df_features["volume_change"] = df_features["volume"].diff()
            df_features["volume_change_pct"] = df_features["volume"].pct_change() * 100
            
            # 计算典型价格
            df_features["typical_price"] = (df_features["high"] + df_features["low"] + df_features["close"]) / 3
            
            # 计算加权收盘价
            df_features["weighted_close"] = (df_features["high"] + df_features["low"] + df_features["close"] * 2) / 4
            
            # 计算价格动量
            df_features["momentum_1"] = df_features["close"].diff(1)
            df_features["momentum_5"] = df_features["close"].diff(5)
            df_features["momentum_10"] = df_features["close"].diff(10)
            
            # 计算收盘价相对于开盘价的变化
            df_features["close_to_open"] = (df_features["close"] - df_features["open"]) / df_features["open"] * 100
            
            # 计算收盘价相对于最高价和最低价的位置
            df_features["close_position"] = (df_features["close"] - df_features["low"]) / (df_features["high"] - df_features["low"])
            
            # 计算K线形态特征
            df_features["body_size"] = abs(df_features["close"] - df_features["open"])
            df_features["upper_shadow"] = df_features["high"] - df_features[["open", "close"]].max(axis=1)
            df_features["lower_shadow"] = df_features[["open", "close"]].min(axis=1) - df_features["low"]
            
            # 计算K线形态比例
            df_features["body_to_range"] = df_features["body_size"] / df_features["price_range"]
            df_features["upper_shadow_to_range"] = df_features["upper_shadow"] / df_features["price_range"]
            df_features["lower_shadow_to_range"] = df_features["lower_shadow"] / df_features["price_range"]
            
            # 添加时间特征
            if "datetime" in df_features.columns:
                df_features["hour"] = df_features["datetime"].dt.hour
                df_features["day_of_week"] = df_features["datetime"].dt.dayofweek
                df_features["day_of_month"] = df_features["datetime"].dt.day
                df_features["week_of_year"] = df_features["datetime"].dt.isocalendar().week
                df_features["month"] = df_features["datetime"].dt.month
                df_features["quarter"] = df_features["datetime"].dt.quarter
                df_features["year"] = df_features["datetime"].dt.year
                
                # 添加交易时段标记
                df_features["is_asia_session"] = ((df_features["hour"] >= 1) & (df_features["hour"] < 9)).astype(int)
                df_features["is_europe_session"] = ((df_features["hour"] >= 8) & (df_features["hour"] < 16)).astype(int)
                df_features["is_us_session"] = ((df_features["hour"] >= 13) | (df_features["hour"] < 1)).astype(int)
            
            # 填充NaN值
            df_features = df_features.fillna(0)
            
            self.logger.info(f"特征工程完成，添加了 {len(df_features.columns) - len(df.columns)} 个特征")
            return df_features
            
        except Exception as e:
            self.logger.error(f"特征工程时出错: {e}")
            return df
    
    async def calculate_indicators(self, df: pd.DataFrame, indicator_list: List[str]) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 带有特征的K线数据DataFrame
            indicator_list: 需要计算的技术指标列表
            
        Returns:
            添加了技术指标的DataFrame
        """
        if df.empty:
            return df
        
        try:
            # 创建副本，避免修改原始数据
            df_indicators = df.copy()
            
            # 计算请求的指标
            for indicator in indicator_list:
                if indicator in self.indicators:
                    df_indicators = self.indicators[indicator](df_indicators)
                else:
                    self.logger.warning(f"未知的指标: {indicator}")
            
            # 填充NaN值
            df_indicators = df_indicators.fillna(0)
            
            self.logger.info(f"技术指标计算完成，计算了 {len(indicator_list)} 个指标")
            return df_indicators
            
        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {e}")
            return df
    
    def _calculate_sma(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 100, 200]) -> pd.DataFrame:
        """
        计算简单移动平均线
        """
        for period in periods:
            df[f"sma_{period}"] = talib.SMA(df["close"].values, timeperiod=period)
        return df
    
    def _calculate_ema(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 100, 200]) -> pd.DataFrame:
        """
        计算指数移动平均线
        """
        for period in periods:
            df[f"ema_{period}"] = talib.EMA(df["close"].values, timeperiod=period)
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        计算MACD指标
        """
        macd, signal, hist = talib.MACD(
            df["close"].values, 
            fastperiod=fast_period, 
            slowperiod=slow_period, 
            signalperiod=signal_period
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, periods: List[int] = [6, 14, 24]) -> pd.DataFrame:
        """
        计算相对强弱指数
        """
        for period in periods:
            df[f"rsi_{period}"] = talib.RSI(df["close"].values, timeperiod=period)
        return df
    
    def _calculate_bbands(self, df: pd.DataFrame, period: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0) -> pd.DataFrame:
        """
        计算布林带
        """
        upper, middle, lower = talib.BBANDS(
            df["close"].values, 
            timeperiod=period, 
            nbdevup=nbdevup, 
            nbdevdn=nbdevdn
        )
        df["bbands_upper"] = upper
        df["bbands_middle"] = middle
        df["bbands_lower"] = lower
        
        # 计算价格相对于布林带的位置
        df["bbands_position"] = (df["close"] - lower) / (upper - lower)
        
        # 计算布林带宽度
        df["bbands_width"] = (upper - lower) / middle
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算平均真实范围
        """
        df["atr"] = talib.ATR(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            timeperiod=period
        )
        
        # 计算相对ATR
        df["atr_pct"] = df["atr"] / df["close"] * 100
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算平均方向指数
        """
        df["adx"] = talib.ADX(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            timeperiod=period
        )
        
        df["plus_di"] = talib.PLUS_DI(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            timeperiod=period
        )
        
        df["minus_di"] = talib.MINUS_DI(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            timeperiod=period
        )
        
        return df
    
    def _calculate_stoch(self, df: pd.DataFrame, fastk_period: int = 5, slowk_period: int = 3, slowd_period: int = 3) -> pd.DataFrame:
        """
        计算随机指标
        """
        slowk, slowd = talib.STOCH(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            fastk_period=fastk_period, 
            slowk_period=slowk_period, 
            slowk_matype=0, 
            slowd_period=slowd_period, 
            slowd_matype=0
        )
        
        df["stoch_k"] = slowk
        df["stoch_d"] = slowd
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算商品通道指数
        """
        df["cci"] = talib.CCI(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            timeperiod=period
        )
        
        return df
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算资金流量指标
        """
        df["mfi"] = talib.MFI(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            df["volume"].values, 
            timeperiod=period
        )
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算能量潮指标
        """
        df["obv"] = talib.OBV(df["close"].values, df["volume"].values)
        
        return df
    
    def _calculate_ad(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算累积/派发线
        """
        df["ad"] = talib.AD(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            df["volume"].values
        )
        
        return df
    
    def _calculate_adosc(self, df: pd.DataFrame, fastperiod: int = 3, slowperiod: int = 10) -> pd.DataFrame:
        """
        计算震荡指标
        """
        df["adosc"] = talib.ADOSC(
            df["high"].values, 
            df["low"].values, 
            df["close"].values, 
            df["volume"].values, 
            fastperiod=fastperiod, 
            slowperiod=slowperiod
        )
        
        return df
    
    def _calculate_price_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算价格通道
        """
        df["price_channel_high"] = df["high"].rolling(window=period).max()
        df["price_channel_low"] = df["low"].rolling(window=period).min()
        df["price_channel_mid"] = (df["price_channel_high"] + df["price_channel_low"]) / 2
        
        # 计算价格相对于通道的位置
        df["price_channel_position"] = (df["close"] - df["price_channel_low"]) / (df["price_channel_high"] - df["price_channel_low"])
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量加权平均价格
        """
        # 计算典型价格
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        
        # 计算典型价格 * 成交量
        price_volume = typical_price * df["volume"]
        
        # 计算累计值
        cumulative_price_volume = price_volume.cumsum()
        cumulative_volume = df["volume"].cumsum()
        
        # 计算VWAP
        df["vwap"] = cumulative_price_volume / cumulative_volume
        
        return df
    
    def _calculate_zigzag(self, df: pd.DataFrame, deviation: float = 5.0) -> pd.DataFrame:
        """
        计算ZigZag指标
        """
        # 初始化ZigZag列
        df["zigzag"] = np.nan
        
        # 计算价格变化百分比
        price_change_pct = abs(df["close"].pct_change() * 100)
        
        # 找到变化超过阈值的点
        turning_points = price_change_pct > deviation
        
        # 标记转折点
        if turning_points.any():
            turning_indices = turning_points[turning_points].index
            df.loc[turning_indices, "zigzag"] = df.loc[turning_indices, "close"]
        
        return df
    
    def _calculate_market_regime(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        计算市场状态（趋势/震荡）
        """
        # 计算ADX
        df = self._calculate_adx(df)
        
        # 计算布林带宽度
        df = self._calculate_bbands(df, period=period)
        
        # 定义市场状态
        # ADX > 25 表示趋势市场，ADX < 20 表示震荡市场
        df["is_trend_market"] = (df["adx"] > 25).astype(int)
        df["is_range_market"] = (df["adx"] < 20).astype(int)
        
        # 布林带宽度可以帮助识别波动性
        # 宽度增加表示波动性增加，宽度减少表示波动性减少
        df["bbands_width_change"] = df["bbands_width"].pct_change()
        df["volatility_increasing"] = (df["bbands_width_change"] > 0).astype(int)
        df["volatility_decreasing"] = (df["bbands_width_change"] < 0).astype(int)
        
        return df
    
    async def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测K线形态
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            添加了形态检测结果的DataFrame
        """
        if df.empty:
            return df
        
        try:
            # 创建副本，避免修改原始数据
            df_patterns = df.copy()
            
            # 使用TA-Lib检测蜡烛图形态
            pattern_functions = {
                "doji": talib.CDLDOJI,
                "hammer": talib.CDLHAMMER,
                "hanging_man": talib.CDLHANGINGMAN,
                "engulfing": talib.CDLENGULFING,
                "morning_star": talib.CDLMORNINGSTAR,
                "evening_star": talib.CDLEVENINGSTAR,
                "shooting_star": talib.CDLSHOOTINGSTAR,
                "three_white_soldiers": talib.CDL3WHITESOLDIERS,
                "three_black_crows": talib.CDL3BLACKCROWS,
                "piercing": talib.CDLPIERCING,
                "dark_cloud_cover": talib.CDLDARKCLOUDCOVER,
                "harami": talib.CDLHARAMI,
                "harami_cross": talib.CDLHARAMICROSS
            }
            
            for pattern_name, pattern_func in pattern_functions.items():
                df_patterns[f"pattern_{pattern_name}"] = pattern_func(
                    df_patterns["open"].values,
                    df_patterns["high"].values,
                    df_patterns["low"].values,
                    df_patterns["close"].values
                )
            
            # 添加一个汇总列，表示是否检测到任何形态
            pattern_columns = [col for col in df_patterns.columns if col.startswith("pattern_")]
            df_patterns["has_pattern"] = (df_patterns[pattern_columns] != 0).any(axis=1).astype(int)
            
            # 添加一个列，表示检测到的形态数量
            df_patterns["pattern_count"] = (df_patterns[pattern_columns] != 0).sum(axis=1)
            
            self.logger.info(f"形态检测完成，检测了 {len(pattern_functions)} 种形态")
            return df_patterns
            
        except Exception as e:
            self.logger.error(f"检测K线形态时出错: {e}")
            return df
    
    async def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        分析市场结构，提供市场状态的摘要
        
        Args:
            df: 处理后的K线数据DataFrame
            
        Returns:
            市场结构分析结果字典
        """
        if df.empty:
            return {"error": "输入数据为空"}
        
        try:
            # 确保数据已经包含必要的指标
            required_indicators = ["adx", "bbands_width", "rsi_14"]
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            
            if missing_indicators:
                # 计算缺失的指标
                for indicator in missing_indicators:
                    if indicator == "adx":
                        df = self._calculate_adx(df)
                    elif indicator == "bbands_width":
                        df = self._calculate_bbands(df)
                    elif indicator == "rsi_14":
                        df = self._calculate_rsi(df)
            
            # 获取最新的市场状态
            latest = df.iloc[-1]
            
            # 分析趋势强度
            adx = latest.get("adx", 0)
            trend_strength = "强" if adx > 25 else "弱" if adx > 20 else "无"
            
            # 分析市场类型
            if adx > 25:
                market_type = "趋势市场"
            elif adx < 20:
                market_type = "震荡市场"
            else:
                market_type = "过渡市场"
            
            # 分析趋势方向
            if "plus_di" in df.columns and "minus_di" in df.columns:
                plus_di = latest.get("plus_di", 0)
                minus_di = latest.get("minus_di", 0)
                trend_direction = "上升" if plus_di > minus_di else "下降"
            else:
                # 使用简单的移动平均线判断趋势方向
                if "sma_20" in df.columns and "sma_50" in df.columns:
                    sma_20 = latest.get("sma_20", 0)
                    sma_50 = latest.get("sma_50", 0)
                    trend_direction = "上升" if sma_20 > sma_50 else "下降"
                else:
                    # 使用最近的价格变化判断趋势方向
                    recent_changes = df["close"].pct_change(5).iloc[-5:]
                    trend_direction = "上升" if recent_changes.mean() > 0 else "下降"
            
            # 分析波动性
            bbands_width = latest.get("bbands_width", 0)
            volatility = "高" if bbands_width > df["bbands_width"].mean() * 1.2 else "低" if bbands_width < df["bbands_width"].mean() * 0.8 else "中"
            
            # 分析超买/超卖状态
            rsi_14 = latest.get("rsi_14", 50)
            overbought_oversold = "超买" if rsi_14 > 70 else "超卖" if rsi_14 < 30 else "中性"
            
            # 分析成交量
            volume_change = latest.get("volume_change_pct", 0)
            volume_trend = "增加" if volume_change > 20 else "减少" if volume_change < -20 else "稳定"
            
            # 汇总分析结果
            analysis = {
                "market_type": market_type,
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "volatility": volatility,
                "overbought_oversold": overbought_oversold,
                "volume_trend": volume_trend,
                "adx": adx,
                "rsi": rsi_14,
                "bbands_width": bbands_width,
                "timestamp": latest.get("datetime", datetime.now())
            }
            
            self.logger.info(f"市场结构分析完成: {analysis}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"分析市场结构时出错: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """
        关闭数据处理器
        """
        await self.storage.close()
        self.logger.info("数据处理器已关闭")


async def main():
    """
    主函数
    """
    # 创建数据处理器
    processor = DataProcessor()
    
    # 创建测试数据
    data = {
        "datetime": pd.date_range(start="2024-01-01", periods=100, freq="1min"),
        "open": np.random.normal(50000, 1000, 100),
        "high": np.random.normal(50500, 1000, 100),
        "low": np.random.normal(49500, 1000, 100),
        "close": np.random.normal(50000, 1000, 100),
        "volume": np.random.normal(10, 5, 100)
    }
    
    # 确保价格逻辑正确
    df = pd.DataFrame(data)
    for i in range(len(df)):
        df.loc[i, "high"] = max(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "high"])
        df.loc[i, "low"] = min(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "low"])
    
    # 处理数据
    processed_df = await processor.process_kline_data(df, indicators=["sma", "ema", "macd", "rsi", "bbands", "atr"])
    
    # 检测形态
    patterns_df = await processor.detect_patterns(processed_df)
    
    # 分析市场结构
    analysis = await processor.analyze_market_structure(patterns_df)
    
    # 打印结果
    print(f"原始数据: {len(df)} 行")
    print(f"处理后数据: {len(processed_df)} 行, {len(processed_df.columns)} 列")
    print(f"市场分析: {analysis}")
    
    # 关闭处理器
    await processor.close()


if __name__ == "__main__":
    asyncio.run(main())
