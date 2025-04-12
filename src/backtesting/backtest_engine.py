import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Type
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path

from ..utils.logger import Logger
from ..utils.config import Config
from ..storage.data_storage import DataStorage
from ..data_processing.data_processor import DataProcessor

class Strategy(ABC):
    """
    交易策略抽象基类，所有策略都应该继承这个类
    """
    def __init__(self, name: str, params: Dict = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            params: 策略参数字典
        """
        self.name = name
        self.params = params or {}
        self.logger = Logger.get_logger(f"strategy_{name}")
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含价格和指标数据的DataFrame
            
        Returns:
            添加了交易信号的DataFrame
        """
        pass
    
    def set_params(self, params: Dict):
        """
        设置策略参数
        
        Args:
            params: 策略参数字典
        """
        self.params = params
        self.logger.info(f"策略参数已更新: {params}")


class BacktestEngine:
    """
    回测引擎，用于在历史数据上测试交易策略
    """
    def __init__(self):
        """
        初始化回测引擎
        """
        self.logger = Logger.get_logger("backtest_engine")
        self.config = Config()
        self.storage = DataStorage()
        self.processor = DataProcessor()
        
        # 回测配置
        self.start_date = None
        self.end_date = None
        self.initial_capital = 10000.0  # 默认初始资金
        self.position = 0.0  # 当前持仓数量
        self.position_value = 0.0  # 当前持仓价值
        self.cash = self.initial_capital  # 当前现金
        self.fee_rate = 0.0006  # 默认手续费率 0.06%
        self.slippage = 0.0001  # 默认滑点 0.01%
        
        # 回测结果
        self.trades = []  # 交易记录
        self.equity_curve = []  # 权益曲线
        self.performance_metrics = {}  # 性能指标
        
        # 创建结果目录
        self.results_dir = Path("data/backtest_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def load_data(self, symbol: str, interval: str, start_date: str, end_date: str, indicators: List[str] = None) -> pd.DataFrame:
        """
        加载并处理回测数据
        
        Args:
            symbol: 交易对符号
            interval: K线时间间隔
            start_date: 开始日期，格式为 "YYYY-MM-DD"
            end_date: 结束日期，格式为 "YYYY-MM-DD"
            indicators: 需要计算的技术指标列表
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 设置回测时间范围
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # 尝试从数据库加载数据
            df = await self.storage.query_kline_data(
                symbol=symbol,
                interval=interval,
                start_time=start_date,
                end_time=end_date
            )
            
            # 如果数据库中没有数据，则从API下载
            if df.empty:
                self.logger.info(f"数据库中未找到数据，将从API下载: {symbol}, {interval}, {start_date} - {end_date}")
                
                # 导入历史数据下载器
                from ..data_collection.historical_data_downloader import BybitHistoricalDataDownloader
                
                # 创建下载器
                downloader = BybitHistoricalDataDownloader(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_date,
                    end_time=end_date
                )
                
                # 下载数据
                df = await downloader.download_all()
                
                # 保存数据到数据库
                if not df.empty:
                    await self.storage.store_kline_data(df)
            
            # 如果仍然没有数据，则抛出异常
            if df.empty:
                raise ValueError(f"无法获取数据: {symbol}, {interval}, {start_date} - {end_date}")
            
            # 处理数据
            df = await self.processor.process_kline_data(df, indicators)
            
            self.logger.info(f"数据加载完成: {symbol}, {interval}, {start_date} - {end_date}, {len(df)} 行")
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise
    
    async def run_backtest(self, strategy: Strategy, df: pd.DataFrame, initial_capital: float = 10000.0, 
                          fee_rate: float = 0.0006, slippage: float = 0.0001) -> Dict:
        """
        运行回测
        
        Args:
            strategy: 交易策略
            df: 包含价格和指标数据的DataFrame
            initial_capital: 初始资金
            fee_rate: 手续费率
            slippage: 滑点
            
        Returns:
            回测结果字典
        """
        try:
            # 重置回测状态
            self.initial_capital = initial_capital
            self.cash = initial_capital
            self.position = 0.0
            self.position_value = 0.0
            self.fee_rate = fee_rate
            self.slippage = slippage
            self.trades = []
            self.equity_curve = []
            
            # 生成交易信号
            df_signals = strategy.generate_signals(df)
            
            # 确保信号列存在
            if "signal" not in df_signals.columns:
                self.logger.error("未找到信号列，请确保策略生成了'signal'列")
                return {"error": "未找到信号列"}
            
            # 执行回测
            df_backtest = await self._execute_backtest(df_signals)
            
            # 计算性能指标
            self.performance_metrics = self._calculate_performance_metrics(df_backtest)
            
            # 保存回测结果
            result_id = await self._save_backtest_results(strategy, df_backtest)
            
            # 返回回测结果
            result = {
                "strategy_name": strategy.name,
                "symbol": df_signals["symbol"].iloc[0] if "symbol" in df_signals.columns else "unknown",
                "start_time": df_signals["datetime"].min(),
                "end_time": df_signals["datetime"].max(),
                "initial_capital": self.initial_capital,
                "final_capital": self.cash + self.position_value,
                "total_return": self.performance_metrics["total_return"],
                "annual_return": self.performance_metrics["annual_return"],
                "sharpe_ratio": self.performance_metrics["sharpe_ratio"],
                "max_drawdown": self.performance_metrics["max_drawdown"],
                "win_rate": self.performance_metrics["win_rate"],
                "profit_factor": self.performance_metrics["profit_factor"],
                "trades_count": len(self.trades),
                "params": strategy.params,
                "result_id": result_id
            }
            
            self.logger.info(f"回测完成: {strategy.name}, 总收益率: {result['total_return']:.2f}%, 夏普比率: {result['sharpe_ratio']:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"运行回测失败: {e}")
            return {"error": str(e)}
    
    async def _execute_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行回测逻辑
        
        Args:
            df: 包含价格和信号数据的DataFrame
            
        Returns:
            添加了回测结果的DataFrame
        """
        # 创建回测结果DataFrame
        df_backtest = df.copy()
        
        # 添加回测结果列
        df_backtest["position"] = 0.0
        df_backtest["cash"] = self.cash
        df_backtest["position_value"] = 0.0
        df_backtest["equity"] = self.cash
        df_backtest["returns"] = 0.0
        df_backtest["trade_price"] = 0.0
        df_backtest["fee"] = 0.0
        
        # 遍历每个时间点
        for i in range(1, len(df_backtest)):
            # 获取当前行和前一行
            prev_row = df_backtest.iloc[i-1]
            curr_row = df_backtest.iloc[i]
            
            # 默认继承前一时间点的状态
            df_backtest.loc[df_backtest.index[i], "position"] = prev_row["position"]
            df_backtest.loc[df_backtest.index[i], "cash"] = prev_row["cash"]
            
            # 获取信号
            signal = curr_row["signal"]
            
            # 处理信号
            if signal != 0 and signal != prev_row["signal"]:
                # 计算交易价格（考虑滑点）
                if signal > 0:  # 买入信号
                    trade_price = curr_row["close"] * (1 + self.slippage)
                else:  # 卖出信号
                    trade_price = curr_row["close"] * (1 - self.slippage)
                
                df_backtest.loc[df_backtest.index[i], "trade_price"] = trade_price
                
                # 平仓现有持仓（如果有）
                if prev_row["position"] != 0:
                    # 计算平仓价值
                    close_value = prev_row["position"] * trade_price
                    
                    # 计算手续费
                    fee = close_value * self.fee_rate
                    df_backtest.loc[df_backtest.index[i], "fee"] += fee
                    
                    # 更新现金
                    df_backtest.loc[df_backtest.index[i], "cash"] = prev_row["cash"] + close_value - fee
                    
                    # 记录交易
                    trade = {
                        "type": "close",
                        "datetime": curr_row["datetime"],
                        "price": trade_price,
                        "size": prev_row["position"],
                        "value": close_value,
                        "fee": fee,
                        "pnl": close_value - prev_row["position_value"] - fee
                    }
                    self.trades.append(trade)
                
                # 开新仓
                if signal != 0:
                    # 计算可用资金（考虑杠杆）
                    available_cash = df_backtest.loc[df_backtest.index[i], "cash"]
                    
                    # 计算仓位大小（使用全部可用资金）
                    position_size = available_cash / trade_price
                    
                    # 计算开仓价值
                    open_value = position_size * trade_price
                    
                    # 计算手续费
                    fee = open_value * self.fee_rate
                    df_backtest.loc[df_backtest.index[i], "fee"] += fee
                    
                    # 更新持仓和现金
                    df_backtest.loc[df_backtest.index[i], "position"] = position_size * signal
                    df_backtest.loc[df_backtest.index[i], "cash"] = available_cash - open_value - fee
                    
                    # 记录交易
                    trade = {
                        "type": "open",
                        "datetime": curr_row["datetime"],
                        "price": trade_price,
                        "size": position_size * signal,
                        "value": open_value,
                        "fee": fee,
                        "pnl": -fee
                    }
                    self.trades.append(trade)
            
            # 更新持仓价值
            df_backtest.loc[df_backtest.index[i], "position_value"] = df_backtest.loc[df_backtest.index[i], "position"] * curr_row["close"]
            
            # 更新权益
            df_backtest.loc[df_backtest.index[i], "equity"] = df_backtest.loc[df_backtest.index[i], "cash"] + df_backtest.loc[df_backtest.index[i], "position_value"]
            
            # 计算收益率
            df_backtest.loc[df_backtest.index[i], "returns"] = df_backtest.loc[df_backtest.index[i], "equity"] / prev_row["equity"] - 1
            
            # 更新当前状态
            self.cash = df_backtest.loc[df_backtest.index[i], "cash"]
            self.position = df_backtest.loc[df_backtest.index[i], "position"]
            self.position_value = df_backtest.loc[df_backtest.index[i], "position_value"]
            
            # 添加到权益曲线
            self.equity_curve.append({
                "datetime": curr_row["datetime"],
                "equity": df_backtest.loc[df_backtest.index[i], "equity"],
                "returns": df_backtest.loc[df_backtest.index[i], "returns"]
            })
        
        return df_backtest
    
    def _calculate_performance_metrics(self, df_backtest: pd.DataFrame) -> Dict:
        """
        计算回测性能指标
        
        Args:
            df_backtest: 回测结果DataFrame
            
        Returns:
            性能指标字典
        """
        # 确保有足够的数据
        if len(df_backtest) < 2:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "trades_count": 0
            }
        
        # 计算总收益率
        initial_equity = df_backtest["equity"].iloc[0]
        final_equity = df_backtest["equity"].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        # 计算年化收益率
        days = (df_backtest["datetime"].iloc[-1] - df_backtest["datetime"].iloc[0]).days
        if days > 0:
            annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        else:
            annual_return = 0.0
        
        # 计算夏普比率
        returns = df_backtest["returns"].dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5)  # 假设252个交易日
        else:
            sharpe_ratio = 0.0
        
        # 计算最大回撤
        equity_curve = df_backtest["equity"]
        max_drawdown = 0.0
        peak = equity_curve.iloc[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        max_drawdown *= 100  # 转换为百分比
        
        # 计算胜率和盈亏比
        if self.trades:
            winning_trades = [t for t in self.trades if t["type"] == "close" and t["pnl"] > 0]
            losing_trades = [t for t in self.trades if t["type"] == "close" and t["pnl"] <= 0]
            
            win_rate = len(winning_trades) / len([t for t in self.trades if t["type"] == "close"]) * 100 if len([t for t in self.trades if t["type"] == "close"]) > 0 else 0.0
            
            total_profit = sum(t["pnl"] for t in winning_trades)
            total_loss = abs(sum(t["pnl"] for t in losing_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trades_count": len([t for t in self.trades if t["type"] == "close"])
        }
    
    async def _save_backtest_results(self, strategy: Strategy, df_backtest: pd.DataFrame) -> str:
        """
        保存回测结果
        
        Args:
            strategy: 交易策略
            df_backtest: 回测结果DataFrame
            
        Returns:
            结果ID
        """
        # 生成结果ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{strategy.name}_{timestamp}"
        
        # 创建结果目录
        result_dir = self.results_dir / result_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存回测数据
        df_backtest.to_csv(result_dir / "backtest_data.csv", index=False)
        
        # 保存交易记录
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(result_dir / "trades.csv", index=False)
        
        # 保存权益曲线
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.to_csv(result_dir / "equity_curve.csv", index=False)
        
        # 保存性能指标
        with open(result_dir / "performance_metrics.json", "w") as f:
            json.dump(self.performance_metrics, f, indent=4)
        
        # 保存策略参数
        with open(result_dir / "strategy_params.json", "w") as f:
            json.dump(strategy.params, f, indent=4)
        
        # 生成回测报告
        await self.generate_report(result_id)
        
        # 保存到数据库
        await self.storage.store_backtest_result({
            "strategy_name": strategy.name,
            "symbol": df_backtest["symbol"].iloc[0] if "symbol" in df_backtest.columns else "unknown",
            "start_time": df_backtest["datetime"].min(),
            "end_time": df_backtest["datetime"].max(),
            "initial_capital": self.initial_capital,
            "final_capital": self.cash + self.position_value,
            "total_return": self.performance_metrics["total_return"],
            "annual_return": self.performance_metrics["annual_return"],
            "sharpe_ratio": self.performance_metrics["sharpe_ratio"],
            "max_drawdown": self.performance_metrics["max_drawdown"],
            "win_rate": self.performance_metrics["win_rate"],
            "profit_factor": self.performance_metrics["profit_factor"],
            "params": strategy.params
        })
        
        self.logger.info(f"回测结果已保存: {result_id}")
        return result_id
    
    async def generate_report(self, result_id: str) -> str:
        """
        生成回测报告
        
        Args:
            result_id: 结果ID
            
        Returns:
            报告文件路径
        """
        try:
            # 获取结果目录
            result_dir = self.results_dir / result_id
            
            # 加载数据
            df_backtest = pd.read_csv(result_dir / "backtest_data.csv")
            
            if os.path.exists(result_dir / "trades.csv"):
                trades_df = pd.read_csv(result_dir / "trades.csv")
            else:
                trades_df = pd.DataFrame()
            
            if os.path.exists(result_dir / "equity_curve.csv"):
                equity_df = pd.read_csv(result_dir / "equity_curve.csv")
            else:
                equity_df = pd.DataFrame()
            
            with open(result_dir / "performance_metrics.json", "r") as f:
                performance_metrics = json.load(f)
            
            with open(result_dir / "strategy_params.json", "r") as f:
                strategy_params = json.load(f)
            
            # 创建报告
            report_path = result_dir / "backtest_report.html"
            
            # 生成HTML报告
            with open(report_path, "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>回测报告 - {result_id}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .metric {{ font-weight: bold; }}
                        .positive {{ color: green; }}
                        .negative {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>回测报告</h1>
                    <p>结果ID: {result_id}</p>
                    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <h2>性能指标</h2>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>值</th>
                        </tr>
                        <tr>
                            <td>总收益率</td>
                            <td class="{self._get_color_class(performance_metrics['total_return'])}">{performance_metrics['total_return']:.2f}%</td>
                        </tr>
                        <tr>
                            <td>年化收益率</td>
                            <td class="{self._get_color_class(performance_metrics['annual_return'])}">{performance_metrics['annual_return']:.2f}%</td>
                        </tr>
                        <tr>
                            <td>夏普比率</td>
                            <td class="{self._get_color_class(performance_metrics['sharpe_ratio'])}">{performance_metrics['sharpe_ratio']:.2f}</td>
                        </tr>
                        <tr>
                            <td>最大回撤</td>
                            <td class="{self._get_color_class(-performance_metrics['max_drawdown'])}">{performance_metrics['max_drawdown']:.2f}%</td>
                        </tr>
                        <tr>
                            <td>胜率</td>
                            <td>{performance_metrics['win_rate']:.2f}%</td>
                        </tr>
                        <tr>
                            <td>盈亏比</td>
                            <td>{performance_metrics['profit_factor']:.2f}</td>
                        </tr>
                        <tr>
                            <td>交易次数</td>
                            <td>{performance_metrics['trades_count']}</td>
                        </tr>
                    </table>
                    
                    <h2>策略参数</h2>
                    <table>
                        <tr>
                            <th>参数</th>
                            <th>值</th>
                        </tr>
                """)
                
                # 添加策略参数
                for param, value in strategy_params.items():
                    f.write(f"""
                        <tr>
                            <td>{param}</td>
                            <td>{value}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                """)
                
                # 添加交易记录
                if not trades_df.empty:
                    f.write("""
                    <h2>交易记录</h2>
                    <table>
                        <tr>
                            <th>时间</th>
                            <th>类型</th>
                            <th>价格</th>
                            <th>数量</th>
                            <th>价值</th>
                            <th>手续费</th>
                            <th>盈亏</th>
                        </tr>
                    """)
                    
                    for _, trade in trades_df.iterrows():
                        if trade["type"] == "close":
                            pnl_class = "positive" if trade["pnl"] > 0 else "negative"
                            f.write(f"""
                            <tr>
                                <td>{trade['datetime']}</td>
                                <td>{trade['type']}</td>
                                <td>{trade['price']:.2f}</td>
                                <td>{trade['size']:.6f}</td>
                                <td>{trade['value']:.2f}</td>
                                <td>{trade['fee']:.2f}</td>
                                <td class="{pnl_class}">{trade['pnl']:.2f}</td>
                            </tr>
                            """)
                    
                    f.write("""
                    </table>
                    """)
                
                f.write("""
                </body>
                </html>
                """)
            
            # 生成图表
            if not equity_df.empty:
                # 转换日期时间
                equity_df["datetime"] = pd.to_datetime(equity_df["datetime"])
                
                # 创建图表目录
                charts_dir = result_dir / "charts"
                charts_dir.mkdir(exist_ok=True)
                
                # 绘制权益曲线
                plt.figure(figsize=(12, 6))
                plt.plot(equity_df["datetime"], equity_df["equity"])
                plt.title("权益曲线")
                plt.xlabel("日期")
                plt.ylabel("权益")
                plt.grid(True)
                plt.savefig(charts_dir / "equity_curve.png")
                plt.close()
                
                # 绘制收益率分布
                plt.figure(figsize=(12, 6))
                sns.histplot(equity_df["returns"].dropna() * 100, kde=True)
                plt.title("收益率分布")
                plt.xlabel("收益率 (%)")
                plt.ylabel("频率")
                plt.grid(True)
                plt.savefig(charts_dir / "returns_distribution.png")
                plt.close()
                
                # 绘制回撤曲线
                equity_values = equity_df["equity"].values
                peak = np.maximum.accumulate(equity_values)
                drawdown = (peak - equity_values) / peak * 100
                
                plt.figure(figsize=(12, 6))
                plt.plot(equity_df["datetime"], drawdown)
                plt.title("回撤曲线")
                plt.xlabel("日期")
                plt.ylabel("回撤 (%)")
                plt.grid(True)
                plt.savefig(charts_dir / "drawdown_curve.png")
                plt.close()
            
            self.logger.info(f"回测报告已生成: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"生成回测报告失败: {e}")
            return ""
    
    def _get_color_class(self, value: float) -> str:
        """
        根据值获取颜色类名
        
        Args:
            value: 数值
            
        Returns:
            颜色类名
        """
        return "positive" if value > 0 else "negative" if value < 0 else ""
    
    async def compare_strategies(self, strategies: List[Strategy], df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
        """
        比较多个策略的性能
        
        Args:
            strategies: 策略列表
            df: 包含价格和指标数据的DataFrame
            initial_capital: 初始资金
            
        Returns:
            比较结果字典
        """
        try:
            results = []
            
            # 运行每个策略的回测
            for strategy in strategies:
                result = await self.run_backtest(strategy, df, initial_capital)
                results.append(result)
            
            # 比较结果
            comparison = {
                "strategies": [r["strategy_name"] for r in results],
                "total_returns": [r["total_return"] for r in results],
                "annual_returns": [r["annual_return"] for r in results],
                "sharpe_ratios": [r["sharpe_ratio"] for r in results],
                "max_drawdowns": [r["max_drawdown"] for r in results],
                "win_rates": [r["win_rate"] for r in results],
                "profit_factors": [r["profit_factor"] for r in results],
                "trades_counts": [r["trades_count"] for r in results],
                "result_ids": [r["result_id"] for r in results]
            }
            
            # 生成比较报告
            comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            comparison_dir = self.results_dir / comparison_id
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存比较结果
            with open(comparison_dir / "comparison_results.json", "w") as f:
                json.dump(comparison, f, indent=4)
            
            # 生成比较图表
            self._generate_comparison_charts(comparison, comparison_dir)
            
            self.logger.info(f"策略比较完成: {comparison_id}")
            return {
                "comparison_id": comparison_id,
                "results": results,
                "comparison": comparison
            }
            
        except Exception as e:
            self.logger.error(f"比较策略失败: {e}")
            return {"error": str(e)}
    
    def _generate_comparison_charts(self, comparison: Dict, comparison_dir: Path):
        """
        生成策略比较图表
        
        Args:
            comparison: 比较结果字典
            comparison_dir: 比较结果目录
        """
        try:
            # 创建图表目录
            charts_dir = comparison_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            # 绘制总收益率比较
            plt.figure(figsize=(12, 6))
            plt.bar(comparison["strategies"], comparison["total_returns"])
            plt.title("总收益率比较")
            plt.xlabel("策略")
            plt.ylabel("总收益率 (%)")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / "total_returns_comparison.png")
            plt.close()
            
            # 绘制夏普比率比较
            plt.figure(figsize=(12, 6))
            plt.bar(comparison["strategies"], comparison["sharpe_ratios"])
            plt.title("夏普比率比较")
            plt.xlabel("策略")
            plt.ylabel("夏普比率")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / "sharpe_ratios_comparison.png")
            plt.close()
            
            # 绘制最大回撤比较
            plt.figure(figsize=(12, 6))
            plt.bar(comparison["strategies"], comparison["max_drawdowns"])
            plt.title("最大回撤比较")
            plt.xlabel("策略")
            plt.ylabel("最大回撤 (%)")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / "max_drawdowns_comparison.png")
            plt.close()
            
            # 绘制胜率比较
            plt.figure(figsize=(12, 6))
            plt.bar(comparison["strategies"], comparison["win_rates"])
            plt.title("胜率比较")
            plt.xlabel("策略")
            plt.ylabel("胜率 (%)")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / "win_rates_comparison.png")
            plt.close()
            
            # 绘制综合评分比较（使用夏普比率、总收益率和最大回撤的加权平均）
            scores = []
            for i in range(len(comparison["strategies"])):
                # 归一化指标
                sharpe_score = comparison["sharpe_ratios"][i] / max(comparison["sharpe_ratios"]) if max(comparison["sharpe_ratios"]) > 0 else 0
                return_score = comparison["total_returns"][i] / max(comparison["total_returns"]) if max(comparison["total_returns"]) > 0 else 0
                drawdown_score = 1 - comparison["max_drawdowns"][i] / max(comparison["max_drawdowns"]) if max(comparison["max_drawdowns"]) > 0 else 0
                
                # 计算加权平均分
                score = 0.4 * sharpe_score + 0.4 * return_score + 0.2 * drawdown_score
                scores.append(score * 100)  # 转换为百分比
            
            plt.figure(figsize=(12, 6))
            plt.bar(comparison["strategies"], scores)
            plt.title("综合评分比较")
            plt.xlabel("策略")
            plt.ylabel("评分")
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / "overall_scores_comparison.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"生成比较图表失败: {e}")
    
    async def close(self):
        """
        关闭回测引擎
        """
        await self.storage.close()
        await self.processor.close()
        self.logger.info("回测引擎已关闭")


# 示例策略类
class MovingAverageCrossoverStrategy(Strategy):
    """
    移动平均线交叉策略
    """
    def __init__(self, name: str = "MA_Crossover", params: Dict = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            params: 策略参数字典
        """
        default_params = {
            "fast_period": 10,
            "slow_period": 30
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含价格和指标数据的DataFrame
            
        Returns:
            添加了交易信号的DataFrame
        """
        # 创建副本，避免修改原始数据
        df_signals = df.copy()
        
        # 获取参数
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        
        # 计算移动平均线
        if f"sma_{fast_period}" not in df_signals.columns:
            df_signals[f"sma_{fast_period}"] = df_signals["close"].rolling(window=fast_period).mean()
        
        if f"sma_{slow_period}" not in df_signals.columns:
            df_signals[f"sma_{slow_period}"] = df_signals["close"].rolling(window=slow_period).mean()
        
        # 计算信号
        df_signals["signal"] = 0
        
        # 金叉：快线上穿慢线
        df_signals.loc[(df_signals[f"sma_{fast_period}"] > df_signals[f"sma_{slow_period}"]) & 
                      (df_signals[f"sma_{fast_period}"].shift(1) <= df_signals[f"sma_{slow_period}"].shift(1)), "signal"] = 1
        
        # 死叉：快线下穿慢线
        df_signals.loc[(df_signals[f"sma_{fast_period}"] < df_signals[f"sma_{slow_period}"]) & 
                      (df_signals[f"sma_{fast_period}"].shift(1) >= df_signals[f"sma_{slow_period}"].shift(1)), "signal"] = -1
        
        # 持有信号：保持前一个信号
        for i in range(1, len(df_signals)):
            if df_signals["signal"].iloc[i] == 0:
                df_signals["signal"].iloc[i] = df_signals["signal"].iloc[i-1]
        
        return df_signals


class RSIStrategy(Strategy):
    """
    RSI策略
    """
    def __init__(self, name: str = "RSI", params: Dict = None):
        """
        初始化策略
        
        Args:
            name: 策略名称
            params: 策略参数字典
        """
        default_params = {
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30
        }
        
        if params:
            default_params.update(params)
        
        super().__init__(name, default_params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            df: 包含价格和指标数据的DataFrame
            
        Returns:
            添加了交易信号的DataFrame
        """
        # 创建副本，避免修改原始数据
        df_signals = df.copy()
        
        # 获取参数
        rsi_period = self.params["rsi_period"]
        overbought = self.params["overbought"]
        oversold = self.params["oversold"]
        
        # 计算RSI
        if f"rsi_{rsi_period}" not in df_signals.columns:
            df_signals[f"rsi_{rsi_period}"] = talib.RSI(df_signals["close"].values, timeperiod=rsi_period)
        
        # 计算信号
        df_signals["signal"] = 0
        
        # 超卖：RSI低于超卖线后回升
        df_signals.loc[(df_signals[f"rsi_{rsi_period}"] > oversold) & 
                      (df_signals[f"rsi_{rsi_period}"].shift(1) <= oversold), "signal"] = 1
        
        # 超买：RSI高于超买线后回落
        df_signals.loc[(df_signals[f"rsi_{rsi_period}"] < overbought) & 
                      (df_signals[f"rsi_{rsi_period}"].shift(1) >= overbought), "signal"] = -1
        
        # 持有信号：保持前一个信号
        for i in range(1, len(df_signals)):
            if df_signals["signal"].iloc[i] == 0:
                df_signals["signal"].iloc[i] = df_signals["signal"].iloc[i-1]
        
        return df_signals


async def main():
    """
    主函数
    """
    # 创建回测引擎
    engine = BacktestEngine()
    
    try:
        # 加载数据
        df = await engine.load_data(
            symbol="BTCUSDT",
            interval="1",
            start_date="2024-01-01",
            end_date="2024-04-01",
            indicators=["sma", "ema", "macd", "rsi", "bbands", "atr"]
        )
        
        # 创建策略
        ma_strategy = MovingAverageCrossoverStrategy(params={"fast_period": 10, "slow_period": 30})
        rsi_strategy = RSIStrategy(params={"rsi_period": 14, "overbought": 70, "oversold": 30})
        
        # 运行回测
        ma_result = await engine.run_backtest(ma_strategy, df)
        rsi_result = await engine.run_backtest(rsi_strategy, df)
        
        # 比较策略
        comparison = await engine.compare_strategies([ma_strategy, rsi_strategy], df)
        
        # 打印结果
        print(f"MA策略总收益率: {ma_result['total_return']:.2f}%, 夏普比率: {ma_result['sharpe_ratio']:.2f}")
        print(f"RSI策略总收益率: {rsi_result['total_return']:.2f}%, 夏普比率: {rsi_result['sharpe_ratio']:.2f}")
        
    finally:
        # 关闭回测引擎
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
