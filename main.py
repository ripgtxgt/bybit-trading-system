#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bybit高频交易系统主程序入口

用法:
    python main.py [模块] [参数]

模块:
    data_collection  - 数据收集模块
    backtest         - 回测模块
    optimize         - 策略优化模块
    trading          - 交易执行模块
    all              - 运行所有模块
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import Logger
from src.utils.config import Config
from src.data_collection.websocket_client import BybitWebSocketClient
from src.data_collection.historical_data_downloader import BybitHistoricalDataDownloader
from src.storage.data_storage import DataStorage
from src.data_processing.data_processor import DataProcessor
from src.backtesting.backtest_engine import BacktestEngine, MovingAverageCrossoverStrategy, RSIStrategy
from src.strategy.strategy_optimizer import StrategyOptimizer


async def run_data_collection(args):
    """
    运行数据收集模块
    """
    logger = Logger.get_logger("data_collection")
    logger.info("启动数据收集模块")
    
    config = Config()
    symbol = args.symbol or config.get("trading", "symbol", "BTCUSDT")
    interval = args.interval or config.get("trading", "interval", "1")
    
    # 创建数据存储
    storage = DataStorage()
    
    # 如果指定了下载历史数据
    if args.download_history:
        logger.info(f"开始下载历史数据: {symbol}, 间隔: {interval}分钟")
        
        # 创建历史数据下载器
        downloader = BybitHistoricalDataDownloader(
            symbol=symbol,
            interval=interval,
            start_time=args.start_date,
            end_time=args.end_date
        )
        
        # 下载数据
        df = await downloader.download_all()
        
        # 保存数据
        file_path = await downloader.save_to_csv(df)
        logger.info(f"历史数据已保存至: {file_path}")
        
        # 存储到数据库
        await storage.store_kline_data(df)
        logger.info(f"历史数据已存储到数据库")
    
    # 如果指定了订阅实时数据
    if args.subscribe_realtime:
        logger.info(f"开始订阅实时数据: {symbol}, 间隔: {interval}分钟")
        
        # 创建WebSocket客户端
        client = BybitWebSocketClient(
            symbol=symbol,
            channel="kline",
            interval=interval
        )
        
        # 定义消息处理函数
        async def handle_message(message):
            try:
                # 处理并存储数据
                await storage.store_websocket_message(message)
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
        
        # 设置消息处理函数
        client.set_message_handler(handle_message)
        
        try:
            # 连接WebSocket
            await client.connect()
            
            # 保持运行直到用户中断
            logger.info("WebSocket客户端已连接，按Ctrl+C中断")
            while True:
                await asyncio.sleep(60)
                logger.info("WebSocket客户端运行中...")
                
        except KeyboardInterrupt:
            logger.info("用户中断，正在关闭WebSocket客户端")
        except Exception as e:
            logger.error(f"WebSocket客户端出错: {e}")
        finally:
            # 断开连接
            await client.disconnect()
    
    # 关闭数据存储
    await storage.close()
    logger.info("数据收集模块已关闭")


async def run_backtest(args):
    """
    运行回测模块
    """
    logger = Logger.get_logger("backtest")
    logger.info("启动回测模块")
    
    config = Config()
    symbol = args.symbol or config.get("trading", "symbol", "BTCUSDT")
    interval = args.interval or config.get("trading", "interval", "1")
    
    # 创建回测引擎
    engine = BacktestEngine()
    
    # 加载数据
    logger.info(f"加载回测数据: {symbol}, 间隔: {interval}分钟, 时间范围: {args.start_date} - {args.end_date}")
    df = await engine.load_data(
        symbol=symbol,
        interval=interval,
        start_date=args.start_date,
        end_date=args.end_date,
        indicators=["sma", "ema", "macd", "rsi", "bbands", "atr"]
    )
    
    # 选择策略
    if args.strategy == "ma":
        logger.info("使用移动平均线交叉策略")
        strategy = MovingAverageCrossoverStrategy(
            params={"fast_period": args.fast_period, "slow_period": args.slow_period}
        )
    elif args.strategy == "rsi":
        logger.info("使用RSI策略")
        strategy = RSIStrategy(
            params={"rsi_period": args.rsi_period, "overbought": args.overbought, "oversold": args.oversold}
        )
    else:
        logger.error(f"未知的策略: {args.strategy}")
        return
    
    # 运行回测
    logger.info("开始回测")
    result = await engine.run_backtest(strategy, df, initial_capital=args.capital)
    
    # 打印结果
    logger.info("回测完成，结果如下:")
    logger.info(f"总收益率: {result['total_return']:.2f}%")
    logger.info(f"年化收益率: {result['annual_return']:.2f}%")
    logger.info(f"夏普比率: {result['sharpe_ratio']:.2f}")
    logger.info(f"最大回撤: {result['max_drawdown']:.2f}%")
    logger.info(f"胜率: {result['win_rate']:.2f}%")
    logger.info(f"盈亏比: {result['profit_factor']:.2f}")
    logger.info(f"交易次数: {result['trades_count']}")
    
    # 保存结果
    if args.save_results:
        report_path = await engine.save_backtest_report(result, strategy.name, symbol, interval)
        logger.info(f"回测报告已保存至: {report_path}")
    
    # 关闭回测引擎
    await engine.close()
    logger.info("回测模块已关闭")


async def run_optimize(args):
    """
    运行策略优化模块
    """
    logger = Logger.get_logger("optimize")
    logger.info("启动策略优化模块")
    
    config = Config()
    symbol = args.symbol or config.get("trading", "symbol", "BTCUSDT")
    interval = args.interval or config.get("trading", "interval", "1")
    
    # 创建回测引擎和优化器
    engine = BacktestEngine()
    optimizer = StrategyOptimizer()
    
    # 加载数据
    logger.info(f"加载优化数据: {symbol}, 间隔: {interval}分钟, 时间范围: {args.start_date} - {args.end_date}")
    df = await engine.load_data(
        symbol=symbol,
        interval=interval,
        start_date=args.start_date,
        end_date=args.end_date,
        indicators=["sma", "ema", "macd", "rsi", "bbands", "atr"]
    )
    
    # 选择策略和参数网格
    if args.strategy == "ma":
        logger.info("优化移动平均线交叉策略")
        strategy_class = MovingAverageCrossoverStrategy
        param_grid = {
            "fast_period": list(range(args.fast_min, args.fast_max + 1, args.fast_step)),
            "slow_period": list(range(args.slow_min, args.slow_max + 1, args.slow_step))
        }
    elif args.strategy == "rsi":
        logger.info("优化RSI策略")
        strategy_class = RSIStrategy
        param_grid = {
            "rsi_period": list(range(args.rsi_min, args.rsi_max + 1, args.rsi_step)),
            "overbought": list(range(args.ob_min, args.ob_max + 1, args.ob_step)),
            "oversold": list(range(args.os_min, args.os_max + 1, args.os_step))
        }
    else:
        logger.error(f"未知的策略: {args.strategy}")
        return
    
    # 打印参数网格
    logger.info(f"参数网格: {param_grid}")
    
    # 选择优化方法
    if args.method == "grid":
        logger.info("使用网格搜索优化")
        result = await optimizer.optimize(
            strategy_class=strategy_class,
            param_grid=param_grid,
            df=df,
            initial_capital=args.capital,
            method="grid",
            metric=args.metric
        )
    elif args.method == "random":
        logger.info("使用随机搜索优化")
        result = await optimizer.optimize(
            strategy_class=strategy_class,
            param_grid=param_grid,
            df=df,
            initial_capital=args.capital,
            method="random",
            metric=args.metric,
            max_iterations=args.iterations
        )
    elif args.method == "genetic":
        logger.info("使用遗传算法优化")
        result = await optimizer.optimize(
            strategy_class=strategy_class,
            param_grid=param_grid,
            df=df,
            initial_capital=args.capital,
            method="genetic",
            metric=args.metric,
            max_iterations=args.iterations,
            population_size=args.population
        )
    elif args.method == "walk_forward":
        logger.info("使用滚动优化")
        result = await optimizer.walk_forward_optimization(
            strategy_class=strategy_class,
            param_grid=param_grid,
            df=df,
            window_size=args.window,
            step_size=args.step,
            method=args.sub_method,
            metric=args.metric
        )
    else:
        logger.error(f"未知的优化方法: {args.method}")
        return
    
    # 打印结果
    logger.info("优化完成，最佳参数如下:")
    logger.info(f"最佳参数: {result['best_params']}")
    logger.info(f"最佳分数 ({args.metric}): {result['best_score']:.4f}")
    
    # 关闭优化器和回测引擎
    await optimizer.close()
    await engine.close()
    logger.info("策略优化模块已关闭")


async def run_trading(args):
    """
    运行交易执行模块
    """
    logger = Logger.get_logger("trading")
    logger.info("启动交易执行模块")
    
    logger.warning("交易执行模块尚未实现")
    logger.info("交易执行模块已关闭")


async def main():
    """
    主函数
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Bybit高频交易系统")
    subparsers = parser.add_subparsers(dest="module", help="模块")
    
    # 数据收集模块参数
    data_parser = subparsers.add_parser("data_collection", help="数据收集模块")
    data_parser.add_argument("--symbol", type=str, help="交易对，如BTCUSDT")
    data_parser.add_argument("--interval", type=str, help="K线间隔，如1（分钟）")
    data_parser.add_argument("--download-history", action="store_true", help="下载历史数据")
    data_parser.add_argument("--subscribe-realtime", action="store_true", help="订阅实时数据")
    data_parser.add_argument("--start-date", type=str, help="开始日期，如2024-01-01")
    data_parser.add_argument("--end-date", type=str, help="结束日期，如2024-04-01")
    
    # 回测模块参数
    backtest_parser = subparsers.add_parser("backtest", help="回测模块")
    backtest_parser.add_argument("--symbol", type=str, help="交易对，如BTCUSDT")
    backtest_parser.add_argument("--interval", type=str, help="K线间隔，如1（分钟）")
    backtest_parser.add_argument("--start-date", type=str, required=True, help="开始日期，如2024-01-01")
    backtest_parser.add_argument("--end-date", type=str, required=True, help="结束日期，如2024-04-01")
    backtest_parser.add_argument("--strategy", type=str, required=True, choices=["ma", "rsi"], help="策略类型")
    backtest_parser.add_argument("--capital", type=float, default=10000.0, help="初始资金")
    backtest_parser.add_argument("--save-results", action="store_true", help="保存回测结果")
    
    # MA策略参数
    backtest_parser.add_argument("--fast-period", type=int, default=10, help="快速均线周期")
    backtest_parser.add_argument("--slow-period", type=int, default=30, help="慢速均线周期")
    
    # RSI策略参数
    backtest_parser.add_argument("--rsi-period", type=int, default=14, help="RSI周期")
    backtest_parser.add_argument("--overbought", type=int, default=70, help="超买阈值")
    backtest_parser.add_argument("--oversold", type=int, default=30, help="超卖阈值")
    
    # 策略优化模块参数
    optimize_parser = subparsers.add_parser("optimize", help="策略优化模块")
    optimize_parser.add_argument("--symbol", type=str, help="交易对，如BTCUSDT")
    optimize_parser.add_argument("--interval", type=str, help="K线间隔，如1（分钟）")
    optimize_parser.add_argument("--start-date", type=str, required=True, help="开始日期，如2024-01-01")
    optimize_parser.add_argument("--end-date", type=str, required=True, help="结束日期，如2024-04-01")
    optimize_parser.add_argument("--strategy", type=str, required=True, choices=["ma", "rsi"], help="策略类型")
    optimize_parser.add_argument("--method", type=str, required=True, 
                                choices=["grid", "random", "genetic", "walk_forward"], help="优化方法")
    optimize_parser.add_argument("--metric", type=str, default="sharpe_ratio", 
                                choices=["sharpe_ratio", "total_return", "annual_return", "max_drawdown", "win_rate", "profit_factor"], 
                                help="优化指标")
    optimize_parser.add_argument("--capital", type=float, default=10000.0, help="初始资金")
    optimize_parser.add_argument("--iterations", type=int, default=100, help="迭代次数（随机搜索和遗传算法）")
    optimize_parser.add_argument("--population", type=int, default=50, help="种群大小（遗传算法）")
    optimize_parser.add_argument("--window", type=int, default=30, help="滚动窗口大小（天）")
    optimize_parser.add_argument("--step", type=int, default=10, help="滚动步长（天）")
    optimize_parser.add_argument("--sub-method", type=str, default="grid", 
                                choices=["grid", "random", "genetic"], help="滚动优化中的子优化方法")
    
    # MA策略参数范围
    optimize_parser.add_argument("--fast-min", type=int, default=5, help="快速均线最小周期")
    optimize_parser.add_argument("--fast-max", type=int, default=20, help="快速均线最大周期")
    optimize_parser.add_argument("--fast-step", type=int, default=5, help="快速均线步长")
    optimize_parser.add_argument("--slow-min", type=int, default=20, help="慢速均线最小周期")
    optimize_parser.add_argument("--slow-max", type=int, default=50, help="慢速均线最大周期")
    optimize_parser.add_argument("--slow-step", type=int, default=10, help="慢速均线步长")
    
    # RSI策略参数范围
    optimize_parser.add_argument("--rsi-min", type=int, default=7, help="RSI最小周期")
    optimize_parser.add_argument("--rsi-max", type=int, default=21, help="RSI最大周期")
    optimize_parser.add_argument("--rsi-step", type=int, default=7, help="RSI步长")
    optimize_parser.add_argument("--ob-min", type=int, default=65, help="超买最小阈值")
    optimize_parser.add_argument("--ob-max", type=int, default=80, help="超买最大阈值")
    optimize_parser.add_argument("--ob-step", type=int, default=5, help="超买步长")
    optimize_parser.add_argument("--os-min", type=int, default=20, help="超卖最小阈值")
    optimize_parser.add_argument("--os-max", type=int, default=35, help="超卖最大阈值")
    optimize_parser.add_argument("--os-step", type=int, default=5, help="超卖步长")
    
    # 交易执行模块参数
    trading_parser = subparsers.add_parser("trading", help="交易执行模块")
    trading_parser.add_argument("--symbol", type=str, help="交易对，如BTCUSDT")
    trading_parser.add_argument("--strategy", type=str, required=True, choices=["ma", "rsi"], help="策略类型")
    trading_parser.add_argument("--paper-trading", action="store_true", help="使用模拟交易")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据模块运行相应的函数
    if args.module == "data_collection":
        await run_data_collection(args)
    elif args.module == "backtest":
        await run_backtest(args)
    elif args.module == "optimize":
        await run_optimize(args)
    elif args.module == "trading":
        await run_trading(args)
    elif args.module == "all":
        # 运行所有模块
        pass
    else:
        parser.print_help()


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行主函数
    asyncio.run(main())
