import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Type, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import random
from tqdm import tqdm

from ..utils.logger import Logger
from ..utils.config import Config
from ..storage.data_storage import DataStorage
from ..data_processing.data_processor import DataProcessor
from ..backtesting.backtest_engine import BacktestEngine, Strategy


class StrategyOptimizer:
    """
    策略优化器，用于寻找最佳策略参数
    """
    def __init__(self):
        """
        初始化策略优化器
        """
        self.logger = Logger.get_logger("strategy_optimizer")
        self.config = Config()
        self.storage = DataStorage()
        self.processor = DataProcessor()
        self.backtest_engine = BacktestEngine()
        
        # 优化配置
        self.optimization_method = "grid"  # 默认使用网格搜索
        self.max_workers = multiprocessing.cpu_count() - 1  # 使用CPU核心数-1个进程
        self.max_iterations = 100  # 默认最大迭代次数
        self.population_size = 50  # 默认种群大小（用于遗传算法）
        self.mutation_rate = 0.1  # 默认变异率（用于遗传算法）
        self.crossover_rate = 0.8  # 默认交叉率（用于遗传算法）
        
        # 创建结果目录
        self.results_dir = Path("data/optimization_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def optimize(self, strategy_class: Type[Strategy], param_grid: Dict[str, List], 
                      df: pd.DataFrame, initial_capital: float = 10000.0, 
                      method: str = "grid", metric: str = "sharpe_ratio",
                      max_workers: int = None, max_iterations: int = None) -> Dict:
        """
        优化策略参数
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格，格式为 {参数名: [参数值列表]}
            df: 包含价格和指标数据的DataFrame
            initial_capital: 初始资金
            method: 优化方法，可选 "grid"（网格搜索）, "random"（随机搜索）, "genetic"（遗传算法）
            metric: 优化指标，可选 "sharpe_ratio", "total_return", "annual_return", "max_drawdown", "win_rate", "profit_factor"
            max_workers: 最大工作进程数
            max_iterations: 最大迭代次数
            
        Returns:
            优化结果字典
        """
        try:
            # 更新配置
            self.optimization_method = method
            if max_workers is not None:
                self.max_workers = max_workers
            if max_iterations is not None:
                self.max_iterations = max_iterations
            
            # 根据优化方法选择优化函数
            if method == "grid":
                result = await self._grid_search(strategy_class, param_grid, df, initial_capital, metric)
            elif method == "random":
                result = await self._random_search(strategy_class, param_grid, df, initial_capital, metric, self.max_iterations)
            elif method == "genetic":
                result = await self._genetic_algorithm(strategy_class, param_grid, df, initial_capital, metric, self.max_iterations, self.population_size)
            else:
                self.logger.error(f"未知的优化方法: {method}")
                return {"error": f"未知的优化方法: {method}"}
            
            # 保存优化结果
            optimization_id = await self._save_optimization_results(strategy_class.__name__, param_grid, result, method, metric)
            
            # 返回优化结果
            return {
                "optimization_id": optimization_id,
                "strategy_name": strategy_class.__name__,
                "method": method,
                "metric": metric,
                "best_params": result["best_params"],
                "best_score": result["best_score"],
                "best_result": result["best_result"],
                "all_results": result["all_results"][:10]  # 只返回前10个结果
            }
            
        except Exception as e:
            self.logger.error(f"优化策略参数失败: {e}")
            return {"error": str(e)}
    
    async def _grid_search(self, strategy_class: Type[Strategy], param_grid: Dict[str, List], 
                          df: pd.DataFrame, initial_capital: float, metric: str) -> Dict:
        """
        网格搜索优化
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格
            df: 数据DataFrame
            initial_capital: 初始资金
            metric: 优化指标
            
        Returns:
            优化结果字典
        """
        self.logger.info(f"开始网格搜索优化: {strategy_class.__name__}")
        
        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"参数组合数量: {len(param_combinations)}")
        
        # 创建参数字典列表
        param_dicts = []
        for combination in param_combinations:
            param_dict = {name: value for name, value in zip(param_names, combination)}
            param_dicts.append(param_dict)
        
        # 并行回测
        results = await self._parallel_backtest(strategy_class, param_dicts, df, initial_capital)
        
        # 根据指标排序
        if metric == "max_drawdown":
            # 对于最大回撤，值越小越好
            sorted_results = sorted(results, key=lambda x: x[metric])
        else:
            # 对于其他指标，值越大越好
            sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
        
        # 获取最佳结果
        best_result = sorted_results[0]
        
        return {
            "best_params": best_result["params"],
            "best_score": best_result[metric],
            "best_result": best_result,
            "all_results": sorted_results
        }
    
    async def _random_search(self, strategy_class: Type[Strategy], param_grid: Dict[str, List], 
                            df: pd.DataFrame, initial_capital: float, metric: str, 
                            n_iterations: int) -> Dict:
        """
        随机搜索优化
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格
            df: 数据DataFrame
            initial_capital: 初始资金
            metric: 优化指标
            n_iterations: 迭代次数
            
        Returns:
            优化结果字典
        """
        self.logger.info(f"开始随机搜索优化: {strategy_class.__name__}, 迭代次数: {n_iterations}")
        
        # 生成随机参数组合
        param_dicts = []
        for _ in range(n_iterations):
            param_dict = {}
            for param_name, param_values in param_grid.items():
                param_dict[param_name] = random.choice(param_values)
            param_dicts.append(param_dict)
        
        # 并行回测
        results = await self._parallel_backtest(strategy_class, param_dicts, df, initial_capital)
        
        # 根据指标排序
        if metric == "max_drawdown":
            # 对于最大回撤，值越小越好
            sorted_results = sorted(results, key=lambda x: x[metric])
        else:
            # 对于其他指标，值越大越好
            sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
        
        # 获取最佳结果
        best_result = sorted_results[0]
        
        return {
            "best_params": best_result["params"],
            "best_score": best_result[metric],
            "best_result": best_result,
            "all_results": sorted_results
        }
    
    async def _genetic_algorithm(self, strategy_class: Type[Strategy], param_grid: Dict[str, List], 
                                df: pd.DataFrame, initial_capital: float, metric: str, 
                                n_generations: int, population_size: int) -> Dict:
        """
        遗传算法优化
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格
            df: 数据DataFrame
            initial_capital: 初始资金
            metric: 优化指标
            n_generations: 代数
            population_size: 种群大小
            
        Returns:
            优化结果字典
        """
        self.logger.info(f"开始遗传算法优化: {strategy_class.__name__}, 代数: {n_generations}, 种群大小: {population_size}")
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, param_values in param_grid.items():
                individual[param_name] = random.choice(param_values)
            population.append(individual)
        
        all_results = []
        best_individual = None
        best_score = float('-inf') if metric != "max_drawdown" else float('inf')
        
        # 进化
        for generation in range(n_generations):
            self.logger.info(f"第 {generation + 1}/{n_generations} 代")
            
            # 评估适应度
            fitness_results = await self._parallel_backtest(strategy_class, population, df, initial_capital)
            all_results.extend(fitness_results)
            
            # 根据指标排序
            if metric == "max_drawdown":
                # 对于最大回撤，值越小越好
                sorted_results = sorted(fitness_results, key=lambda x: x[metric])
            else:
                # 对于其他指标，值越大越好
                sorted_results = sorted(fitness_results, key=lambda x: x[metric], reverse=True)
            
            # 更新最佳个体
            current_best = sorted_results[0]
            if metric == "max_drawdown":
                if current_best[metric] < best_score:
                    best_individual = current_best
                    best_score = current_best[metric]
            else:
                if current_best[metric] > best_score:
                    best_individual = current_best
                    best_score = current_best[metric]
            
            # 选择精英个体
            elite_size = max(1, int(population_size * 0.1))  # 保留10%的精英
            elites = [result["params"] for result in sorted_results[:elite_size]]
            
            # 创建新一代
            new_population = elites.copy()
            
            # 通过交叉和变异生成剩余个体
            while len(new_population) < population_size:
                # 选择父母
                parent1 = self._tournament_selection(sorted_results, metric)
                parent2 = self._tournament_selection(sorted_results, metric)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1["params"], parent2["params"])
                else:
                    child = parent1["params"].copy()
                
                # 变异
                child = self._mutate(child, param_grid)
                
                new_population.append(child)
            
            # 更新种群
            population = new_population
        
        # 根据指标排序所有结果
        if metric == "max_drawdown":
            sorted_all_results = sorted(all_results, key=lambda x: x[metric])
        else:
            sorted_all_results = sorted(all_results, key=lambda x: x[metric], reverse=True)
        
        return {
            "best_params": best_individual["params"],
            "best_score": best_score,
            "best_result": best_individual,
            "all_results": sorted_all_results
        }
    
    def _tournament_selection(self, results: List[Dict], metric: str, tournament_size: int = 3) -> Dict:
        """
        锦标赛选择
        
        Args:
            results: 结果列表
            metric: 优化指标
            tournament_size: 锦标赛大小
            
        Returns:
            选中的个体
        """
        tournament = random.sample(results, min(tournament_size, len(results)))
        
        if metric == "max_drawdown":
            return min(tournament, key=lambda x: x[metric])
        else:
            return max(tournament, key=lambda x: x[metric])
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        交叉操作
        
        Args:
            parent1: 父亲参数
            parent2: 母亲参数
            
        Returns:
            子代参数
        """
        child = {}
        for param_name in parent1.keys():
            # 随机选择父母的一个参数
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child
    
    def _mutate(self, individual: Dict, param_grid: Dict[str, List]) -> Dict:
        """
        变异操作
        
        Args:
            individual: 个体参数
            param_grid: 参数网格
            
        Returns:
            变异后的参数
        """
        mutated = individual.copy()
        for param_name, param_values in param_grid.items():
            # 以变异率的概率进行变异
            if random.random() < self.mutation_rate:
                mutated[param_name] = random.choice(param_values)
        return mutated
    
    async def _parallel_backtest(self, strategy_class: Type[Strategy], param_dicts: List[Dict], 
                               df: pd.DataFrame, initial_capital: float) -> List[Dict]:
        """
        并行回测
        
        Args:
            strategy_class: 策略类
            param_dicts: 参数字典列表
            df: 数据DataFrame
            initial_capital: 初始资金
            
        Returns:
            回测结果列表
        """
        results = []
        
        # 创建进度条
        progress_bar = tqdm(total=len(param_dicts), desc="回测进度")
        
        # 定义回测函数
        async def backtest_with_params(params):
            try:
                # 创建策略实例
                strategy = strategy_class(params=params)
                
                # 运行回测
                result = await self.backtest_engine.run_backtest(strategy, df, initial_capital)
                
                # 更新进度条
                progress_bar.update(1)
                
                return result
            except Exception as e:
                self.logger.error(f"回测失败: {params}, 错误: {e}")
                return None
        
        # 创建任务
        tasks = [backtest_with_params(params) for params in param_dicts]
        
        # 并行执行任务
        for future in asyncio.as_completed(tasks):
            result = await future
            if result and "error" not in result:
                results.append(result)
        
        # 关闭进度条
        progress_bar.close()
        
        return results
    
    async def _save_optimization_results(self, strategy_name: str, param_grid: Dict[str, List], 
                                       result: Dict, method: str, metric: str) -> str:
        """
        保存优化结果
        
        Args:
            strategy_name: 策略名称
            param_grid: 参数网格
            result: 优化结果
            method: 优化方法
            metric: 优化指标
            
        Returns:
            优化结果ID
        """
        # 生成结果ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimization_id = f"{strategy_name}_{method}_{metric}_{timestamp}"
        
        # 创建结果目录
        result_dir = self.results_dir / optimization_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存参数网格
        with open(result_dir / "param_grid.json", "w") as f:
            json.dump(param_grid, f, indent=4)
        
        # 保存最佳参数
        with open(result_dir / "best_params.json", "w") as f:
            json.dump(result["best_params"], f, indent=4)
        
        # 保存最佳结果
        with open(result_dir / "best_result.json", "w") as f:
            json.dump(result["best_result"], f, indent=4)
        
        # 保存所有结果
        all_results_df = pd.DataFrame(result["all_results"])
        all_results_df.to_csv(result_dir / "all_results.csv", index=False)
        
        # 生成优化报告
        await self._generate_optimization_report(optimization_id, strategy_name, param_grid, result, method, metric)
        
        # 保存到数据库
        await self.storage.store_strategy_params(strategy_name, "BTCUSDT", result["best_params"])
        
        self.logger.info(f"优化结果已保存: {optimization_id}")
        return optimization_id
    
    async def _generate_optimization_report(self, optimization_id: str, strategy_name: str, 
                                          param_grid: Dict[str, List], result: Dict, 
                                          method: str, metric: str) -> str:
        """
        生成优化报告
        
        Args:
            optimization_id: 优化结果ID
            strategy_name: 策略名称
            param_grid: 参数网格
            result: 优化结果
            method: 优化方法
            metric: 优化指标
            
        Returns:
            报告文件路径
        """
        try:
            # 获取结果目录
            result_dir = self.results_dir / optimization_id
            
            # 创建图表目录
            charts_dir = result_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            # 加载所有结果
            all_results_df = pd.read_csv(result_dir / "all_results.csv")
            
            # 创建报告
            report_path = result_dir / "optimization_report.html"
            
            # 生成HTML报告
            with open(report_path, "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>优化报告 - {optimization_id}</title>
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
                    <h1>策略优化报告</h1>
                    <p>优化ID: {optimization_id}</p>
                    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <h2>优化配置</h2>
                    <table>
                        <tr>
                            <th>配置项</th>
                            <th>值</th>
                        </tr>
                        <tr>
                            <td>策略名称</td>
                            <td>{strategy_name}</td>
                        </tr>
                        <tr>
                            <td>优化方法</td>
                            <td>{method}</td>
                        </tr>
                        <tr>
                            <td>优化指标</td>
                            <td>{metric}</td>
                        </tr>
                    </table>
                    
                    <h2>参数网格</h2>
                    <table>
                        <tr>
                            <th>参数</th>
                            <th>值范围</th>
                        </tr>
                """)
                
                # 添加参数网格
                for param_name, param_values in param_grid.items():
                    f.write(f"""
                        <tr>
                            <td>{param_name}</td>
                            <td>{param_values}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                    
                    <h2>最佳参数</h2>
                    <table>
                        <tr>
                            <th>参数</th>
                            <th>值</th>
                        </tr>
                """)
                
                # 添加最佳参数
                for param_name, param_value in result["best_params"].items():
                    f.write(f"""
                        <tr>
                            <td>{param_name}</td>
                            <td>{param_value}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                    
                    <h2>最佳结果</h2>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>值</th>
                        </tr>
                """)
                
                # 添加最佳结果
                metrics = ["total_return", "annual_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor", "trades_count"]
                for m in metrics:
                    if m in result["best_result"]:
                        value = result["best_result"][m]
                        if m in ["total_return", "annual_return", "max_drawdown", "win_rate"]:
                            value_str = f"{value:.2f}%"
                        elif m in ["sharpe_ratio", "profit_factor"]:
                            value_str = f"{value:.2f}"
                        else:
                            value_str = str(value)
                        
                        color_class = ""
                        if m in ["total_return", "annual_return", "sharpe_ratio", "win_rate", "profit_factor"]:
                            color_class = "positive" if value > 0 else "negative" if value < 0 else ""
                        elif m == "max_drawdown":
                            color_class = "negative"
                        
                        f.write(f"""
                            <tr>
                                <td>{m}</td>
                                <td class="{color_class}">{value_str}</td>
                            </tr>
                        """)
                
                f.write("""
                    </table>
                    
                    <h2>优化结果分布</h2>
                    <p>下面的图表展示了不同参数组合的性能分布。</p>
                    <img src="charts/metric_distribution.png" alt="指标分布" style="width: 100%; max-width: 800px;">
                    
                    <h2>参数敏感性分析</h2>
                    <p>下面的图表展示了不同参数对性能的影响。</p>
                """)
                
                # 为每个参数添加敏感性分析图
                for param_name in param_grid.keys():
                    f.write(f"""
                    <h3>参数: {param_name}</h3>
                    <img src="charts/sensitivity_{param_name}.png" alt="{param_name}敏感性" style="width: 100%; max-width: 800px;">
                    """)
                
                f.write("""
                    <h2>前10个最佳结果</h2>
                    <table>
                        <tr>
                            <th>排名</th>
                """)
                
                # 添加参数列
                for param_name in param_grid.keys():
                    f.write(f"""
                            <th>{param_name}</th>
                    """)
                
                # 添加指标列
                for m in metrics:
                    if m in all_results_df.columns:
                        f.write(f"""
                                <th>{m}</th>
                        """)
                
                f.write("""
                        </tr>
                """)
                
                # 添加前10个最佳结果
                top_results = all_results_df.sort_values(by=metric, ascending=(metric == "max_drawdown")).head(10)
                for i, (_, row) in enumerate(top_results.iterrows()):
                    f.write(f"""
                        <tr>
                            <td>{i+1}</td>
                    """)
                    
                    # 添加参数值
                    for param_name in param_grid.keys():
                        param_value = row["params"].get(param_name, "") if isinstance(row["params"], dict) else ""
                        f.write(f"""
                                <td>{param_value}</td>
                        """)
                    
                    # 添加指标值
                    for m in metrics:
                        if m in row:
                            value = row[m]
                            if m in ["total_return", "annual_return", "max_drawdown", "win_rate"]:
                                value_str = f"{value:.2f}%"
                            elif m in ["sharpe_ratio", "profit_factor"]:
                                value_str = f"{value:.2f}"
                            else:
                                value_str = str(value)
                            
                            color_class = ""
                            if m in ["total_return", "annual_return", "sharpe_ratio", "win_rate", "profit_factor"]:
                                color_class = "positive" if value > 0 else "negative" if value < 0 else ""
                            elif m == "max_drawdown":
                                color_class = "negative"
                            
                            f.write(f"""
                                    <td class="{color_class}">{value_str}</td>
                            """)
                    
                    f.write("""
                        </tr>
                    """)
                
                f.write("""
                    </table>
                </body>
                </html>
                """)
            
            # 生成图表
            self._generate_optimization_charts(all_results_df, param_grid, metric, charts_dir)
            
            self.logger.info(f"优化报告已生成: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"生成优化报告失败: {e}")
            return ""
    
    def _generate_optimization_charts(self, results_df: pd.DataFrame, param_grid: Dict[str, List], 
                                     metric: str, charts_dir: Path):
        """
        生成优化图表
        
        Args:
            results_df: 结果DataFrame
            param_grid: 参数网格
            metric: 优化指标
            charts_dir: 图表目录
        """
        try:
            # 绘制指标分布
            plt.figure(figsize=(12, 6))
            sns.histplot(results_df[metric].dropna(), kde=True)
            plt.title(f"{metric}分布")
            plt.xlabel(metric)
            plt.ylabel("频率")
            plt.grid(True)
            plt.savefig(charts_dir / "metric_distribution.png")
            plt.close()
            
            # 为每个参数绘制敏感性分析图
            for param_name in param_grid.keys():
                # 提取参数值
                param_values = []
                for params_str in results_df["params"]:
                    if isinstance(params_str, str):
                        try:
                            params = json.loads(params_str.replace("'", "\""))
                            param_values.append(params.get(param_name, None))
                        except:
                            param_values.append(None)
                    elif isinstance(params_str, dict):
                        param_values.append(params_str.get(param_name, None))
                    else:
                        param_values.append(None)
                
                # 创建参数值列
                results_df[f"param_{param_name}"] = param_values
                
                # 绘制箱线图
                plt.figure(figsize=(12, 6))
                sns.boxplot(x=f"param_{param_name}", y=metric, data=results_df)
                plt.title(f"{param_name}对{metric}的影响")
                plt.xlabel(param_name)
                plt.ylabel(metric)
                plt.grid(True)
                plt.savefig(charts_dir / f"sensitivity_{param_name}.png")
                plt.close()
            
        except Exception as e:
            self.logger.error(f"生成优化图表失败: {e}")
    
    async def walk_forward_optimization(self, strategy_class: Type[Strategy], param_grid: Dict[str, List], 
                                       df: pd.DataFrame, window_size: int = 30, step_size: int = 10, 
                                       method: str = "grid", metric: str = "sharpe_ratio") -> Dict:
        """
        滚动优化
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格
            df: 数据DataFrame
            window_size: 窗口大小（天）
            step_size: 步长（天）
            method: 优化方法
            metric: 优化指标
            
        Returns:
            优化结果字典
        """
        try:
            self.logger.info(f"开始滚动优化: {strategy_class.__name__}, 窗口大小: {window_size}天, 步长: {step_size}天")
            
            # 确保日期列存在
            if "datetime" not in df.columns:
                self.logger.error("数据中缺少datetime列")
                return {"error": "数据中缺少datetime列"}
            
            # 转换日期列
            df["datetime"] = pd.to_datetime(df["datetime"])
            
            # 获取日期范围
            start_date = df["datetime"].min()
            end_date = df["datetime"].max()
            
            # 计算窗口
            current_date = start_date
            windows = []
            
            while current_date + timedelta(days=window_size) <= end_date:
                train_start = current_date
                train_end = current_date + timedelta(days=window_size)
                test_start = train_end
                test_end = min(test_start + timedelta(days=step_size), end_date)
                
                windows.append({
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end
                })
                
                current_date += timedelta(days=step_size)
            
            self.logger.info(f"共创建 {len(windows)} 个窗口")
            
            # 对每个窗口进行优化
            window_results = []
            
            for i, window in enumerate(windows):
                self.logger.info(f"优化窗口 {i+1}/{len(windows)}: {window['train_start']} - {window['train_end']}")
                
                # 获取训练数据
                train_df = df[(df["datetime"] >= window["train_start"]) & (df["datetime"] < window["train_end"])]
                
                # 优化参数
                optimization_result = await self.optimize(
                    strategy_class=strategy_class,
                    param_grid=param_grid,
                    df=train_df,
                    method=method,
                    metric=metric
                )
                
                # 获取测试数据
                test_df = df[(df["datetime"] >= window["test_start"]) & (df["datetime"] < window["test_end"])]
                
                # 使用最佳参数在测试数据上回测
                strategy = strategy_class(params=optimization_result["best_params"])
                test_result = await self.backtest_engine.run_backtest(strategy, test_df)
                
                # 记录结果
                window_result = {
                    "window_index": i,
                    "train_start": window["train_start"],
                    "train_end": window["train_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "best_params": optimization_result["best_params"],
                    "train_score": optimization_result["best_score"],
                    "test_score": test_result[metric],
                    "test_result": test_result
                }
                
                window_results.append(window_result)
            
            # 计算整体性能
            overall_performance = self._calculate_walk_forward_performance(window_results, metric)
            
            # 保存结果
            wfo_id = await self._save_walk_forward_results(strategy_class.__name__, param_grid, window_results, overall_performance, method, metric)
            
            return {
                "wfo_id": wfo_id,
                "strategy_name": strategy_class.__name__,
                "method": method,
                "metric": metric,
                "window_size": window_size,
                "step_size": step_size,
                "window_results": window_results,
                "overall_performance": overall_performance
            }
            
        except Exception as e:
            self.logger.error(f"滚动优化失败: {e}")
            return {"error": str(e)}
    
    def _calculate_walk_forward_performance(self, window_results: List[Dict], metric: str) -> Dict:
        """
        计算滚动优化的整体性能
        
        Args:
            window_results: 窗口结果列表
            metric: 优化指标
            
        Returns:
            整体性能字典
        """
        # 提取测试分数
        test_scores = [result["test_score"] for result in window_results]
        
        # 计算统计量
        mean_score = np.mean(test_scores)
        median_score = np.median(test_scores)
        std_score = np.std(test_scores)
        min_score = np.min(test_scores)
        max_score = np.max(test_scores)
        
        # 计算稳定性（变异系数）
        stability = std_score / abs(mean_score) if mean_score != 0 else float('inf')
        
        # 计算过拟合程度（训练分数与测试分数的比率）
        train_scores = [result["train_score"] for result in window_results]
        mean_train_score = np.mean(train_scores)
        overfitting_ratio = mean_train_score / mean_score if mean_score != 0 else float('inf')
        
        return {
            "mean_score": mean_score,
            "median_score": median_score,
            "std_score": std_score,
            "min_score": min_score,
            "max_score": max_score,
            "stability": stability,
            "overfitting_ratio": overfitting_ratio
        }
    
    async def _save_walk_forward_results(self, strategy_name: str, param_grid: Dict[str, List], 
                                       window_results: List[Dict], overall_performance: Dict, 
                                       method: str, metric: str) -> str:
        """
        保存滚动优化结果
        
        Args:
            strategy_name: 策略名称
            param_grid: 参数网格
            window_results: 窗口结果列表
            overall_performance: 整体性能
            method: 优化方法
            metric: 优化指标
            
        Returns:
            结果ID
        """
        # 生成结果ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wfo_id = f"{strategy_name}_wfo_{method}_{metric}_{timestamp}"
        
        # 创建结果目录
        result_dir = self.results_dir / wfo_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存参数网格
        with open(result_dir / "param_grid.json", "w") as f:
            json.dump(param_grid, f, indent=4)
        
        # 保存窗口结果
        window_results_df = pd.DataFrame([
            {
                "window_index": result["window_index"],
                "train_start": result["train_start"],
                "train_end": result["train_end"],
                "test_start": result["test_start"],
                "test_end": result["test_end"],
                "best_params": json.dumps(result["best_params"]),
                "train_score": result["train_score"],
                "test_score": result["test_score"]
            }
            for result in window_results
        ])
        window_results_df.to_csv(result_dir / "window_results.csv", index=False)
        
        # 保存整体性能
        with open(result_dir / "overall_performance.json", "w") as f:
            json.dump(overall_performance, f, indent=4)
        
        # 生成报告
        await self._generate_walk_forward_report(wfo_id, strategy_name, param_grid, window_results, overall_performance, method, metric)
        
        self.logger.info(f"滚动优化结果已保存: {wfo_id}")
        return wfo_id
    
    async def _generate_walk_forward_report(self, wfo_id: str, strategy_name: str, 
                                          param_grid: Dict[str, List], window_results: List[Dict], 
                                          overall_performance: Dict, method: str, metric: str) -> str:
        """
        生成滚动优化报告
        
        Args:
            wfo_id: 结果ID
            strategy_name: 策略名称
            param_grid: 参数网格
            window_results: 窗口结果列表
            overall_performance: 整体性能
            method: 优化方法
            metric: 优化指标
            
        Returns:
            报告文件路径
        """
        try:
            # 获取结果目录
            result_dir = self.results_dir / wfo_id
            
            # 创建图表目录
            charts_dir = result_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            # 创建报告
            report_path = result_dir / "walk_forward_report.html"
            
            # 生成HTML报告
            with open(report_path, "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>滚动优化报告 - {wfo_id}</title>
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
                    <h1>滚动优化报告</h1>
                    <p>优化ID: {wfo_id}</p>
                    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <h2>优化配置</h2>
                    <table>
                        <tr>
                            <th>配置项</th>
                            <th>值</th>
                        </tr>
                        <tr>
                            <td>策略名称</td>
                            <td>{strategy_name}</td>
                        </tr>
                        <tr>
                            <td>优化方法</td>
                            <td>{method}</td>
                        </tr>
                        <tr>
                            <td>优化指标</td>
                            <td>{metric}</td>
                        </tr>
                        <tr>
                            <td>窗口数量</td>
                            <td>{len(window_results)}</td>
                        </tr>
                    </table>
                    
                    <h2>参数网格</h2>
                    <table>
                        <tr>
                            <th>参数</th>
                            <th>值范围</th>
                        </tr>
                """)
                
                # 添加参数网格
                for param_name, param_values in param_grid.items():
                    f.write(f"""
                        <tr>
                            <td>{param_name}</td>
                            <td>{param_values}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                    
                    <h2>整体性能</h2>
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>值</th>
                        </tr>
                """)
                
                # 添加整体性能
                for metric_name, value in overall_performance.items():
                    f.write(f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{value:.4f}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                    
                    <h2>窗口结果</h2>
                    <table>
                        <tr>
                            <th>窗口</th>
                            <th>训练开始</th>
                            <th>训练结束</th>
                            <th>测试开始</th>
                            <th>测试结束</th>
                            <th>训练分数</th>
                            <th>测试分数</th>
                            <th>最佳参数</th>
                        </tr>
                """)
                
                # 添加窗口结果
                for result in window_results:
                    f.write(f"""
                        <tr>
                            <td>{result['window_index'] + 1}</td>
                            <td>{result['train_start']}</td>
                            <td>{result['train_end']}</td>
                            <td>{result['test_start']}</td>
                            <td>{result['test_end']}</td>
                            <td>{result['train_score']:.4f}</td>
                            <td>{result['test_score']:.4f}</td>
                            <td>{json.dumps(result['best_params'])}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                    
                    <h2>参数稳定性分析</h2>
                    <p>下面的图表展示了不同窗口中最佳参数的变化。</p>
                    <img src="charts/parameter_stability.png" alt="参数稳定性" style="width: 100%; max-width: 800px;">
                    
                    <h2>性能分析</h2>
                    <p>下面的图表展示了训练分数和测试分数的对比。</p>
                    <img src="charts/performance_comparison.png" alt="性能对比" style="width: 100%; max-width: 800px;">
                </body>
                </html>
                """)
            
            # 生成图表
            self._generate_walk_forward_charts(window_results, param_grid, metric, charts_dir)
            
            self.logger.info(f"滚动优化报告已生成: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"生成滚动优化报告失败: {e}")
            return ""
    
    def _generate_walk_forward_charts(self, window_results: List[Dict], param_grid: Dict[str, List], 
                                     metric: str, charts_dir: Path):
        """
        生成滚动优化图表
        
        Args:
            window_results: 窗口结果列表
            param_grid: 参数网格
            metric: 优化指标
            charts_dir: 图表目录
        """
        try:
            # 创建DataFrame
            results_df = pd.DataFrame([
                {
                    "window_index": result["window_index"],
                    "train_score": result["train_score"],
                    "test_score": result["test_score"],
                    **{f"param_{param_name}": result["best_params"].get(param_name, None) for param_name in param_grid.keys()}
                }
                for result in window_results
            ])
            
            # 绘制参数稳定性图
            plt.figure(figsize=(12, 6))
            for param_name in param_grid.keys():
                plt.plot(results_df["window_index"], results_df[f"param_{param_name}"], marker='o', label=param_name)
            plt.title("参数稳定性")
            plt.xlabel("窗口索引")
            plt.ylabel("参数值")
            plt.legend()
            plt.grid(True)
            plt.savefig(charts_dir / "parameter_stability.png")
            plt.close()
            
            # 绘制性能对比图
            plt.figure(figsize=(12, 6))
            plt.plot(results_df["window_index"], results_df["train_score"], marker='o', label="训练分数")
            plt.plot(results_df["window_index"], results_df["test_score"], marker='x', label="测试分数")
            plt.title("训练分数与测试分数对比")
            plt.xlabel("窗口索引")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(charts_dir / "performance_comparison.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"生成滚动优化图表失败: {e}")
    
    async def close(self):
        """
        关闭策略优化器
        """
        await self.storage.close()
        await self.processor.close()
        await self.backtest_engine.close()
        self.logger.info("策略优化器已关闭")


async def main():
    """
    主函数
    """
    # 导入示例策略
    from ..backtesting.backtest_engine import MovingAverageCrossoverStrategy, RSIStrategy
    
    # 创建策略优化器
    optimizer = StrategyOptimizer()
    
    try:
        # 加载数据
        engine = BacktestEngine()
        df = await engine.load_data(
            symbol="BTCUSDT",
            interval="1",
            start_date="2024-01-01",
            end_date="2024-04-01",
            indicators=["sma", "ema", "macd", "rsi", "bbands", "atr"]
        )
        
        # 定义参数网格
        ma_param_grid = {
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 50]
        }
        
        rsi_param_grid = {
            "rsi_period": [7, 14, 21],
            "overbought": [65, 70, 75, 80],
            "oversold": [20, 25, 30, 35]
        }
        
        # 优化MA策略
        ma_result = await optimizer.optimize(
            strategy_class=MovingAverageCrossoverStrategy,
            param_grid=ma_param_grid,
            df=df,
            method="grid",
            metric="sharpe_ratio"
        )
        
        # 优化RSI策略
        rsi_result = await optimizer.optimize(
            strategy_class=RSIStrategy,
            param_grid=rsi_param_grid,
            df=df,
            method="random",
            metric="sharpe_ratio",
            max_iterations=50
        )
        
        # 打印结果
        print(f"MA策略最佳参数: {ma_result['best_params']}, 夏普比率: {ma_result['best_score']:.2f}")
        print(f"RSI策略最佳参数: {rsi_result['best_params']}, 夏普比率: {rsi_result['best_score']:.2f}")
        
        # 滚动优化
        wfo_result = await optimizer.walk_forward_optimization(
            strategy_class=MovingAverageCrossoverStrategy,
            param_grid=ma_param_grid,
            df=df,
            window_size=30,
            step_size=10,
            method="grid",
            metric="sharpe_ratio"
        )
        
        print(f"滚动优化完成，整体性能: {wfo_result['overall_performance']}")
        
    finally:
        # 关闭优化器
        await optimizer.close()


if __name__ == "__main__":
    asyncio.run(main())
