import asyncio
import json
import time
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..utils.logger import Logger
from ..utils.config import Config

class BybitHistoricalDataDownloader:
    """
    Bybit历史数据下载器，用于获取历史K线数据
    """
    def __init__(self, symbol: str, interval: str = "1", start_time: str = None, end_time: str = None):
        """
        初始化历史数据下载器
        
        Args:
            symbol: 交易对符号，例如 "BTCUSDT"
            interval: K线时间间隔，默认为1分钟
            start_time: 开始时间，格式为 "YYYY-MM-DD"，默认为None
            end_time: 结束时间，格式为 "YYYY-MM-DD"，默认为当前时间
        """
        self.logger = Logger.get_logger("bybit_historical_downloader")
        self.config = Config()
        
        self.symbol = symbol.upper()
        self.interval = interval
        self.base_url = self.config.get('api', 'base_url')
        self.api_key = self.config.get('api', 'key')
        self.api_secret = self.config.get('api', 'secret')
        
        # 设置默认时间范围
        if start_time is None:
            self.start_time = datetime(2024, 1, 1)
        else:
            self.start_time = datetime.strptime(start_time, "%Y-%m-%d")
            
        if end_time is None:
            self.end_time = datetime.now()
        else:
            self.end_time = datetime.strptime(end_time, "%Y-%m-%d")
        
        # 确保数据目录存在
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API请求限制
        self.rate_limit = 10  # 每秒请求数
        self.last_request_time = 0
        self.max_retries = 5
        self.retry_delay = 2  # 重试延迟，单位秒
        
        # 每个请求的最大数据点数
        self.max_limit = 1000
        
    async def download_chunk(self, start_ms: int, end_ms: int, limit: int = 1000) -> List[Dict]:
        """
        下载指定时间范围的数据块
        
        Args:
            start_ms: 开始时间戳（毫秒）
            end_ms: 结束时间戳（毫秒）
            limit: 返回的数据点数量，最大为1000
            
        Returns:
            下载的数据列表
        """
        # 限速控制
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 1 / self.rate_limit:
            await asyncio.sleep(1 / self.rate_limit - time_since_last_request)
        
        self.last_request_time = time.time()
        
        # 构建请求URL
        endpoint = "/v5/market/kline"
        url = f"{self.base_url}{endpoint}"
        
        # 构建请求参数
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "interval": self.interval,
            "start": start_ms,
            "end": end_ms,
            "limit": limit
        }
        
        # 发送请求
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data["retCode"] == 0 and "result" in data and "list" in data["result"]:
                                self.logger.debug(f"成功下载数据块: {start_ms} - {end_ms}, 数据点数: {len(data['result']['list'])}")
                                return data["result"]["list"]
                            else:
                                self.logger.error(f"API返回错误: {data}")
                                retry_count += 1
                                await asyncio.sleep(self.retry_delay * (2 ** retry_count))
                        else:
                            self.logger.error(f"HTTP错误: {response.status}, {await response.text()}")
                            retry_count += 1
                            await asyncio.sleep(self.retry_delay * (2 ** retry_count))
            except Exception as e:
                self.logger.error(f"请求异常: {e}")
                retry_count += 1
                await asyncio.sleep(self.retry_delay * (2 ** retry_count))
        
        self.logger.error(f"下载数据块失败，已达到最大重试次数: {start_ms} - {end_ms}")
        return []
    
    async def download_all(self) -> pd.DataFrame:
        """
        下载所有历史数据
        
        Returns:
            包含所有历史数据的DataFrame
        """
        self.logger.info(f"开始下载 {self.symbol} 的历史数据，时间范围: {self.start_time} - {self.end_time}")
        
        # 将时间转换为毫秒时间戳
        start_ms = int(self.start_time.timestamp() * 1000)
        end_ms = int(self.end_time.timestamp() * 1000)
        
        # 计算时间块
        # Bybit API限制每次请求最多返回1000个数据点
        # 对于1分钟K线，1000个数据点约为16.7小时
        # 我们使用12小时作为一个时间块，确保不会超过限制
        time_chunks = []
        chunk_size_ms = 12 * 60 * 60 * 1000  # 12小时，单位毫秒
        
        current_start = start_ms
        while current_start < end_ms:
            current_end = min(current_start + chunk_size_ms, end_ms)
            time_chunks.append((current_start, current_end))
            current_start = current_end
        
        self.logger.info(f"将下载 {len(time_chunks)} 个时间块")
        
        # 并发下载数据
        all_data = []
        tasks = []
        
        # 限制并发数量，避免超过API限制
        semaphore = asyncio.Semaphore(5)
        
        async def download_with_semaphore(start, end):
            async with semaphore:
                return await self.download_chunk(start, end, self.max_limit)
        
        for start, end in time_chunks:
            tasks.append(download_with_semaphore(start, end))
        
        chunks_data = await asyncio.gather(*tasks)
        
        for chunk in chunks_data:
            all_data.extend(chunk)
        
        self.logger.info(f"下载完成，共获取 {len(all_data)} 条数据")
        
        # 将数据转换为DataFrame
        if not all_data:
            self.logger.warning("未获取到任何数据")
            return pd.DataFrame()
        
        # Bybit API返回的数据是按时间倒序排列的，需要反转
        all_data.reverse()
        
        # 解析数据
        # Bybit K线数据格式: [start_time, open, high, low, close, volume, turnover]
        df = pd.DataFrame(all_data, columns=[
            "start_time", "open", "high", "low", "close", "volume", "turnover"
        ])
        
        # 转换数据类型
        df["start_time"] = pd.to_numeric(df["start_time"])
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["turnover"] = pd.to_numeric(df["turnover"])
        
        # 添加日期时间列
        df["datetime"] = pd.to_datetime(df["start_time"], unit="ms")
        
        # 验证数据完整性
        self.validate_data(df)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性
        
        Args:
            df: 数据DataFrame
            
        Returns:
            数据是否完整
        """
        if df.empty:
            self.logger.warning("数据为空，无法验证")
            return False
        
        # 检查时间序列是否连续
        df = df.sort_values("start_time")
        
        # 对于1分钟K线，相邻时间戳的差值应该是60000毫秒（1分钟）
        expected_interval_ms = self.get_interval_ms(self.interval)
        
        time_diffs = df["start_time"].diff().dropna()
        
        # 允许有少量缺失（比如交易所维护期间）
        missing_intervals = time_diffs[time_diffs > expected_interval_ms]
        
        if not missing_intervals.empty:
            missing_count = len(missing_intervals)
            missing_percentage = missing_count / len(df) * 100
            
            self.logger.warning(f"检测到 {missing_count} 个缺失间隔 ({missing_percentage:.2f}%)")
            
            if missing_percentage > 5:
                self.logger.error(f"缺失数据过多，可能影响分析结果")
                return False
            
            # 记录缺失的时间段
            for i, (idx, diff) in enumerate(missing_intervals.items()):
                if i < 10:  # 只记录前10个缺失
                    start_time = df.loc[idx-1, "datetime"]
                    end_time = df.loc[idx, "datetime"]
                    self.logger.warning(f"缺失时间段: {start_time} - {end_time}")
                elif i == 10:
                    self.logger.warning("更多缺失时间段省略...")
        
        self.logger.info("数据验证完成")
        return True
    
    def get_interval_ms(self, interval: str) -> int:
        """
        获取时间间隔的毫秒数
        
        Args:
            interval: 时间间隔字符串
            
        Returns:
            时间间隔的毫秒数
        """
        if interval == "1":
            return 60 * 1000  # 1分钟
        elif interval == "3":
            return 3 * 60 * 1000  # 3分钟
        elif interval == "5":
            return 5 * 60 * 1000  # 5分钟
        elif interval == "15":
            return 15 * 60 * 1000  # 15分钟
        elif interval == "30":
            return 30 * 60 * 1000  # 30分钟
        elif interval == "60":
            return 60 * 60 * 1000  # 1小时
        elif interval == "120":
            return 2 * 60 * 60 * 1000  # 2小时
        elif interval == "240":
            return 4 * 60 * 60 * 1000  # 4小时
        elif interval == "360":
            return 6 * 60 * 60 * 1000  # 6小时
        elif interval == "720":
            return 12 * 60 * 60 * 1000  # 12小时
        elif interval == "D":
            return 24 * 60 * 60 * 1000  # 1天
        elif interval == "W":
            return 7 * 24 * 60 * 60 * 1000  # 1周
        elif interval == "M":
            return 30 * 24 * 60 * 60 * 1000  # 1月（近似值）
        else:
            self.logger.error(f"未知的时间间隔: {interval}")
            return 60 * 1000  # 默认1分钟
    
    async def save_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        将数据保存为CSV文件
        
        Args:
            df: 数据DataFrame
            filename: 文件名，默认为None（自动生成）
            
        Returns:
            保存的文件路径
        """
        if df.empty:
            self.logger.warning("数据为空，无法保存")
            return ""
        
        if filename is None:
            start_date = df["datetime"].min().strftime("%Y%m%d")
            end_date = df["datetime"].max().strftime("%Y%m%d")
            filename = f"{self.symbol}_{self.interval}_{start_date}_{end_date}.csv"
        
        file_path = self.data_dir / filename
        
        df.to_csv(file_path, index=False)
        self.logger.info(f"数据已保存至: {file_path}")
        
        return str(file_path)
    
    async def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的数据DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            
            # 转换日期时间列
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            
            self.logger.info(f"从 {file_path} 加载了 {len(df)} 条数据")
            return df
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()


async def main():
    """
    主函数
    """
    # 创建历史数据下载器
    downloader = BybitHistoricalDataDownloader(
        symbol="BTCUSDT",
        interval="1",
        start_time="2024-01-01",
        end_time=datetime.now().strftime("%Y-%m-%d")
    )
    
    # 下载数据
    df = await downloader.download_all()
    
    if not df.empty:
        # 保存数据
        file_path = await downloader.save_to_csv(df)
        
        # 显示数据统计
        print(f"数据点数: {len(df)}")
        print(f"时间范围: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"数据已保存至: {file_path}")
    else:
        print("未获取到数据")


if __name__ == "__main__":
    asyncio.run(main())
