import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collection.websocket_client import BybitWebSocketClient
from src.data_collection.historical_data_downloader import BybitHistoricalDataDownloader
from src.storage.data_storage import DataStorage
from src.utils.logger import Logger

async def test_websocket_client():
    """
    测试WebSocket客户端
    """
    logger = Logger.get_logger("test_websocket")
    logger.info("开始测试WebSocket客户端")
    
    # 创建WebSocket客户端
    client = BybitWebSocketClient(
        symbol="BTCUSDT",
        channel="kline",
        interval="1"
    )
    
    # 创建数据存储
    storage = DataStorage()
    
    # 定义消息处理函数
    async def handle_message(message):
        try:
            logger.info(f"收到消息: {message[:100]}...")
            
            # 解析消息
            data = json.loads(message)
            
            # 检查消息类型
            if "topic" in data and "data" in data:
                topic = data["topic"]
                
                if "kline" in topic:
                    # 处理K线数据
                    kline_data = data["data"]
                    logger.info(f"收到K线数据: {len(kline_data)} 条")
                    
                    # 存储数据
                    await storage.store_kline_data(kline_data)
                    
                    # 打印第一条数据
                    if kline_data:
                        logger.info(f"K线数据示例: {kline_data[0]}")
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
    
    # 设置消息处理函数
    client.set_message_handler(handle_message)
    
    try:
        # 连接WebSocket
        await client.connect()
        
        # 等待接收数据
        logger.info("等待接收数据，将在30秒后断开连接...")
        await asyncio.sleep(30)
        
    finally:
        # 断开连接
        await client.disconnect()
        await storage.close()
        
    logger.info("WebSocket客户端测试完成")

async def test_historical_data_downloader():
    """
    测试历史数据下载器
    """
    logger = Logger.get_logger("test_downloader")
    logger.info("开始测试历史数据下载器")
    
    # 设置时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # 下载最近7天的数据
    
    # 创建下载器
    downloader = BybitHistoricalDataDownloader(
        symbol="BTCUSDT",
        interval="1",
        start_time=start_date.strftime("%Y-%m-%d"),
        end_time=end_date.strftime("%Y-%m-%d")
    )
    
    try:
        # 下载数据
        logger.info(f"开始下载历史数据: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        df = await downloader.download_all()
        
        # 检查数据
        if df.empty:
            logger.error("未获取到数据")
            return
        
        logger.info(f"成功下载 {len(df)} 条数据")
        logger.info(f"数据时间范围: {df['datetime'].min()} - {df['datetime'].max()}")
        
        # 显示数据样本
        logger.info(f"数据样本:\n{df.head()}")
        
        # 保存数据
        file_path = await downloader.save_to_csv(df)
        logger.info(f"数据已保存至: {file_path}")
        
        # 创建数据存储
        storage = DataStorage()
        
        # 存储数据到数据库
        logger.info("将数据存储到数据库...")
        success = await storage.store_kline_data(df)
        
        if success:
            logger.info("数据成功存储到数据库")
        else:
            logger.error("存储数据到数据库失败")
        
        # 从数据库查询数据
        logger.info("从数据库查询数据...")
        query_df = await storage.query_kline_data(
            symbol="BTCUSDT",
            interval="1",
            start_time=start_date.strftime("%Y-%m-%d"),
            end_time=end_date.strftime("%Y-%m-%d")
        )
        
        if query_df.empty:
            logger.error("从数据库查询数据失败")
        else:
            logger.info(f"从数据库查询到 {len(query_df)} 条数据")
            logger.info(f"查询数据时间范围: {query_df['datetime'].min()} - {query_df['datetime'].max()}")
        
        await storage.close()
        
    except Exception as e:
        logger.error(f"测试历史数据下载器时出错: {e}")
    
    logger.info("历史数据下载器测试完成")

async def test_data_collection_system():
    """
    测试数据收集系统
    """
    logger = Logger.get_logger("test_system")
    logger.info("开始测试数据收集系统")
    
    # 测试历史数据下载器
    logger.info("=== 测试历史数据下载器 ===")
    await test_historical_data_downloader()
    
    # 测试WebSocket客户端
    logger.info("=== 测试WebSocket客户端 ===")
    await test_websocket_client()
    
    logger.info("数据收集系统测试完成")

if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    asyncio.run(test_data_collection_system())
