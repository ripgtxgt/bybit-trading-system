import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import mysql.connector
from mysql.connector import pooling

from ..utils.logger import Logger
from ..utils.config import Config

class DataStorage:
    """
    数据存储解决方案，支持InfluxDB和MySQL
    """
    def __init__(self):
        """
        初始化数据存储
        """
        self.logger = Logger.get_logger("data_storage")
        self.config = Config()
        
        # 初始化InfluxDB客户端
        self._init_influxdb()
        
        # 初始化MySQL连接池
        self._init_mysql()
    
    def _init_influxdb(self):
        """
        初始化InfluxDB客户端
        """
        try:
            # 获取InfluxDB配置
            influxdb_config = self.config.get('database', 'influxdb')
            
            if not influxdb_config:
                self.logger.warning("未找到InfluxDB配置，使用默认配置")
                influxdb_config = {
                    "url": "http://localhost:8086",
                    "token": "my-super-secret-auth-token",
                    "org": "bybit_trading",
                    "bucket": "market_data"
                }
            
            self.influxdb_url = influxdb_config.get("url", "http://localhost:8086")
            self.influxdb_token = influxdb_config.get("token", "my-super-secret-auth-token")
            self.influxdb_org = influxdb_config.get("org", "bybit_trading")
            self.influxdb_bucket = influxdb_config.get("bucket", "market_data")
            
            # 创建InfluxDB客户端
            self.influxdb_client = InfluxDBClient(
                url=self.influxdb_url,
                token=self.influxdb_token,
                org=self.influxdb_org
            )
            
            # 创建写入API
            self.influxdb_write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
            
            # 创建查询API
            self.influxdb_query_api = self.influxdb_client.query_api()
            
            self.logger.info("InfluxDB客户端初始化成功")
            
        except Exception as e:
            self.logger.error(f"初始化InfluxDB客户端失败: {e}")
            self.influxdb_client = None
            self.influxdb_write_api = None
            self.influxdb_query_api = None
    
    def _init_mysql(self):
        """
        初始化MySQL连接池
        """
        try:
            # 获取MySQL配置
            mysql_config = self.config.get('database', 'mysql')
            
            if not mysql_config:
                self.logger.warning("未找到MySQL配置，使用默认配置")
                mysql_config = {
                    "host": "localhost",
                    "port": 3306,
                    "user": "bybit",
                    "password": "bybitpassword",
                    "database": "bybit_trading"
                }
            
            self.mysql_host = mysql_config.get("host", "localhost")
            self.mysql_port = mysql_config.get("port", 3306)
            self.mysql_user = mysql_config.get("user", "bybit")
            self.mysql_password = mysql_config.get("password", "bybitpassword")
            self.mysql_database = mysql_config.get("database", "bybit_trading")
            
            # 创建MySQL连接池
            self.mysql_pool = pooling.MySQLConnectionPool(
                pool_name="bybit_pool",
                pool_size=5,
                host=self.mysql_host,
                port=self.mysql_port,
                user=self.mysql_user,
                password=self.mysql_password,
                database=self.mysql_database
            )
            
            # 创建必要的表
            self._create_mysql_tables()
            
            self.logger.info("MySQL连接池初始化成功")
            
        except Exception as e:
            self.logger.error(f"初始化MySQL连接池失败: {e}")
            self.mysql_pool = None
    
    def _create_mysql_tables(self):
        """
        创建必要的MySQL表
        """
        if not self.mysql_pool:
            self.logger.error("MySQL连接池未初始化，无法创建表")
            return
        
        try:
            connection = self.mysql_pool.get_connection()
            cursor = connection.cursor()
            
            # 创建交易记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_records (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    order_type VARCHAR(20) NOT NULL,
                    price DECIMAL(20, 8) NOT NULL,
                    quantity DECIMAL(20, 8) NOT NULL,
                    order_id VARCHAR(50) NOT NULL,
                    trade_id VARCHAR(50) NOT NULL,
                    trade_time DATETIME NOT NULL,
                    commission DECIMAL(20, 8) NOT NULL,
                    realized_pnl DECIMAL(20, 8) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_symbol (symbol),
                    INDEX idx_trade_time (trade_time)
                )
            """)
            
            # 创建策略参数表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_params (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    strategy_name VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    params JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY idx_strategy_symbol (strategy_name, symbol)
                )
            """)
            
            # 创建回测结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    strategy_name VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    initial_capital DECIMAL(20, 8) NOT NULL,
                    final_capital DECIMAL(20, 8) NOT NULL,
                    total_return DECIMAL(10, 4) NOT NULL,
                    annual_return DECIMAL(10, 4) NOT NULL,
                    sharpe_ratio DECIMAL(10, 4) NOT NULL,
                    max_drawdown DECIMAL(10, 4) NOT NULL,
                    win_rate DECIMAL(10, 4) NOT NULL,
                    profit_factor DECIMAL(10, 4) NOT NULL,
                    params JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_strategy_symbol (strategy_name, symbol),
                    INDEX idx_created_at (created_at)
                )
            """)
            
            connection.commit()
            self.logger.info("MySQL表创建成功")
            
        except Exception as e:
            self.logger.error(f"创建MySQL表失败: {e}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
    
    async def store_kline_data(self, data: Union[Dict, List[Dict], pd.DataFrame], measurement: str = "kline") -> bool:
        """
        存储K线数据到InfluxDB
        
        Args:
            data: K线数据，可以是字典、字典列表或DataFrame
            measurement: 测量名称，默认为"kline"
            
        Returns:
            是否成功存储
        """
        if not self.influxdb_write_api:
            self.logger.error("InfluxDB写入API未初始化，无法存储数据")
            return False
        
        try:
            points = []
            
            # 处理不同类型的输入数据
            if isinstance(data, dict):
                # 单个K线数据字典
                points.append(self._create_kline_point(data, measurement))
            elif isinstance(data, list):
                # K线数据字典列表
                for item in data:
                    points.append(self._create_kline_point(item, measurement))
            elif isinstance(data, pd.DataFrame):
                # DataFrame
                for _, row in data.iterrows():
                    point_data = row.to_dict()
                    points.append(self._create_kline_point(point_data, measurement))
            else:
                self.logger.error(f"不支持的数据类型: {type(data)}")
                return False
            
            # 写入数据
            self.influxdb_write_api.write(
                bucket=self.influxdb_bucket,
                org=self.influxdb_org,
                record=points
            )
            
            self.logger.info(f"成功存储 {len(points)} 条K线数据到InfluxDB")
            return True
            
        except Exception as e:
            self.logger.error(f"存储K线数据到InfluxDB失败: {e}")
            return False
    
    def _create_kline_point(self, data: Dict, measurement: str) -> Point:
        """
        创建K线数据点
        
        Args:
            data: K线数据字典
            measurement: 测量名称
            
        Returns:
            InfluxDB数据点
        """
        # 获取时间戳
        if "start_time" in data:
            timestamp = int(data["start_time"])
        elif "timestamp" in data:
            timestamp = int(data["timestamp"])
        elif "datetime" in data:
            if isinstance(data["datetime"], str):
                timestamp = int(datetime.fromisoformat(data["datetime"].replace('Z', '+00:00')).timestamp() * 1000)
            else:
                timestamp = int(data["datetime"].timestamp() * 1000)
        else:
            timestamp = int(time.time() * 1000)
        
        # 获取交易对
        symbol = data.get("symbol", "BTCUSDT")
        
        # 创建数据点
        point = Point(measurement)
        
        # 添加标签
        point.tag("symbol", symbol)
        
        # 添加字段
        if "open" in data:
            point.field("open", float(data["open"]))
        if "high" in data:
            point.field("high", float(data["high"]))
        if "low" in data:
            point.field("low", float(data["low"]))
        if "close" in data:
            point.field("close", float(data["close"]))
        if "volume" in data:
            point.field("volume", float(data["volume"]))
        if "turnover" in data:
            point.field("turnover", float(data["turnover"]))
        if "interval" in data:
            point.tag("interval", data["interval"])
        if "confirm" in data:
            point.field("confirm", bool(data["confirm"]))
        
        # 设置时间戳
        point.time(timestamp, WritePrecision.MS)
        
        return point
    
    async def query_kline_data(self, symbol: str, interval: str = "1", start_time: str = None, end_time: str = None) -> pd.DataFrame:
        """
        从InfluxDB查询K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线时间间隔
            start_time: 开始时间，格式为 "YYYY-MM-DD HH:MM:SS"
            end_time: 结束时间，格式为 "YYYY-MM-DD HH:MM:SS"
            
        Returns:
            包含K线数据的DataFrame
        """
        if not self.influxdb_query_api:
            self.logger.error("InfluxDB查询API未初始化，无法查询数据")
            return pd.DataFrame()
        
        try:
            # 构建Flux查询
            query = f'''
                from(bucket: "{self.influxdb_bucket}")
                |> range(start: {start_time if start_time else "-30d"}, stop: {end_time if end_time else "now()"})
                |> filter(fn: (r) => r._measurement == "kline")
                |> filter(fn: (r) => r.symbol == "{symbol}")
            '''
            
            if interval:
                query += f'|> filter(fn: (r) => r.interval == "{interval}")'
            
            query += '''
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''
            
            # 执行查询
            tables = self.influxdb_query_api.query(query, org=self.influxdb_org)
            
            # 将结果转换为DataFrame
            if not tables:
                self.logger.warning(f"未查询到数据: {symbol}, {interval}, {start_time}, {end_time}")
                return pd.DataFrame()
            
            records = []
            for table in tables:
                for record in table.records:
                    records.append({
                        "time": record.get_time(),
                        "symbol": record.values.get("symbol"),
                        "interval": record.values.get("interval"),
                        "open": record.values.get("open"),
                        "high": record.values.get("high"),
                        "low": record.values.get("low"),
                        "close": record.values.get("close"),
                        "volume": record.values.get("volume"),
                        "turnover": record.values.get("turnover"),
                        "confirm": record.values.get("confirm")
                    })
            
            df = pd.DataFrame(records)
            
            if not df.empty:
                # 转换时间列
                df["datetime"] = pd.to_datetime(df["time"])
                df["timestamp"] = df["datetime"].astype(int) // 10**9 * 1000  # 转换为毫秒时间戳
                
                # 排序
                df = df.sort_values("datetime")
            
            self.logger.info(f"成功查询到 {len(df)} 条K线数据")
            return df
            
        except Exception as e:
            self.logger.error(f"查询K线数据失败: {e}")
            return pd.DataFrame()
    
    async def store_trade_record(self, trade: Dict) -> bool:
        """
        存储交易记录到MySQL
        
        Args:
            trade: 交易记录字典
            
        Returns:
            是否成功存储
        """
        if not self.mysql_pool:
            self.logger.error("MySQL连接池未初始化，无法存储交易记录")
            return False
        
        try:
            connection = self.mysql_pool.get_connection()
            cursor = connection.cursor()
            
            # 插入交易记录
            query = """
                INSERT INTO trade_records 
                (symbol, side, order_type, price, quantity, order_id, trade_id, trade_time, commission, realized_pnl)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                trade.get("symbol"),
                trade.get("side"),
                trade.get("order_type"),
                trade.get("price"),
                trade.get("quantity"),
                trade.get("order_id"),
                trade.get("trade_id"),
                trade.get("trade_time"),
                trade.get("commission"),
                trade.get("realized_pnl")
            )
            
            cursor.execute(query, values)
            connection.commit()
            
            self.logger.info(f"成功存储交易记录: {trade.get('trade_id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"存储交易记录失败: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
    
    async def query_trade_records(self, symbol: str = None, start_time: str = None, end_time: str = None) -> pd.DataFrame:
        """
        从MySQL查询交易记录
        
        Args:
            symbol: 交易对符号
            start_time: 开始时间，格式为 "YYYY-MM-DD HH:MM:SS"
            end_time: 结束时间，格式为 "YYYY-MM-DD HH:MM:SS"
            
        Returns:
            包含交易记录的DataFrame
        """
        if not self.mysql_pool:
            self.logger.error("MySQL连接池未初始化，无法查询交易记录")
            return pd.DataFrame()
        
        try:
            connection = self.mysql_pool.get_connection()
            
            # 构建查询
            query = "SELECT * FROM trade_records WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            if start_time:
                query += " AND trade_time >= %s"
                params.append(start_time)
            
            if end_time:
                query += " AND trade_time <= %s"
                params.append(end_time)
            
            query += " ORDER BY trade_time"
            
            # 执行查询
            df = pd.read_sql(query, connection, params=params)
            
            self.logger.info(f"成功查询到 {len(df)} 条交易记录")
            return df
            
        except Exception as e:
            self.logger.error(f"查询交易记录失败: {e}")
            return pd.DataFrame()
        finally:
            if 'connection' in locals():
                connection.close()
    
    async def store_strategy_params(self, strategy_name: str, symbol: str, params: Dict) -> bool:
        """
        存储策略参数到MySQL
        
        Args:
            strategy_name: 策略名称
            symbol: 交易对符号
            params: 策略参数字典
            
        Returns:
            是否成功存储
        """
        if not self.mysql_pool:
            self.logger.error("MySQL连接池未初始化，无法存储策略参数")
            return False
        
        try:
            connection = self.mysql_pool.get_connection()
            cursor = connection.cursor()
            
            # 将参数转换为JSON字符串
            params_json = json.dumps(params)
            
            # 插入或更新策略参数
            query = """
                INSERT INTO strategy_params 
                (strategy_name, symbol, params)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                params = %s,
                updated_at = CURRENT_TIMESTAMP
            """
            
            values = (strategy_name, symbol, params_json, params_json)
            
            cursor.execute(query, values)
            connection.commit()
            
            self.logger.info(f"成功存储策略参数: {strategy_name}, {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"存储策略参数失败: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
    
    async def query_strategy_params(self, strategy_name: str = None, symbol: str = None) -> List[Dict]:
        """
        从MySQL查询策略参数
        
        Args:
            strategy_name: 策略名称
            symbol: 交易对符号
            
        Returns:
            包含策略参数的字典列表
        """
        if not self.mysql_pool:
            self.logger.error("MySQL连接池未初始化，无法查询策略参数")
            return []
        
        try:
            connection = self.mysql_pool.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # 构建查询
            query = "SELECT * FROM strategy_params WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = %s"
                params.append(strategy_name)
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            # 执行查询
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # 解析JSON参数
            for result in results:
                if "params" in result and result["params"]:
                    result["params"] = json.loads(result["params"])
            
            self.logger.info(f"成功查询到 {len(results)} 条策略参数记录")
            return results
            
        except Exception as e:
            self.logger.error(f"查询策略参数失败: {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
    
    async def store_backtest_result(self, result: Dict) -> bool:
        """
        存储回测结果到MySQL
        
        Args:
            result: 回测结果字典
            
        Returns:
            是否成功存储
        """
        if not self.mysql_pool:
            self.logger.error("MySQL连接池未初始化，无法存储回测结果")
            return False
        
        try:
            connection = self.mysql_pool.get_connection()
            cursor = connection.cursor()
            
            # 将参数转换为JSON字符串
            params_json = json.dumps(result.get("params", {}))
            
            # 插入回测结果
            query = """
                INSERT INTO backtest_results 
                (strategy_name, symbol, start_time, end_time, initial_capital, final_capital, 
                total_return, annual_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, params)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                result.get("strategy_name"),
                result.get("symbol"),
                result.get("start_time"),
                result.get("end_time"),
                result.get("initial_capital"),
                result.get("final_capital"),
                result.get("total_return"),
                result.get("annual_return"),
                result.get("sharpe_ratio"),
                result.get("max_drawdown"),
                result.get("win_rate"),
                result.get("profit_factor"),
                params_json
            )
            
            cursor.execute(query, values)
            connection.commit()
            
            self.logger.info(f"成功存储回测结果: {result.get('strategy_name')}, {result.get('symbol')}")
            return True
            
        except Exception as e:
            self.logger.error(f"存储回测结果失败: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
    
    async def query_backtest_results(self, strategy_name: str = None, symbol: str = None) -> pd.DataFrame:
        """
        从MySQL查询回测结果
        
        Args:
            strategy_name: 策略名称
            symbol: 交易对符号
            
        Returns:
            包含回测结果的DataFrame
        """
        if not self.mysql_pool:
            self.logger.error("MySQL连接池未初始化，无法查询回测结果")
            return pd.DataFrame()
        
        try:
            connection = self.mysql_pool.get_connection()
            
            # 构建查询
            query = "SELECT * FROM backtest_results WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = %s"
                params.append(strategy_name)
            
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            
            query += " ORDER BY created_at DESC"
            
            # 执行查询
            df = pd.read_sql(query, connection, params=params)
            
            # 解析JSON参数
            if not df.empty and "params" in df.columns:
                df["params"] = df["params"].apply(json.loads)
            
            self.logger.info(f"成功查询到 {len(df)} 条回测结果")
            return df
            
        except Exception as e:
            self.logger.error(f"查询回测结果失败: {e}")
            return pd.DataFrame()
        finally:
            if 'connection' in locals():
                connection.close()
    
    async def close(self):
        """
        关闭数据库连接
        """
        if self.influxdb_client:
            self.influxdb_client.close()
            self.logger.info("InfluxDB客户端已关闭")
        
        # MySQL连接池会自动管理连接的关闭
        self.logger.info("数据存储已关闭")


async def main():
    """
    主函数
    """
    # 创建数据存储
    storage = DataStorage()
    
    # 测试存储K线数据
    test_data = {
        "symbol": "BTCUSDT",
        "interval": "1",
        "start_time": int(time.time() * 1000) - 60000,  # 1分钟前
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 10.5,
        "turnover": 525000.0,
        "confirm": True
    }
    
    success = await storage.store_kline_data(test_data)
    print(f"存储K线数据: {'成功' if success else '失败'}")
    
    # 测试查询K线数据
    df = await storage.query_kline_data("BTCUSDT", "1", start_time="-1h")
    print(f"查询到 {len(df)} 条K线数据")
    if not df.empty:
        print(df.head())
    
    # 关闭数据存储
    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
