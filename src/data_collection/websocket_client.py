import asyncio
import json
import time
import websockets
import hmac
import hashlib
from queue import Queue
from typing import Dict, List, Optional, Callable, Any

from ..utils.logger import Logger
from ..utils.config import Config

class BybitWebSocketClient:
    """
    Bybit WebSocket客户端，用于订阅实时K线数据
    """
    def __init__(self, symbol: str, interval: str = "1"):
        """
        初始化WebSocket客户端
        
        Args:
            symbol: 交易对符号，例如 "BTCUSDT"
            interval: K线时间间隔，默认为1分钟
        """
        self.logger = Logger.get_logger("bybit_websocket")
        self.config = Config()
        
        self.symbol = symbol.upper()
        self.interval = interval
        self.ws_url = self.config.get('api', 'ws_public_url')
        self.api_key = self.config.get('api', 'key')
        self.api_secret = self.config.get('api', 'secret')
        
        self.ws = None
        self.connected = False
        self.reconnect_delay = 5  # 重连延迟，单位秒
        self.max_reconnect_delay = 60  # 最大重连延迟，单位秒
        self.heartbeat_interval = 20  # 心跳包发送间隔，单位秒
        
        self.data_queue = Queue()
        self.callbacks = {}  # 回调函数字典
        self.last_received_time = 0
        
    async def connect(self):
        """
        建立WebSocket连接
        """
        self.logger.info(f"正在连接到Bybit WebSocket: {self.ws_url}")
        
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.connected = True
            self.last_received_time = time.time()
            self.logger.info("WebSocket连接成功")
            
            # 启动心跳任务
            asyncio.create_task(self.heartbeat_loop())
            
            # 订阅K线数据
            await self.subscribe_kline()
            
            # 开始消息处理循环
            await self.message_loop()
            
        except Exception as e:
            self.logger.error(f"WebSocket连接失败: {e}")
            self.connected = False
            await self.reconnect()
    
    async def reconnect(self):
        """
        断线重连逻辑
        """
        self.connected = False
        
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            
        self.logger.info(f"将在 {self.reconnect_delay} 秒后尝试重新连接...")
        await asyncio.sleep(self.reconnect_delay)
        
        # 指数退避策略
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        
        await self.connect()
    
    async def heartbeat_loop(self):
        """
        定期发送心跳包
        """
        while self.connected:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if self.connected:
                    await self.send_heartbeat()
                    
                    # 检查最后接收消息的时间，如果超过2倍心跳间隔没有收到消息，则重连
                    if time.time() - self.last_received_time > 2 * self.heartbeat_interval:
                        self.logger.warning("长时间未收到消息，正在重新连接...")
                        await self.reconnect()
                        break
            except Exception as e:
                self.logger.error(f"心跳循环异常: {e}")
                if self.connected:
                    await self.reconnect()
                break
    
    async def send_heartbeat(self):
        """
        发送心跳包
        """
        if not self.connected:
            return
            
        try:
            ping_message = {"op": "ping"}
            await self.ws.send(json.dumps(ping_message))
            self.logger.debug("已发送心跳包")
        except Exception as e:
            self.logger.error(f"发送心跳包失败: {e}")
            await self.reconnect()
    
    async def subscribe_kline(self):
        """
        订阅K线数据
        """
        if not self.connected:
            return
            
        try:
            topic = f"kline.{self.interval}.{self.symbol}"
            subscribe_message = {
                "op": "subscribe",
                "args": [topic]
            }
            
            await self.ws.send(json.dumps(subscribe_message))
            self.logger.info(f"已订阅主题: {topic}")
        except Exception as e:
            self.logger.error(f"订阅K线数据失败: {e}")
            await self.reconnect()
    
    async def unsubscribe_kline(self):
        """
        取消订阅K线数据
        """
        if not self.connected:
            return
            
        try:
            topic = f"kline.{self.interval}.{self.symbol}"
            unsubscribe_message = {
                "op": "unsubscribe",
                "args": [topic]
            }
            
            await self.ws.send(json.dumps(unsubscribe_message))
            self.logger.info(f"已取消订阅主题: {topic}")
        except Exception as e:
            self.logger.error(f"取消订阅K线数据失败: {e}")
    
    async def message_loop(self):
        """
        消息处理循环
        """
        while self.connected:
            try:
                message = await self.ws.recv()
                self.last_received_time = time.time()
                
                # 处理接收到的消息
                await self.handle_message(message)
                
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"WebSocket连接已关闭: {e}")
                await self.reconnect()
                break
            except Exception as e:
                self.logger.error(f"处理消息时出错: {e}")
                if self.connected:
                    await self.reconnect()
                break
    
    async def handle_message(self, message: str):
        """
        处理接收到的WebSocket消息
        
        Args:
            message: 接收到的WebSocket消息
        """
        try:
            data = json.loads(message)
            
            # 处理心跳响应
            if "op" in data and data["op"] == "pong":
                self.logger.debug("收到心跳响应")
                return
                
            # 处理订阅响应
            if "op" in data and data["op"] == "subscribe":
                self.logger.info(f"订阅成功: {data}")
                return
                
            # 处理K线数据
            if "topic" in data and data["topic"].startswith("kline."):
                self.logger.debug(f"收到K线数据: {data}")
                
                # 将数据放入队列
                self.data_queue.put(data)
                
                # 调用回调函数
                topic = data["topic"]
                if topic in self.callbacks:
                    for callback in self.callbacks[topic]:
                        asyncio.create_task(callback(data))
                
                return
                
            # 处理其他消息
            self.logger.debug(f"收到其他消息: {data}")
            
        except json.JSONDecodeError:
            self.logger.error(f"解析消息失败: {message}")
        except Exception as e:
            self.logger.error(f"处理消息时出错: {e}")
    
    def register_callback(self, topic: str, callback: Callable[[Dict], Any]):
        """
        注册回调函数
        
        Args:
            topic: 主题名称
            callback: 回调函数
        """
        if topic not in self.callbacks:
            self.callbacks[topic] = []
        
        self.callbacks[topic].append(callback)
        self.logger.info(f"已注册回调函数到主题: {topic}")
    
    def get_latest_data(self):
        """
        获取最新的K线数据
        
        Returns:
            最新的K线数据，如果队列为空则返回None
        """
        if self.data_queue.empty():
            return None
        
        return self.data_queue.get()
    
    async def close(self):
        """
        关闭WebSocket连接
        """
        self.logger.info("正在关闭WebSocket连接...")
        self.connected = False
        
        if self.ws:
            try:
                await self.unsubscribe_kline()
                await self.ws.close()
                self.logger.info("WebSocket连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭WebSocket连接时出错: {e}")
    
    @staticmethod
    async def process_kline_data(data: Dict) -> Dict:
        """
        处理K线数据
        
        Args:
            data: 原始K线数据
            
        Returns:
            处理后的K线数据
        """
        if "data" not in data:
            return {}
            
        kline_data = data["data"]
        
        processed_data = {
            "symbol": data.get("topic", "").split(".")[-1],
            "interval": data.get("topic", "").split(".")[1],
            "timestamp": kline_data.get("timestamp", 0),
            "open": float(kline_data.get("open", 0)),
            "high": float(kline_data.get("high", 0)),
            "low": float(kline_data.get("low", 0)),
            "close": float(kline_data.get("close", 0)),
            "volume": float(kline_data.get("volume", 0)),
            "turnover": float(kline_data.get("turnover", 0)),
            "confirm": kline_data.get("confirm", False),
            "start_time": kline_data.get("start", 0),
            "end_time": kline_data.get("end", 0)
        }
        
        return processed_data


async def example_callback(data):
    """
    示例回调函数
    """
    print(f"收到K线数据回调: {data}")


async def main():
    """
    主函数
    """
    # 创建WebSocket客户端
    client = BybitWebSocketClient(symbol="BTCUSDT", interval="1")
    
    # 注册回调函数
    client.register_callback("kline.1.BTCUSDT", example_callback)
    
    # 连接WebSocket
    connection_task = asyncio.create_task(client.connect())
    
    try:
        # 运行一段时间
        await asyncio.sleep(300)  # 运行5分钟
    finally:
        # 关闭连接
        await client.close()
        await connection_task


if __name__ == "__main__":
    asyncio.run(main())
