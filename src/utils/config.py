import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        # 加载环境变量
        env = os.getenv('ENVIRONMENT', 'development')
        env_file = f'.env.{env}'
        
        if Path(env_file).exists():
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        # 加载配置文件
        config_path = os.getenv('CONFIG_PATH', f'config/{env}/config.yaml')
        
        # 如果配置文件不存在，使用默认配置
        if not Path(config_path).exists():
            self.config = {
                'api': {
                    'base_url': 'https://api-testnet.bybit.com',
                    'ws_public_url': 'wss://stream-testnet.bybit.com/v5/public/linear',
                    'ws_private_url': 'wss://stream-testnet.bybit.com/v5/private',
                    'testnet': True
                },
                'logging': {
                    'level': 'DEBUG',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': 'logs/development.log'
                },
                'trading': {
                    'symbol': 'BTCUSDT',
                    'interval': '1'
                }
            }
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # 添加环境变量到配置
        self.config['api']['key'] = os.getenv('BYBIT_API_KEY', '')
        self.config['api']['secret'] = os.getenv('BYBIT_API_SECRET', '')
    
    def get(self, section, key=None):
        if key is None:
            return self.config.get(section, {})
        
        return self.config.get(section, {}).get(key)
