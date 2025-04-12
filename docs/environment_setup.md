# Bybit高频交易系统环境搭建指南

## 1. 开发环境要求

### 1.1 硬件要求

#### 开发环境
- CPU: 4核心及以上
- 内存: 16GB RAM及以上
- 存储: 256GB SSD及以上
- 网络: 稳定的互联网连接

#### 生产环境
- **主交易服务器**：
  - CPU: 8核心高频处理器
  - 内存: 32GB RAM
  - 存储: 500GB SSD
  - 网络: 低延迟网络连接，靠近Bybit服务器
  - 操作系统: Ubuntu Server 22.04 LTS

- **数据库服务器**：
  - CPU: 8核心处理器
  - 内存: 64GB RAM
  - 存储: 2TB SSD (RAID配置)
  - 网络: 高带宽网络连接
  - 操作系统: Ubuntu Server 22.04 LTS

- **回测服务器**：
  - CPU: 16核心处理器
  - 内存: 128GB RAM
  - 存储: 1TB SSD
  - GPU: NVIDIA RTX 3080或更高(用于加速参数优化)
  - 操作系统: Ubuntu Server 22.04 LTS

### 1.2 软件要求

#### 操作系统
- 开发环境: Ubuntu 22.04 LTS / Windows 10 + WSL2 / macOS
- 生产环境: Ubuntu Server 22.04 LTS

#### 编程语言
- Python 3.10+

#### 数据库
- InfluxDB 2.x (时序数据库)
- MySQL 8.0+ (关系数据库)
- Redis 6.x+ (缓存和消息队列)

#### 容器化
- Docker 20.10+
- Docker Compose 2.x+

#### 监控工具
- Prometheus
- Grafana

#### 开发工具
- Visual Studio Code / PyCharm
- Git
- Jupyter Notebook

## 2. 开发环境搭建步骤

### 2.1 基础环境安装

#### Ubuntu系统安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础开发工具
sudo apt install -y build-essential git curl wget vim
```

#### Python环境安装

```bash
# 安装Python和pip
sudo apt install -y python3 python3-pip python3-dev python3-venv

# 创建并激活虚拟环境
mkdir -p ~/bybit_trading_system
cd ~/bybit_trading_system
python3 -m venv venv
source venv/bin/activate

# 升级pip
pip install --upgrade pip
```

### 2.2 安装项目依赖

创建requirements.txt文件，包含以下依赖：

```
# 网络和API
websockets==11.0.3
aiohttp==3.8.5
requests==2.31.0
ccxt==4.0.0

# 数据处理和分析
pandas==2.1.0
numpy==1.24.4
scipy==1.11.3
scikit-learn==1.3.0
ta-lib==0.4.28
matplotlib==3.7.3
seaborn==0.12.2

# 数据库
influxdb-client==1.36.1
mysql-connector-python==8.1.0
redis==4.6.0
sqlalchemy==2.0.20

# 并发和异步
asyncio==3.4.3
aioredis==2.0.1

# 监控和日志
prometheus-client==0.17.1
python-json-logger==2.0.7

# 优化
optuna==3.3.0

# 测试
pytest==7.4.0
pytest-asyncio==0.21.1

# 工具
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1
```

安装依赖：

```bash
pip install -r requirements.txt
```

### 2.3 安装TA-Lib

TA-Lib是一个技术分析库，需要单独安装：

```bash
# 安装TA-Lib依赖
sudo apt install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# 安装Python TA-Lib包
pip install TA-Lib
```

### 2.4 安装Docker和Docker Compose

```bash
# 安装Docker
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce

# 将当前用户添加到docker组
sudo usermod -aG docker $USER
newgrp docker

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2.5 安装数据库

#### 使用Docker安装InfluxDB

创建docker-compose.yml文件：

```yaml
version: '3'

services:
  influxdb:
    image: influxdb:2.7
    container_name: influxdb
    ports:
      - "8086:8086"
    volumes:
      - influxdb-data:/var/lib/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword
      - DOCKER_INFLUXDB_INIT_ORG=bybit_trading
      - DOCKER_INFLUXDB_INIT_BUCKET=market_data
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-auth-token

  mysql:
    image: mysql:8.0
    container_name: mysql
    ports:
      - "3306:3306"
    volumes:
      - mysql-data:/var/lib/mysql
    environment:
      - MYSQL_ROOT_PASSWORD=rootpassword
      - MYSQL_DATABASE=bybit_trading
      - MYSQL_USER=bybit
      - MYSQL_PASSWORD=bybitpassword

  redis:
    image: redis:6.2
    container_name: redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  grafana:
    image: grafana/grafana:9.5.2
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=adminpassword
    depends_on:
      - influxdb

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

volumes:
  influxdb-data:
  mysql-data:
  redis-data:
  grafana-data:
  prometheus-data:
```

创建prometheus.yml配置文件：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'bybit_trading_system'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

启动服务：

```bash
docker-compose up -d
```

### 2.6 项目目录结构设置

```bash
mkdir -p bybit_trading_system/{src,data,docs,tests,config,scripts,notebooks}

# 创建源代码目录结构
mkdir -p bybit_trading_system/src/{data_collection,data_processing,storage,strategy,backtesting,risk_management,trading,monitoring,utils}

# 创建配置文件目录
mkdir -p bybit_trading_system/config/{development,production}

# 创建测试目录结构
mkdir -p bybit_trading_system/tests/{unit,integration,system}

# 创建数据目录
mkdir -p bybit_trading_system/data/{raw,processed,backtest_results}
```

### 2.7 创建配置文件

#### 开发环境配置文件

创建`bybit_trading_system/config/development/config.yaml`：

```yaml
# 开发环境配置

# API配置
api:
  base_url: "https://api-testnet.bybit.com"
  ws_public_url: "wss://stream-testnet.bybit.com/v5/public/linear"
  ws_private_url: "wss://stream-testnet.bybit.com/v5/private"
  testnet: true

# 数据库配置
database:
  influxdb:
    url: "http://localhost:8086"
    token: "my-super-secret-auth-token"
    org: "bybit_trading"
    bucket: "market_data"
  
  mysql:
    host: "localhost"
    port: 3306
    user: "bybit"
    password: "bybitpassword"
    database: "bybit_trading"
  
  redis:
    host: "localhost"
    port: 6379
    db: 0

# 日志配置
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/development.log"

# 交易配置
trading:
  symbol: "BTCUSDT"
  interval: "1"  # 1分钟
  max_leverage: 10
  max_position_size: 0.1  # BTC
  max_order_size: 0.05  # BTC
  max_daily_drawdown: 0.05  # 5%
  stop_loss_pct: 0.02  # 2%
  take_profit_pct: 0.03  # 3%

# 回测配置
backtest:
  start_date: "2024-01-01"
  end_date: "2025-04-10"
  initial_capital: 10000  # USDT
  fee_rate: 0.0006  # 0.06%
  slippage: 0.0001  # 0.01%
```

#### 生产环境配置文件

创建`bybit_trading_system/config/production/config.yaml`：

```yaml
# 生产环境配置

# API配置
api:
  base_url: "https://api.bybit.com"
  ws_public_url: "wss://stream.bybit.com/v5/public/linear"
  ws_private_url: "wss://stream.bybit.com/v5/private"
  testnet: false

# 数据库配置
database:
  influxdb:
    url: "http://influxdb:8086"
    token: "production-auth-token"
    org: "bybit_trading"
    bucket: "market_data"
  
  mysql:
    host: "mysql"
    port: 3306
    user: "bybit"
    password: "strong-production-password"
    database: "bybit_trading"
  
  redis:
    host: "redis"
    port: 6379
    db: 0

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/production.log"

# 交易配置
trading:
  symbol: "BTCUSDT"
  interval: "1"  # 1分钟
  max_leverage: 5
  max_position_size: 0.1  # BTC
  max_order_size: 0.02  # BTC
  max_daily_drawdown: 0.03  # 3%
  stop_loss_pct: 0.015  # 1.5%
  take_profit_pct: 0.025  # 2.5%

# 风险控制配置
risk_management:
  max_open_orders: 5
  max_daily_trades: 50
  max_position_value: 10000  # USDT
  circuit_breaker:
    enabled: true
    price_change_threshold: 0.05  # 5%
    time_window_seconds: 300  # 5分钟
  emergency_stop:
    enabled: true
    daily_loss_threshold: 0.05  # 5%
    weekly_loss_threshold: 0.1  # 10%
```

### 2.8 创建环境变量文件

创建`.env.development`文件：

```
# 开发环境变量

# API密钥（请替换为实际的测试网API密钥）
BYBIT_API_KEY=your_testnet_api_key
BYBIT_API_SECRET=your_testnet_api_secret

# 环境设置
ENVIRONMENT=development
CONFIG_PATH=config/development/config.yaml

# 数据库凭证
INFLUXDB_TOKEN=my-super-secret-auth-token
MYSQL_PASSWORD=bybitpassword
REDIS_PASSWORD=

# 日志设置
LOG_LEVEL=DEBUG
```

创建`.env.production`文件（注意：不要将实际的生产密钥提交到版本控制系统）：

```
# 生产环境变量

# API密钥（请替换为实际的API密钥）
BYBIT_API_KEY=your_production_api_key
BYBIT_API_SECRET=your_production_api_secret

# 环境设置
ENVIRONMENT=production
CONFIG_PATH=config/production/config.yaml

# 数据库凭证
INFLUXDB_TOKEN=production-auth-token
MYSQL_PASSWORD=strong-production-password
REDIS_PASSWORD=strong-redis-password

# 日志设置
LOG_LEVEL=INFO
```

### 2.9 创建基础工具类

#### 配置加载器

创建`bybit_trading_system/src/utils/config.py`：

```python
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
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 添加环境变量到配置
        self.config['api']['key'] = os.getenv('BYBIT_API_KEY', '')
        self.config['api']['secret'] = os.getenv('BYBIT_API_SECRET', '')
    
    def get(self, section, key=None):
        if key is None:
            return self.config.get(section, {})
        
        return self.config.get(section, {}).get(key)
```

#### 日志工具

创建`bybit_trading_system/src/utils/logger.py`：

```python
import os
import logging
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from pathlib import Path

from .config import Config

class Logger:
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name):
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 创建新的logger
        logger = logging.getLogger(name)
        
        # 获取配置
        config = Config()
        log_level = os.getenv('LOG_LEVEL', config.get('logging', 'level'))
        log_format = config.get('logging', 'format')
        log_file = config.get('logging', 'file')
        
        # 设置日志级别
        logger.setLevel(getattr(logging, log_level))
        
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # 创建文件处理器
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level))
        
        # 创建JSON处理器用于结构化日志
        json_handler = RotatingFileHandler(
            log_file.replace('.log', '.json.log'), 
            maxBytes=10*1024*1024, 
            backupCount=5
        )
        json_handler.setLevel(getattr(logging, log_level))
        
        # 设置格式化器
        formatter = logging.Formatter(log_format)
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        json_handler.setFormatter(json_formatter)
        
        # 添加处理器到logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(json_handler)
        
        # 保存logger
        cls._loggers[name] = logger
        
        return logger
```

## 3. 验证环境设置

创建一个简单的测试脚本来验证环境设置：

```python
# bybit_trading_system/scripts/test_environment.py

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import Config
from src.utils.logger import Logger

def test_environment():
    # 测试配置加载
    config = Config()
    api_config = config.get('api')
    
    logger = Logger.get_logger('test')
    logger.info("环境测试开始")
    
    # 测试API配置
    logger.info(f"API基础URL: {api_config.get('base_url')}")
    logger.info(f"是否使用测试网: {api_config.get('testnet')}")
    
    # 测试数据库配置
    db_config = config.get('database')
    logger.info(f"InfluxDB URL: {db_config.get('influxdb', {}).get('url')}")
    logger.info(f"MySQL主机: {db_config.get('mysql', {}).get('host')}")
    
    # 测试Python依赖
    try:
        import pandas as pd
        import numpy as np
        import websockets
        import aiohttp
        import influxdb_client
        import mysql.connector
        
        logger.info("所有必要的Python依赖已成功导入")
    except ImportError as e:
        logger.error(f"依赖导入失败: {e}")
        return False
    
    logger.info("环境测试完成")
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
```

运行测试脚本：

```bash
python scripts/test_environment.py
```

## 4. 开发工作流程

### 4.1 代码版本控制

```bash
# 初始化Git仓库
cd bybit_trading_system
git init

# 创建.gitignore文件
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
ENV/

# 环境变量和密钥
.env*
*.key
*.pem

# 日志
logs/
*.log

# 数据文件
data/raw/
data/processed/
*.csv
*.parquet
*.feather

# IDE
.idea/
.vscode/
*.swp
*.swo

# Docker
.docker/

# 其他
.DS_Store
EOF

# 添加文件并提交
git add .
git commit -m "初始化项目结构和环境配置"
```

### 4.2 开发流程

1. 创建功能分支
   ```bash
   git checkout -b feature/websocket-client
   ```

2. 开发功能

3. 运行测试
   ```bash
   pytest tests/
   ```

4. 提交更改
   ```bash
   git add .
   git commit -m "实现WebSocket客户端"
   ```

5. 合并到主分支
   ```bash
   git checkout main
   git merge feature/websocket-client
   ```

## 5. 生产环境部署准备

### 5.1 Docker部署

创建`Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV CONFIG_PATH=config/production/config.yaml

# 运行应用
CMD ["python", "src/main.py"]
```

创建`docker-compose.production.yml`：

```yaml
version: '3'

services:
  trading_app:
    build: .
    container_name: bybit_trading_app
    restart: always
    env_file:
      - .env.production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - influxdb
      - mysql
      - redis

  influxdb:
    image: influxdb:2.7
    container_name: influxdb
    restart: always
    ports:
      - "8086:8086"
    volumes:
      - influxdb-data:/var/lib/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=${INFLUXDB_ADMIN_PASSWORD}
      - DOCKER_INFLUXDB_INIT_ORG=bybit_trading
      - DOCKER_INFLUXDB_INIT_BUCKET=market_data
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=${INFLUXDB_TOKEN}

  mysql:
    image: mysql:8.0
    container_name: mysql
    restart: always
    ports:
      - "3306:3306"
    volumes:
      - mysql-data:/var/lib/mysql
    environment:
      - MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD}
      - MYSQL_DATABASE=bybit_trading
      - MYSQL_USER=bybit
      - MYSQL_PASSWORD=${MYSQL_PASSWORD}

  redis:
    image: redis:6.2
    container_name: redis
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --requirepass ${REDIS_PASSWORD}

  grafana:
    image: grafana/grafana:9.5.2
    container_name: grafana
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    depends_on:
      - influxdb
      - prometheus

  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

volumes:
  influxdb-data:
  mysql-data:
  redis-data:
  grafana-data:
  prometheus-data:
```

### 5.2 部署脚本

创建`scripts/deploy.sh`：

```bash
#!/bin/bash

# 部署脚本

# 检查环境变量文件
if [ ! -f .env.production ]; then
    echo "错误: .env.production 文件不存在"
    exit 1
fi

# 加载环境变量
source .env.production

# 构建和启动容器
docker-compose -f docker-compose.production.yml up -d

# 检查服务状态
echo "检查服务状态..."
sleep 10
docker-compose -f docker-compose.production.yml ps

echo "部署完成!"
```

赋予执行权限：

```bash
chmod +x scripts/deploy.sh
```

## 6. 总结

本文档详细介绍了Bybit高频交易系统的环境搭建步骤，包括：

1. 开发环境和生产环境的硬件和软件要求
2. 基础环境安装步骤
3. 项目依赖安装
4. 数据库和监控工具的设置
5. 项目目录结构和配置文件创建
6. 基础工具类的实现
7. 环境验证方法
8. 开发工作流程
9. 生产环境部署准备

按照本文档的指导，可以快速搭建一个完整的高频交易系统开发环境，为后续的系统实现和部署做好准备。

## 7. 下一步

完成环境搭建后，下一步将进行以下工作：

1. 实现WebSocket客户端，订阅实时K线数据
2. 实现历史数据下载器
3. 创建数据存储解决方案
4. 实现数据处理管道
5. 开发回测框架和策略优化模块
