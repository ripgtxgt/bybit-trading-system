# Bybit高频交易系统部署指南

本指南提供了详细的步骤，帮助您部署和配置Bybit高频交易系统。请按照以下步骤操作，确保系统正确安装和运行。

## 目录

1. [前置准备](#前置准备)
2. [系统安装](#系统安装)
3. [数据库配置](#数据库配置)
4. [系统配置](#系统配置)
5. [启动系统](#启动系统)
6. [监控与维护](#监控与维护)
7. [常见问题](#常见问题)

## 前置准备

在开始部署之前，请确保您的服务器环境满足[服务器需求文档](server_requirements.md)中列出的所有要求。

### 操作系统准备

以下步骤以Ubuntu 22.04 LTS为例：

1. 更新系统包：

```bash
sudo apt update && sudo apt upgrade -y
```

2. 安装基础工具：

```bash
sudo apt install -y build-essential git curl wget vim htop net-tools
```

3. 配置时区和NTP：

```bash
sudo timedatectl set-timezone Asia/Shanghai
sudo apt install -y ntp
sudo systemctl enable ntp
sudo systemctl start ntp
```

4. 优化系统参数：

```bash
# 创建系统优化配置文件
sudo tee /etc/sysctl.d/99-trading-system.conf > /dev/null <<EOF
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 60
net.ipv4.tcp_keepalive_probes = 10
fs.file-max = 1000000
vm.swappiness = 10
EOF

# 应用系统参数
sudo sysctl -p /etc/sysctl.d/99-trading-system.conf
```

5. 增加文件描述符限制：

```bash
sudo tee /etc/security/limits.d/99-trading-system.conf > /dev/null <<EOF
*               soft    nofile          1000000
*               hard    nofile          1000000
EOF
```

### 安装Python环境

1. 安装Python 3.9或更高版本：

```bash
sudo apt install -y python3 python3-pip python3-dev python3-venv
```

2. 验证Python版本：

```bash
python3 --version
```

确保版本为3.9或更高。

### 安装Docker（可选，用于容器化部署）

1. 安装Docker：

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

2. 安装Docker Compose：

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.15.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

3. 将当前用户添加到docker组：

```bash
sudo usermod -aG docker $USER
```

注意：添加用户到docker组后，需要重新登录才能生效。

## 系统安装

### 方法1：直接安装（推荐用于开发/测试环境）

1. 克隆代码仓库：

```bash
git clone https://github.com/yourusername/bybit_trading_system.git
cd bybit_trading_system
```

2. 创建并激活虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 安装TA-Lib（技术分析库）：

```bash
# 安装TA-Lib依赖
sudo apt install -y build-essential libssl-dev
sudo apt install -y libta-lib-dev

# 安装Python TA-Lib包
pip install ta-lib
```

如果上述命令安装失败，可以尝试使用预编译的二进制包：

```bash
pip install ta-lib-binary
```

### 方法2：Docker容器化部署（推荐用于生产环境）

1. 克隆代码仓库：

```bash
git clone https://github.com/yourusername/bybit_trading_system.git
cd bybit_trading_system
```

2. 构建Docker镜像：

```bash
docker-compose build
```

## 数据库配置

### 安装和配置InfluxDB

1. 安装InfluxDB：

```bash
# 添加InfluxData仓库
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

# 更新包列表并安装InfluxDB
sudo apt update
sudo apt install -y influxdb2
```

2. 启动InfluxDB服务：

```bash
sudo systemctl enable influxdb
sudo systemctl start influxdb
```

3. 初始化InfluxDB：

```bash
# 访问InfluxDB UI进行初始化
# 打开浏览器访问 http://your_server_ip:8086
# 按照UI向导创建组织、用户和初始bucket

# 或者使用命令行初始化
influx setup \
  --username admin \
  --password YourSecurePassword \
  --org bybit_trading \
  --bucket market_data \
  --retention 30d \
  --force
```

4. 创建API令牌：

```bash
# 获取组织ID
ORG_ID=$(influx org find --name bybit_trading --json | jq -r '.[0].id')

# 创建API令牌
influx auth create --org-id $ORG_ID --description "Trading System Token" --all-access
```

记下生成的API令牌，稍后将在系统配置中使用。

### 安装和配置MySQL

1. 安装MySQL：

```bash
sudo apt install -y mysql-server
```

2. 配置MySQL安全设置：

```bash
sudo mysql_secure_installation
```

按照提示完成安全配置。

3. 创建数据库和用户：

```bash
sudo mysql -e "CREATE DATABASE bybit_trading;"
sudo mysql -e "CREATE USER 'bybit'@'localhost' IDENTIFIED BY 'YourSecurePassword';"
sudo mysql -e "GRANT ALL PRIVILEGES ON bybit_trading.* TO 'bybit'@'localhost';"
sudo mysql -e "FLUSH PRIVILEGES;"
```

## 系统配置

### 配置文件设置

1. 创建配置文件：

```bash
cp config.example.ini config.ini
```

2. 编辑配置文件：

```bash
vim config.ini
```

根据您的环境修改以下配置：

```ini
[api]
base_url = https://api.bybit.com
key = YOUR_BYBIT_API_KEY
secret = YOUR_BYBIT_API_SECRET

[database]
[influxdb]
url = http://localhost:8086
token = YOUR_INFLUXDB_TOKEN
org = bybit_trading
bucket = market_data

[mysql]
host = localhost
port = 3306
user = bybit
password = YourSecurePassword
database = bybit_trading

[websocket]
url = wss://stream.bybit.com/v5/public
ping_interval = 30
reconnect_interval = 5
max_reconnects = 10

[trading]
symbol = BTCUSDT
interval = 1
leverage = 1
risk_per_trade = 0.02
max_positions = 1
```

### 环境变量设置

为了安全起见，建议使用环境变量存储敏感信息：

1. 创建环境变量文件：

```bash
cp .env.example .env
```

2. 编辑环境变量文件：

```bash
vim .env
```

添加以下内容：

```
BYBIT_API_KEY=YOUR_BYBIT_API_KEY
BYBIT_API_SECRET=YOUR_BYBIT_API_SECRET
INFLUXDB_TOKEN=YOUR_INFLUXDB_TOKEN
MYSQL_PASSWORD=YourSecurePassword
```

## 启动系统

### 方法1：直接启动（开发/测试环境）

1. 激活虚拟环境：

```bash
source venv/bin/activate
```

2. 启动数据收集系统：

```bash
python -m src.main data_collection
```

3. 启动回测系统（可选）：

```bash
python -m src.main backtest
```

### 方法2：使用Supervisor管理进程（生产环境）

1. 安装Supervisor：

```bash
sudo apt install -y supervisor
```

2. 创建Supervisor配置文件：

```bash
sudo tee /etc/supervisor/conf.d/bybit_trading.conf > /dev/null <<EOF
[program:bybit_data_collection]
command=/home/ubuntu/bybit_trading_system/venv/bin/python -m src.main data_collection
directory=/home/ubuntu/bybit_trading_system
user=ubuntu
autostart=true
autorestart=true
startretries=10
stopwaitsecs=30
stdout_logfile=/home/ubuntu/bybit_trading_system/logs/data_collection.log
stderr_logfile=/home/ubuntu/bybit_trading_system/logs/data_collection_error.log
environment=PYTHONUNBUFFERED=1

[program:bybit_trading]
command=/home/ubuntu/bybit_trading_system/venv/bin/python -m src.main trading
directory=/home/ubuntu/bybit_trading_system
user=ubuntu
autostart=true
autorestart=true
startretries=10
stopwaitsecs=30
stdout_logfile=/home/ubuntu/bybit_trading_system/logs/trading.log
stderr_logfile=/home/ubuntu/bybit_trading_system/logs/trading_error.log
environment=PYTHONUNBUFFERED=1
EOF
```

3. 创建日志目录：

```bash
mkdir -p logs
```

4. 重新加载Supervisor配置并启动服务：

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start all
```

### 方法3：使用Docker Compose（容器化部署）

1. 启动所有服务：

```bash
docker-compose up -d
```

2. 查看服务状态：

```bash
docker-compose ps
```

3. 查看日志：

```bash
docker-compose logs -f
```

## 监控与维护

### 设置基本监控

1. 安装Prometheus和Grafana（可选）：

```bash
# 安装Prometheus
sudo apt install -y prometheus

# 安装Grafana
sudo apt-get install -y apt-transport-https software-properties-common
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt update
sudo apt install -y grafana
```

2. 配置Prometheus监控：

```bash
# 创建Prometheus配置
sudo tee /etc/prometheus/prometheus.yml > /dev/null <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'trading_system'
    static_configs:
      - targets: ['localhost:8000']
EOF

# 重启Prometheus
sudo systemctl restart prometheus
```

3. 启动Grafana：

```bash
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

访问Grafana UI（http://your_server_ip:3000）并配置Prometheus数据源和仪表板。

### 日常维护任务

1. 备份数据库：

```bash
# 备份MySQL数据库
mysqldump -u bybit -p bybit_trading > backup_$(date +%Y%m%d).sql

# 备份InfluxDB数据（需要先安装influxdb-cli）
influx backup /path/to/backup/$(date +%Y%m%d) -t YOUR_INFLUXDB_TOKEN
```

2. 日志轮转：

```bash
# 安装logrotate（如果尚未安装）
sudo apt install -y logrotate

# 配置日志轮转
sudo tee /etc/logrotate.d/bybit_trading > /dev/null <<EOF
/home/ubuntu/bybit_trading_system/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 ubuntu ubuntu
}
EOF
```

3. 系统更新：

```bash
# 更新代码库
cd /home/ubuntu/bybit_trading_system
git pull

# 更新依赖
source venv/bin/activate
pip install -r requirements.txt

# 重启服务
sudo supervisorctl restart all
```

## 常见问题

### 1. 系统无法连接到Bybit API

**问题**: 系统日志显示无法连接到Bybit API或WebSocket。

**解决方案**:
- 检查API密钥和密钥是否正确
- 验证服务器网络连接
- 确认Bybit API服务是否可用
- 检查防火墙设置是否阻止了出站连接

### 2. 数据库连接错误

**问题**: 系统无法连接到InfluxDB或MySQL数据库。

**解决方案**:
- 验证数据库服务是否正在运行：`sudo systemctl status influxdb` 或 `sudo systemctl status mysql`
- 检查数据库凭据是否正确
- 确认数据库用户权限设置
- 检查数据库连接字符串

### 3. 系统性能问题

**问题**: 系统响应缓慢或CPU/内存使用率过高。

**解决方案**:
- 检查系统资源使用情况：`htop`
- 优化数据库查询
- 增加服务器资源（CPU/内存）
- 调整系统配置参数

### 4. WebSocket连接频繁断开

**问题**: WebSocket连接不稳定，频繁断开重连。

**解决方案**:
- 检查网络稳定性和延迟
- 增加重连间隔和最大重连次数
- 使用更稳定的网络连接
- 考虑使用代理服务器

### 5. 数据不完整或丢失

**问题**: 系统收集的数据不完整或有缺失。

**解决方案**:
- 检查WebSocket连接状态
- 验证历史数据下载器配置
- 检查数据存储过程
- 实施数据验证和修复机制

## 结论

按照本指南完成部署后，您的Bybit高频交易系统应该已经正常运行。如果遇到任何问题，请查阅系统日志或联系技术支持。

建议在生产环境中部署前，先在测试环境中进行充分测试，确保系统在各种条件下都能稳定运行。
