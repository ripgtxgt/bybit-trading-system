import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

class Logger:
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name):
        if name in cls._loggers:
            return cls._loggers[name]
        
        # 创建新的logger
        logger = logging.getLogger(name)
        
        # 设置日志级别
        log_level = os.getenv('LOG_LEVEL', 'DEBUG')
        logger.setLevel(getattr(logging, log_level))
        
        # 确保日志目录存在
        log_dir = 'logs'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = f"{log_dir}/{name}.log"
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # 创建文件处理器
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level))
        
        # 设置格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        # 保存logger
        cls._loggers[name] = logger
        
        return logger
