import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def setup_logger():
    # 获取当前脚本文件名作为日志文件名（不包括扩展名）
    script_name = os.path.basename(__file__).replace('.py', '')
    
    # 获取当前日期，并格式化为 YYYY-MM-DD 格式
    current_date = datetime.now().strftime('%Y-%m-%d')

    # 设置日志文件夹路径
    log_folder = "log"
    log_filename = f"{script_name}_{current_date}.log"  # 日志文件名基于调用脚本名称

    # 检查并创建日志文件夹，如果不存在则创建
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    # 配置日志文件路径
    log_file_path = os.path.join(log_folder, log_filename)

    # 配置日志处理器（按日期流转日志，保留最近7天）
    log_handler = TimedRotatingFileHandler(
        log_file_path, when="midnight", interval=1, backupCount=7
    )
    log_handler.setLevel(logging.INFO)  # 设置日志级别为 INFO
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 配置控制台输出日志的处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 配置根日志记录器，输出到文件和控制台
    logger = logging.getLogger(script_name)  # 使用脚本名作为 logger 名称
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有的 handler（防止重复添加）
    logger.addHandler(log_handler)  # 输出到文件
    logger.addHandler(console_handler)  # 输出到控制台
    
    return logger
