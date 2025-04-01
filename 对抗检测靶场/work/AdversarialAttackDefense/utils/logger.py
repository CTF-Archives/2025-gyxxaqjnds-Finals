import logging
import os

def setup_logger(name, log_file, level=logging.INFO):

    """
        DEBUG：用于开发和调试阶段，记录详细的程序执行信息。
        INFO：用于记录程序的正常运行状态，例如启动和关闭事件。
        WARNING：用于记录潜在问题，不影响程序运行，但需要关注。
        ERROR：用于记录由于某些问题导致的功能失效。
        CRITICAL：用于记录导致程序中止的严重错误。

    """

    """设置日志配置"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建日志文件夹
    log_dir = os.path.dirname(log_file)
    absolute_log_path = os.path.abspath(log_file)
    print("Absolute log path:", absolute_log_path)
    if not os.path.exists(log_dir):

        os.makedirs(log_dir)

    # 创建文件处理器
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)


    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

# 示例用法
if __name__ == "__main__":
    log_path = './logs/app.log'
    logger = setup_logger('my_logger', log_path)
    
    logger.info("This is an info message")
    logger.error("This is an error message")