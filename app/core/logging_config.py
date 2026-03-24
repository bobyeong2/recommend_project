"""
중앙 로깅 설정

모든 모듈에서 logging.getLogger(__name__)으로 적용하면

해당 설정이 적용되도록 하였음.

로그 출력:
- 콘솔: 가독성 좋은 포맷
- 파일: Json 포맷 (logs/app.log)

사용법:
    from app.core.logging_config import setup_logging
    setup_logging() # main.py에서 1회 호출함
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


class JSONFormatter(logging.Formatter):
    """
    Json 구조화 로그 포맷터
    """
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "level":record.levelname,
            "logger":record.name,
            "message":record.getMessage(),
            "module":record.module,
            "function":record.funcName,
            "line":record.lineno,
        }
        
        if  record.exc_info and record.exc_info[0] is not None :
            log_data["exception"] = self.formatException(record.exc_info)
            
        # 추가 필드 (extra로 전달된 값)
        for key in ("user_id","movie_id","method","path","status_code","duration_ms","strategy","cache_hit"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data, ensure_ascii=False)
    
class ConsoleFormatter(logging.Formatter):
    """
    콘솔용 가독성 포맷터
    """
    
    def format(self, record):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(8)
        name = record.name.split(".")[-1] # 마지막 모듈명만
        message = record.getMessage()
        
        base = f"{timestamp} | {level} | {name} | {message}"
        
        if record.exc_info and record.exc_info[0] is not None :
            base += "\n" + self.formatException(record.exc_info)
            
        return base
    
def setup_logging(log_level:str = "INFO", log_dir: str = "logs"):
    """
    앱 전체 로깅 설정
    
    Args:
        log_level: 로그레벨 설정(BEBUG, INFO, WARNING, ERROR)
        log_dir: 로그 파일 저장 위치
    """
    os.makedirs(log_dir, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 기존 핸들러 제거 (중복을 방지)
    root_logger.handlers.clear()
    
    # Console 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)
    
    # File 핸들러
    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024, # 10MB
        backupCount=1000,
        encoding="utf8"
    )
    
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # 에러 전용 파일
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir,"error.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=100,
        encoding="utf8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)
    
    #Sqlalchemy level 조정
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    
    #uvicorn 접근 로그는 미들웨어에서 처리해서 억제함.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    logging.getLogger(__name__).info("logging initialized", extra={"log_dir":log_dir})