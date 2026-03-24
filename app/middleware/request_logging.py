"""
REQ/RES 응답 로깅 미들웨어

모든 API 호춣에 대해 자동적으로
- 응답 시간
- http method, path, status_code
- req IP
를 기록함

느린 요청 (1초 이상)은 WARNING으로 별도 표시
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.middleware.request")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    
    async def dispatch(self, request: Request, call_next) -> Response:
        #health check는 로깅 스킵
        if request.url.path in ("/health","/metrics","/"):
            return await call_next(request)
        
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        
        try :
            response = await call_next(request)
        except Exception as e :
            durations_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{request.method} {request.url.path} -> 500 ({durations_ms:.1f}ms)",
                extra={
                    "method":request.method,
                    "path":request.url.path,
                    "status_code":500,
                    "duration_ms":round(durations_ms, 1),
                    "client_ip": client_ip
                }
            )
            raise 
        durations_ms = (time.perf_counter() - start_time) * 1000
        status_code = response.status_code
        
        log_data = {
                    "method":request.method,
                    "path":request.url.path,
                    "status_code":status_code,
                    "duration_ms":round(durations_ms, 1),
                    "client_ip": client_ip
                }
        
        message = f"{request.method} {request.url.path} -> {status_code} ({durations_ms:.1f}ms)"
        
        if durations_ms > 1000:
            logger.warning(f"SLOW {message}",extra=log_data)
            
        elif status_code >= 400 :
            logger.warning(message, extra=log_data)
            
        else: 
            logger.info(message, extra=log_data)
            
        return response
    