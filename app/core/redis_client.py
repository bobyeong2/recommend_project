# app/core/redis_client.py

from redis import asyncio as aioredis
from typing import Optional
import json
import logging
import time
logger = logging.getLogger(__name__)

class RedisClient:
    """
    Redis 클라이언트 (싱글톤)
    
    추천 결과 캐싱 및 JWT 블랙리스트 관리
    """
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        
    async def connect(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Redis 연결
        """
        try :
            self.redis = await aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses = True
            )
            # 연결 테스트
            await self.redis.ping()
            logger.info("Redis 연결 성공")
        except Exception as e :
            logger.error(" Redis 연결 실패")
            self.redis = None
            
    async def disconnect(self):
        """
        Redis 연결 종료
        """
        if self.redis:
            await self.redis.close()
            logger.info("Redis 연결 종료")
        
    async def get_recommendation_cache(self, user_id: int) -> Optional[dict]:
        """
        추천 결과 캐싱 조회
        
        Args:
            iser_id: 사용자 ID
        
        Return:
            캐싱된 추천결과 또는 None
        """
        if not self.redis:
            return None
        
        try :
            key = f"recommendation:user:{user_id}"
            data = await self.redis.get(key)
            
            if data:
                logger.info(f"캐싱 히트 : user_id = {user_id}")
                return json.loads(data)
            logger.info(f"캐싱 미스 : user_id={user_id}")
            return None
        except Exception as e :
            logger.error(f"캐싱 조회 오류 : {e}")
            return None
        
    async def set_recommendation_cache(
        self,
        user_id: int,
        recommendations: dict,
        ttl: int = 3600 # 1시간
    ):
        """_summary_

        Args:
            user_id : 사용자 ID
            recommendations : 추천 결과
            ttl : 캐시 유효 시간(초)
        """
        
        if not self.redis:
            return None
        
        try :
            key = f"recommendation:user:{user_id}"
            await self.redis.setex(
                key,
                ttl,
                json.dumps(recommendations, ensure_ascii=False)
            )
            logger.info(f"캐시 저장 : user_id = {user_id}, ttl={ttl}s")
            
        except Exception as e :
            logger.error(f"캐시 저장 오류: {e}")
            
    async def invalidate_user_cache(self, user_id: int) :
        """_summary_

        Args:
            user_id: 사용자 ID
        """
        
        if not self.redis:
            return
        
        try :
            key = f"recommendation:user:{user_id}"
            deleted = await self.redis.delete(key)
            
            if deleted:
                logger.info(f"캐시 삭제: user_id={user_id}")
            else:
                logger.info(f"캐시 없음: user_id={user_id}")
        except Exception as e :
            logger.error(f"캐시 삭제 오류: {e}")
            
    async def blacklist_token(self, token: str, exp: int):
        """_summary_
        JWT 토큰 블랙리스트 등록(로그아웃)
        Args:
            token (str):  JWT 토큰
            exp (int): 만료 시간 (Unix timestamp)
        """
        
        if not self.redis:
            return
        
        try :
            key = f"blacklist:token:{token}"
            ttl = exp - int(time.time())
            
            if ttl > 0 :
                await self.redis.setex(key, ttl, "1")
                logger.info(f"토큰 블랙리스트 등록: ttl={ttl}s")
                
        except Exception as e :
            logger.error(f"토큰 블랙리스트 오류: {e}")
            
    async def is_token_blacklisted(self, token: str) -> bool:
        """_summary_

        Args:
            token (str): JWT 토큰

        Returns:
            bool: 블랙리스트 여부
        """
        
        if not self.redis:
            return False
        
        try :
            key = f"blacklist:token:{token}"
            exists = await self.redis.exists(key)
            return exists > 0 
        except Exception as e :
            logger.error(f"블랙리스트 확인 오류: {e}")
            return False
        
redis_client = RedisClient()
