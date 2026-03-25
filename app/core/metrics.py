"""
Prometheus 메트릭을 정의하는 공간

사용법: 
    from app.core.metrics import metrics
    
    metrics.recommendation_latency.observe(0.123)
    metrics.cache_hits.inc()
    
"""

from prometheus_client import Counter, Histogram, Gauge, Info

class AppMetrics:
    """
    앱 전체 메트릭을 관리하는 클래스
    """
    
    def __init__(self):
        #추천 API
        self.recommendation_requests = Counter(
            "recommendation_requests_total",
            "TOtal recommedation API requests",
            ["strategy"] # popular, content based, hybrid
        )
        
        self.recommendation_latency = Histogram(
            "recommendation_latency_seconds",
            "Recommendation generation latency",
            ["strategy"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Redis 캐시
        self.cache_hits = Counter(
            "redis_cache_hits_total",
            "Total Redis cache hits"
        )
        
        self.cache_misses = Counter(
            "redis_cache_misses_total",
            "Total Redis cache misses"
        )
        
        # Model Predict
        self.model_inference_latency = Histogram(
            "model_inference_latency_seconds",
            "NCF model inference latency",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        # 평점
        self.rating_operations = Counter(
            "rating_operations_total",
            "Total rating operations",
            ["operation"] # create, update, delete
        )
        
        # active user
        self.active_users = Gauge(
            "active_users_current",
            "Currently active users (approximate)"
        )
        
        # 앱 정보
        self.app_info = Info(
            "Bob_movie_app",
            "Application infomation"
        )
        
        self.app_info.info({
            "version": "2.2.0",
            "model": "NCF",
            "framework": "FastAPI"
        })
        
        
metrics = AppMetrics()