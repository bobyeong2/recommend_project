import random
from tests.load.locust import HttpUser, task, between, events


class BobUser(HttpUser):
    host = "http://localhost:8100"
    wait_time = between(1, 3)

    def on_start(self):
        self.token = None
        self.user_rating_ids = []
        self._login()

    def _login(self):
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@test.com",
                "password": "testpass123"
            },
            name="POST /auth/login"
        )
        if response.status_code == 200:
            self.token = response.json().get("access_token")

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    @task(4)
    def get_recommendations(self):
        if not self.token:
            return
        self.client.get(
            "/api/v1/recommendations",
            headers=self._headers(),
            name="GET /recommendations"
        )

    @task(3)
    def get_movies(self):
        skip = random.randint(0, 100)
        self.client.get(
            f"/api/v1/movies?skip={skip}&limit=20",
            name="GET /movies"
        )

    @task(2)
    def get_movie_detail(self):
        movie_id = random.randint(1, 1000)
        self.client.get(
            f"/api/v1/movies/{movie_id}",
            name="GET /movies/{id}"
        )

    @task(1)
    def post_rating(self):
        if not self.token:
            return
        movie_id = random.randint(1, 1000)
        score = round(random.uniform(1.0, 10.0), 1)
        response = self.client.post(
            "/api/v1/ratings",
            json={"movie_id": movie_id, "score": score},
            headers=self._headers(),
            name="POST /ratings"
        )
        if response.status_code == 200:
            rating_id = response.json().get("id")
            if rating_id:
                self.user_rating_ids.append(rating_id)