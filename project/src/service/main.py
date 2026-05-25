from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List
from pathlib import Path
import sys
import pandas as pd 
from contextlib import asynccontextmanager
from src.service.logs import logger



sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from src.recommender.model import UserCFRecommender
    logger.info("Импорт успешен")
except Exception as e:
    logger.error(f"Ошибка импорта: {e}")
    raise

recommender = None
movie_titles = {}  

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender, movie_titles
    
    logger.info("Загрузка модели...")
    model_path = Path(__file__).parent.parent / "models" / "usercf_model.pkl"
    logger.info(f"Путь к модели: {model_path}")
    
    if not model_path.exists():
        logger.error(f"Модель не найдена: {model_path}")
        recommender = None
    else:
        recommender = UserCFRecommender(str(model_path))

        movies_path = Path(__file__).parent.parent / "data" / "movies.csv"
        if movies_path.exists():
            movies_df = pd.read_csv(movies_path)
            movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
            logger.info(f"Загружено названий фильмов: {len(movie_titles)}")
        else:
            logger.warning("Файл с названиями не найден")
            for i in range(1, 20000):
                movie_titles[i] = f"Movie_{i}"
        
        logger.info("Модель загружена!")
    
    yield
    recommender = None

app = FastAPI(title="Movie Recommender API", lifespan=lifespan)

class RatingRequest(BaseModel):
    user_id: int
    movie_id: int

class RatingResponse(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[dict]
    count: int

@app.get("/recommend/{user_id}", response_model=RecommendResponse)
async def recommend(user_id: int, k: int = 10):
    if recommender is None:
        logger.error("Модель не загружена")
        raise HTTPException(503, "Модель не загружена")
    try:
        recs = recommender.recommend(user_id, k)
        recommendations_with_titles = []
        for movie_id in recs:
            title = movie_titles.get(movie_id, f"Movie_{movie_id}")
            recommendations_with_titles.append({
                "movie_id": movie_id,
                "title": title
            })
        
        logger.info(f"Рекомендации для user {user_id}: {recs}")

        return RecommendResponse(
            user_id=user_id, 
            recommendations=recommendations_with_titles, 
            count=len(recommendations_with_titles)
        )
        
    except Exception as e:
        logger.error(f"Ошибка в recommend: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/predict", response_model=RatingResponse)
async def predict(request: RatingRequest):
    if recommender is None:
        logger.error("Модель не загружена")
        raise HTTPException(503, "Модель не загружена")
    try:
        rating = recommender.predict(request.user_id, request.movie_id)
        logger.info(f"Предсказание: user={request.user_id}, movie={request.movie_id}, rating={rating}")
        return RatingResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=round(rating, 2)
        )
    except Exception as e:
        logger.error(f"Ошибка в predict: {e}")
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health():
    if recommender is None:
        logger.warning("Health check: модель не загружена")
        return {"status": "error", "message": "Модель не загружена"}
    
    logger.info("Health check OK")
    return {
        "status": "ok", 
        "users": len(recommender.user_ids), 
        "movies": len(recommender.movie_ids)
    }

@app.get("/")
async def root():
    return {
        "message": "Movie Recommender API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "recommend": "/recommend/{user_id}",
        }
    }

