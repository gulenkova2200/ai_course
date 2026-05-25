from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
print("Путь добавлен")

from src.recommender.model import UserCFRecommender
print("Модель импортирована")

app = FastAPI()
print("app создана")

@app.get("/")
def root():
    return {"ok": True}
print("готово")
