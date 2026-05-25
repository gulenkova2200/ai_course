import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))


from src.recommender.model import UserCFRecommender


model_path = root / "src" / "models" / "usercf_model.pkl"

print(f"путь к модели: {model_path}")
print(f"модель существует: {model_path.exists()}")

if not model_path.exists():
    print("модель не найдена!")
    for pkl in root.rglob("*.pkl"):
        print(f"  {pkl.relative_to(root)}")
    exit(1)

recommender = UserCFRecommender(str(model_path))

print(f"\модель загружена")
print(f"пользователей: {len(recommender.user_ids)}")
print(f"фильмов: {len(recommender.movie_ids)}")
print(f"5 пользователей: {recommender.user_ids[:5]}")
print(f"5 фильмов: {recommender.movie_ids[:5]}")


print("тест рекомендаций:")

try:
    if 1 in recommender.user_ids:
        print(f"пользователь 1 найден в модели")
        recs = recommender.recommend(1, k=5)
        print(f"Рекомендации: {recs}")
    else:
        print(f"пользователь 1 не найден в модели")
        
except Exception as e:
    print(f"ошибка: {e}")
    import traceback
    traceback.print_exc()