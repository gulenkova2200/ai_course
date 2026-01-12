# HW06 – Report

> Файл: `homeworks/HW06/report.md`  


## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: (1800, 39)
- Целевая переменная: `target` ('0': 73.74%, '1': 26.24%)
- Признаки: все данные числового типа(float64), target(int64)

## 2. Protocol

- Разбиение: train/test (0.2, `random_state`=42)
- Подбор: CV на train 5 фолдов (max_depth, min_samples_leaf, ccp_alpha, max_features, ccp_alpha, max_leaf_nodes)
- Метрики: accuracy(базовая наиболее понятная метрика, поэтому была выбрана), F1(опять же поскольку классы не сбалансированы), ROC-AUС(классы не сбалансированы)

## 3. Models

Сравниваемые модели:

- DummyClassifier ('stratified', потому что было важно сохранить баланс классов) (параметры не подбирались)
- LogisticRegression (подбирался параметр C)
- DecisionTreeClassifier (контроль сложности: `max_depth` + `min_samples_leaf` или `ccp_alpha`. Все параметры подбирались через cv. Подбор всех параметров позволил улучшить метрики, но замедлил выполнение кода)
- RandomForestClassifier(max_depth, min_samles_leaf, max_features. Все параметры подбирались через cv. Ограничение глубины позволяет контролировать переобучение.)
- GradientBoosting(learning_rate(Определяет, насколько сильно каждое новое дерево «исправляет» ошибки предыдущих.), max_depth(глубина дерева контролирует переобучение), max_leaf_nodes(задает максимальное число листьев для контроля переобучения).)
- StackingClassifier (Разделили на 5 фолдов. Было выбрано для него 3 модели(LogisticRegression, RandomForest, GradientBoosting) для мета-модели)

## 4. Results

- Таблица/список финальных метрик на test по всем моделям - в файле metrics_test.json
- Победитель (по ROC-AUC) - в файле best_model_meta. Stacking превзошёл одиночные модели, потому что объединяет их сильные стороны, компенсируя индивидуальные слабости. 

## 5. Analysis

- Проведено 5 запусков модели Random Forest с разными значениями random_state.
Метрика ROC-AUC менялась незначительно: в пределах тысячных значений (0.002-0.004(по модулю)).
Это говорит о высокой устойчивости результатов.
- Ошибки: confusion matrix для stacking - artifacts/figures/stacking_confusion_matrix_test.png. Модель предсказывает точно в 91.7% случаев
- Интерпретация: permutation importance (top-10/15). Наиболее важные признаки ('f16' - сильный отрыв от остальных, 'f01', 'f07', 'f19', 'f30' ). Путь: artifacts/figures/stacking(permutation importance)

## 6. Conclusion

- Одиночные деревья склонны к переобучению при большой глубине.
- Ансамблевые методы значительно повышают устойчивость и качество за счет агрегации.
- Устойчивость к random_state - один из признаков надежности модели.
