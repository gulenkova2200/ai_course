# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- `CIFAR100` дла понятного classification трека.
- `OxfordIIITPet`для понятного  segmentation трека.
-  В части А сравнивалась CNN с аугментациями и без аугментаций. В части B модели DeepLabV3_ResNet50_Weights (score_threshold = 0.3 и score_threshold = 0.7).

## 2. Среда и воспроизводимость

- Python: 3.12
- torch / torchvision: 2.11.0+cu126
- Устройство (CPU/GPU): GPU 
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `CIFAR100`
- Разделение: train/val/test: 80%/20% от train, отдельный test
- Базовые transforms: ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)
- Augmentation transforms: RandomHorizontalFlip(p=0.5),  RandomCrop(32, padding=4)
- Комментарий (2-4 предложения): CIFAR100 содержит 60,000 изображений размером 32x32 пикселя с 100 классами, что делает задачу значительно сложнее чем CIFAR10. Маленький размер изображений (32x32) и большое количество классов (100) делают задачу сложной для простых архитектур. Аугментации особенно важны из-за малого размера изображений.

### 3.2. Часть B: structured vision

- Датасет: `OxfordIIITPet` 
- Трек: `segmentation`
- Что считается ground truth: Маска где pixel value == 1 (foreground - животное), pixel value == 0 (background)
- Какие предсказания использовались: Бинарная маска после порога 0.5 на выходе модели DeepLabV3_ResNet50
- Комментарий (2-4 предложения): Сегментация животных на OxfordIIITPet — разумная задача для демонстрации semantic segmentation, так как требует точного выделения объекта сложной формы на разнообразном фоне. Метрика IoU хорошо отражает качество сегментации.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): SimpleCNN, no Augmentation, no Pretrained, Val Accuracy - 0.141
- C2 (simple-cnn-aug): SimpleCNN, Augmentated, no Pretrained, Val Accuracy - 0.119
- C3 (resnet18-head-only): ResNet18, Augmentated, head-only Pretrained, Val Accurecy - 0.349
- C4 (resnet18-finetune): ResNet18, Augmentated, fine-tune, Val Accurecy - 0.495, Test Accuracy - 0.549

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam (lr=0.001 для C1-C3, layer4:1e-4 + fc:1e-3 для C4)
- Batch size:  64
- Epochs (макс): 10 (C1-C2), 3 (C3-C4, fast mode)
- Критерий выбора лучшей модели: Максимальная validation accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран segmentation track

- Модель: DeepLabV3_ResNet50 (pretrained на COCO)
- Что считается foreground: 0.3
- V1: threshold = 0.3, без постобработки
- V2: remove components < 200 pixels (удаление мелких компонент)
- Как считался mean IoU: Среднее IoU по 100 случайным сэмплам из датасета
- Считались ли дополнительные pixel-level метрики: нет

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/segmentaton_metrics.png`, `./artifacts/figures/segmentation_examples.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: C4 (ResNet18 с fine-tuning layer4 и fc)
- Лучшая `val_accuracy`: 0.495 (C4)
- Итоговая `test_accuracy` лучшего классификатора: 0.549 (C4)
- Что дали аугментации (C2 vs C1):  Аугментации не дали улучшения (0.119 vs 0.141), возможно из-за малого количества эпох (10) для простой CNN на сложном датасете
- Что дал transfer learning (C3/C4 vs C1/C2): Значительное улучшение (0.349-0.495 vs 0.119-0.141), transfer learning дал ~3.5x улучшение accuracy
- Что оказалось лучше: head-only или partial fine-tuning: Partial fine-tuning (C4: 0.495 vs C3: 0.349), fine-tuning layer4 дал +42% относительно head-on
- Что показал режим V1 во второй части: : Mean IoU = 0.2958, базовая сегментация без постобработки
- Что показал режим V2 во второй части:  Mean IoU = 0.2958, удаление мелких компонент не улучшило результат (возможно, порог 200 пикселей слишком высокий)
- Как интерпретируются метрики второй части: Mean IoU показывает среднее качество сегментации по всем классам, значение 0.2958 указывает на умеренное качество сегментации

## 7. Анализ

(8-15 предложений)

Нужно прокомментировать:

Простая CNN показала низкие результаты на CIFAR100 (14.1% без аугментаций), что объясняется сложностью датасета (100 классов, маленькие изображения 32x32). Аугментации не дали устойчивого улучшения, возможно из-за недостаточного количества эпох для обучения сложной архитектуры с нуля.
Transfer learning с pretrained ResNet18 дал значительное улучшение (34.9% для head-only), что подтверждает эффективность transfer learning для сложных задач классификации. Partial fine-tuning (C4) показал лучший результат (49.5% val, 54.9% test), что показывает важность fine-tuning не только последнего слоя, но и предпоследних слоёв предобученной модели.
Head-only подход дал хорошее базовое качество, но partial fine-tuning дал дополнительное улучшение на ~15 процентных пунктов, что показывает важность адаптации feature extractor под конкретную задачу.
В части B метрика Mean IoU хорошо подходит для задачи сегментации, так как учитывает как precision, так и recall на pixel level. Режим V2 с удалением мелких компонент не дал улучшения, возможно из-за того, что порог 200 пикселей слишком высокий для некоторых животных на изображениях.
Основные ошибки модели в части B связаны с неточным выделением границ животных и смешиванием с фоном, что типично для semantic segmentation задач.4

## 8. Итоговый вывод

В качестве базового конфига классификации я бы взял C4 (ResNet18 с fine-tuning), так как он показал лучший результат и демонстрирует эффективность transfer learning.
Главное, что я понял про transfer learning: предобученные модели дают значительное преимущество на сложных датасетах, а fine-tuning нескольких слоёв работает лучше чем только head.
Главное про segmentation: метрика IoU хорошо отражает качество сегментации, но требует тщательного подбора постобработки и порогов для конкретных задач.ач.

