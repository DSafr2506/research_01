# Исследование функций активации и оптимизаторов на Tiny ImageNet

##  Описание проекта

Целью данного проекта является изучение влияния различных **функций активации** и **оптимизаторов** на **сходимость нейросетей** при классификации изображений. В качестве эталонного датасета использован [Tiny ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet), включающий 200 категорий изображений размером 64×64.

---

##  Используемая архитектура нейросети

- **Вход**: 64×64×3 изображения  
- **Скрытый слой**: 512 нейронов  
- **Выход**: 200 классов  
- **Функция потерь**: `CrossEntropy`  
- **Batch size**: 64  
- **Эпох**: 10  
- **Инициализация**: Xavier (Glorot Uniform)

---

##  Исследуемые функции активации

- `ReLU`  
- `Swish`  
- `GELU`  
- `PAReLU`  
- `DynamicSwish`  
- `AdaptiveGELUMish`

### Графики активаций:

<p align="center">
  <img src="img/activation_relu.png" width="300"/>
  <img src="img/activation_swish.png" width="300"/>
  <img src="img/activation_gelu.png" width="300"/>
</p>
<p align="center">
  <img src="img/activation_parelu.png" width="300"/>
  <img src="img/activation_dynamicswish.png" width="300"/>
  <img src="img/activation_adaptivegelumish.png" width="300"/>
</p>

---

##  Использованные оптимизаторы

- `SGD`
- `Adam`
- `AdamW`
- `RAdam`
- `Sophia`
- `LAMB`

Также протестированы комбинации:  
- `SGD + Adam`  
- `AdamW + RAdam`  
- `SGD + Adam + RAdam`

---

##  Результаты

### 1. Все конфигурации:

<p align="center">
  <img src="img/all_configs_loss.png" width="600"/>
</p>

### 2. Только одиночные оптимизаторы:

<p align="center">
  <img src="img/single_optimizers_loss.png" width="600"/>
</p>

### 3. Комбинированные оптимизаторы:

<p align="center">
  <img src="img/combined_optimizers_loss.png" width="600"/>
</p>

---

##  Лучшие конфигурации

| Активация       | Оптимизатор | Loss @ Epoch 10 | Accuracy @ Epoch 10 | Время (сек) |
|-----------------|-------------|------------------|----------------------|-------------|
| Swish           | AdamW       | **0.1557**       | **82.4%**            | 136         |
| PAReLU          | RAdam       | 0.1660           | 81.1%                | 145         |
| GELU            | Sophia      | 0.1823           | 79.8%                | 157         |

---

##  Выводы

- **GELU** и **Swish** обеспечивают лучшую сходимость.
- **Adam**, **RAdam** и **AdamW** — наиболее устойчивые оптимизаторы.
- Комбинации дают выигрыш в устойчивости, но часто переусложняют обучение.
- Sophia требует дальнейшей адаптации, а LAMB — хорош для масштабируемости.

---

##  Как запустить

```bash
pip install -r requirements.txt
jupyter notebook research_02_with_patterns.ipynb
```


