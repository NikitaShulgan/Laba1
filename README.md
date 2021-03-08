# Лабораторная работа #1
## Подготовка окружения для решения задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения
###### Графики изначальные
 Оранживая - обучающая выборка, Синия - валидационная выборка (на всех графиках в данном отчете)
https://tensorboard.dev/experiment/TW5KbIZ5T0efWcXAi7WTyA/#scalars&run=train  

<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba1/main/epoch_categorical_accuracy%20(1).svg">

epoch_categorical_accuracy
![image](https://user-images.githubusercontent.com/80168174/110246503-e59ce580-7f78-11eb-934e-78ad2a30a6b2.png)

epoch_loss
![image](https://user-images.githubusercontent.com/80168174/110246531-02391d80-7f79-11eb-8751-7edd820b3c8a.png)
## Описание архитектуры
###### Сверточный слой, 8 фильтров, ядро 3x3.
```
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
```
###### Операция макс пуллинга для 2х мерных данных
```
x = tf.keras.layers.MaxPool2D()(x)
```
###### Flatten используется для конвертации входящих данных в меньшую размерность.
```
x = tf.keras.layers.Flatten()(x)
```
###### Dense-слой получает информацию со всех узлов предыдущего слоя, функция активации softmax
```
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```

## Анализ полученных результатов
Из-за ряда факторов, таких как:
1) избыточна сложная модель
2) недостаточность датасета
3) и д.р.
у нас модель переобучилась.

# Создать и обучить сверточную нейронную сеть произвольной архитектуры с количеством сверточных слоев >3.

## Сверточная нейронная сеть организована из стеков Conv2D, функции активации ReLU и операций MaxPooling2D.
```
 inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.activations.relu(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=16, kernel_size=3)(inputs)
  x = tf.keras.activations.relu(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=32, kernel_size=3)(inputs)
  x = tf.keras.activations.relu(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=64, kernel_size=3)(inputs)
  x = tf.keras.activations.relu(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=128, kernel_size=3)(inputs)
  x = tf.keras.activations.relu(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
 ```
  https://tensorboard.dev/experiment/8e972T9ERIqHqEGpvNRfYg/#scalars
  epoch_categorical_accuracy
  ![image](https://user-images.githubusercontent.com/80168174/110256009-9de18280-7fa7-11eb-86d7-da0a45158d6b.png)

 epoch_loss
![image](https://user-images.githubusercontent.com/80168174/110256019-aafe7180-7fa7-11eb-96f7-2cec5180112c.png)

 ## Анализ результатов
 У нас изначально было переобучение, из-за сложности модели и небольшого датасета. Я усложнил модель, датасет остался, следовательно результаты данной модели хуже чем изначальные. 
