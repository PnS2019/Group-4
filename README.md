# Facial Expressions Recognition using Raspberry PI

### Introduction

A facial expressions recognition implemented by keras based on tensorflow, which can classify 7 emotions(anger, contempt, disgust, fear, happy, sadness, and surprise). datasets used are CK+ and JAFFE.

### Dependencies

```
python 3.6
tensorflow 1.13.1
scikit-learn 0.20.3
numpy 1.16.3
matplotlib 3.0.3 
opencv 3.4.2
```
### Labels Distribution
 ![image](https://github.com/PnS2019/Group-4/blob/master/FaceMaster/label_distribution.png)

### Neural Network and Training Results

```
Layer (type)
input_1 (InputLayer)
conv2d (Conv2D)
batch_normalization_v1
conv2d_1 (Conv2D)
batch_normalization_v1_1
average_pooling2d
conv2d_2 (Conv2D)
batch_normalization_v1_2
average_pooling2d_1
flatten (Flatten)
dense (Dense)
dropout (Dropout)
dense_1 (Dense)
dropout_1 (Dropout)
dense_2 (Dense) 

Total params: 608,296
Trainable params: 608,168
Non-trainable params: 128
Train on 431 samples, validate on 108 samples

Train loss:  0.13326735931109387
Train accuracy:  95.82366347312927
Test loss:  0.7961734533309937
Test accuracy:  75.9259283542633
```


## Authors

* **Xiaorui Yin** 
* **Kevin** 
* **Arsim** 
