import numpy as np
import cv2
import matplotlib.pyplot as plt

emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def predict(img, model):
    pred = model.predict(img)[0]
    emotion = emotions[np.argmax(model.predict(img), axis=1).astype(np.int)]

    x = [i for i in range(8)]
    plt.bar(x=x, height=pred, width=0.8, alpha=0.8, color='red', label="Prediction")
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.xticks([i for i in x], emotions)
    plt.xlabel("Emotion")
    plt.title("Your emotion is " + emotion + "!")
    plt.legend()
    plt.show()
