import base64
from io import BytesIO
from typing import Any
import cv2
import eventlet  # type: ignore
import eventlet.wsgi  # type: ignore
from flask import Flask
import numpy as np
from PIL import Image
import socketio  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 20
model: Any = None


def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


@sio.on('telemetry')  # type: ignore
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image, verbose=0)[0][0])
    throttle = 1.0 - speed / speed_limit
    print(f'{steering_angle} {throttle} {speed}')
    send_control(steering_angle, throttle)


@sio.on('connect')  # type: ignore
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })


if __name__ == '__main__':
    model = load_model('./model.h5', compile=False)
    app = socketio.Middleware(sio, app)  # type: ignore
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
