from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
import os
import captcha_config as config

def random_captcha():
    captcha_text = []
    for i in range(config.MAX_CAPTCHA):
        c = random.choice(config.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)

def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

if not os.path.exists(config.TRAIN_DATASET_PATH):
    os.makedirs(config.TRAIN_DATASET_PATH)

for i in range(config.TRAIN_DATASET_COUNT):
    now = str(int(time.time()))
    text, image = gen_captcha_text_and_image()
    filename = text + '_' + now + '.png'
    image.save(config.TRAIN_DATASET_PATH + os.path.sep + filename)

if not os.path.exists(config.TEST_DATASET_PATH):
    os.makedirs(config.TEST_DATASET_PATH)

for i in range(config.TEST_DATASET_COUNT):
    now = str(int(time.time()))
    text, image = gen_captcha_text_and_image()
    filename = text + '_' + now + '.png'
    image.save(config.TEST_DATASET_PATH + os.path.sep + filename)
