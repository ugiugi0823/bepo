# 파일명: image_classification_train.py

'''
from image_classification_train_sub import exec_train, exec_init_svc, exec_inference
'''
import logging


def train(tm):

    exec_train(tm)
    logging.info('[hunmin log] the end line of the function [train]')


def init_svc(im):

    params = exec_init_svc(im)
    logging.info('[hunmin log] the end line of the function [init_svc]')

    return { **params }


def inference(text, params, batch_id):

    result = exec_inference(text, params, batch_id)
    logging.info('[hunmin log] the end line of the function [inference]')

    return { **result }
