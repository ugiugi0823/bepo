# 파일명: image_classification_train_sub.py

# Imports
import os
import numpy as np
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import utils
# from tensorflow.keras import layers
# from tensorflow.keras.models import load_model
import logging
import base64
import io
from PIL import Image


import random
import torch
import pandas as pd

from torch.utils.data import TensorDataset, random_split
from transformers import AutoTokenizer
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup




logging.info(f'[hunmin log] torch ver : {torch.__version__}')

# 사용할 gpu 번호를 적는다.
os.environ["CUDA_VISIBLE_DEVICES"]='0'

gpus = torch.cuda.is_available()
if gpus:
    try:
        # torch.cuda.device_count()

        logging.info('[hunmin log] gpu set complete')
        logging.info('[hunmin log] num of gpu: {}'.format(torch.cuda.device_count()))

    except RuntimeError as e:
        logging.info('[hunmin log] gpu set failed')
        logging.info(e)


def exec_train(tm):

    logging.info('[hunmin log] the start line of the function [exec_train]')

    logging.info('[hunmin log] tm.train_data_path : {}'.format(tm.train_data_path))

    # 저장 파일 확인
    list_files_directories(tm.train_data_path)

    ###########################################################################
    ## 1. 데이터셋 준비(Data Setup)
    ###########################################################################

    my_path = os.path.join(tm.train_data_path, 'dataset') + '/'

    df = pd.read_csv(f'{my_path}/train.csv')

    # Tokenize all of the sentences and map the tokens to thier word IDs.

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    input_ids, attention_masks, labels = get_input_mask_label(df, tokenizer)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)


    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # test_size = len(test_dataset)

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    # print('{:>5,} test samples'.format(test_size))



    # dataset_numpy[5] 가 envelope numpy 이다.
    logging.info('[hunmin log] envelope : ')


    ###########################################################################
    ## 2. 데이터 전처리(Data Preprocessing)
    ###########################################################################



    # encoding된 결과 확인 및 원래 배열의 형태와 비교
    # logging.info('[hunmin log] Y_train : {}'.format(Y_train.shape))
    # logging.info('[hunmin log] Y_train_cnn : {}'.format(Y_train_cnn.shape))
    # logging.info('[hunmin log] class number : {}'.format(num_classes))



    ###########################################################################
    ## 3. 학습 모델 훈련(Train Model)
    ###########################################################################

    # 모델 구축 (Build Model)
    # 이미지 분류를 위해 아주 간단한 CNN 모델을 Keras를 이용하여 구축하고자 한다.
    model = model_build_and_compile(num_classes=3)


    # 사용자 입력 파라미터
    batch_size = int(tm.param_info['batch_size'])
    epochs = int(tm.param_info['epoch'])


    low_avg_val_accuracy = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)


    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

    validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


    optimizer = torch.optim.SGD(model.parameters(),
                      lr = 0.0005,
                      momentum=0.9
                      )



    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)







    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    #torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.


    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.


        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.


                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.   '.format(step, len(train_dataloader)))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
            loss, logits = outputs['loss'], outputs['logits']

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.




        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))


        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")



        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs = model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                loss, logits = outputs['loss'], outputs['logits']

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.


            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            flat_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

            total_eval_accuracy += flat_accuracy


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        if low_avg_val_accuracy < avg_val_accuracy:
          low_avg_val_accuracy = avg_val_accuracy
          print('val_accuracy 가 최고 갱신')




          model.save_pretrained(f"./meta_data/pytorch_model_test.bin")

        else:
          print('최고 갱신 못했어요!')


        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.


        print("  Validation Loss: {0:.2f}".format(avg_val_loss))


        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
            }
        )

    print("")
    print("Training complete!")









    ###########################################################################
    ## 플랫폼 시각화
    ###########################################################################
    '''
    plot_metrics(tm, history, model, X_test_cnn, Y_test_cnn)
    '''


    ###########################################################################
    ## 학습 모델 저장
    ###########################################################################

    # logging.info('[hunmin log] tm.model_path : {}'.format(tm.model_path))
    # model.save(os.path.join(tm.model_path, 'cnn_model.h5'))

    # # 저장 파일 확인
    # list_files_directories(tm.model_path)

    # logging.info('[hunmin log]  the finish line of the function [exec_train]')



def exec_init_svc(im):

    logging.info('[hunmin log] im.model_path : {}'.format(im.model_path))

    # 저장 파일 확인
    list_files_directories(im.model_path)

    ###########################################################################
    ## 학습 모델 준비
    ###########################################################################



    # 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels = 3,
        output_attentions = False, # 모델이 어탠션 가중치를 반환하는지 여부.
        output_hidden_states = False, # 모델이 all hidden-state를 반환하는지 여부.
    )
    # load the model
    model.load_state_dict(torch.load('./meta_data/pytorch_model.bin'))

    return {'model' : model}



def exec_inference(df, params, batch_id):

    ###########################################################################
    ## 4. 추론(Inference)
    ###########################################################################

    logging.info('[hunmin log] the start line of the function [exec_inference]')

    ## 학습 모델 준비
    model = params['model']
    print(model)
    logging.info('[hunmin log] model.summary() :')




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")




    input_ids, attention_masks, labels = get_input_mask_label(df, tokenizer)


    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    test_dataloader = DataLoader(
            test_dataset,
            batch_size = 1
        )


    model.eval()

    total_eval_accuracy = 0




    y_an = []
    y_pred = []

    for batch in test_dataloader:



      # Unpack this training batch from our dataloader.
      #
      # As we unpack the batch, we'll also copy each tensor to the GPU using
      # the `to` method.
      #
      # `batch` contains three pytorch tensors:
      #   [0]: input ids
      #   [1]: attention masks
      #   [2]: labels
      b_input_ids = batch[0].cuda()
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)

      # Tell pytorch not to bother with constructing the compute graph during
      # the forward pass, since this is only needed for backprop (training).
      with torch.no_grad():

          # Forward pass, calculate logit predictions.
          # token_type_ids is the same as the "segment ids", which
          # differentiates sentence 1 and 2 in 2-sentence tasks.
          # The documentation for this `model` function is here:
          # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
          # Get the "logits" output by the model. The "logits" are the output
          # values prior to applying an activation function like the softmax.
          outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
          loss, logits = outputs['loss'], outputs['logits']

      # Accumulate the test loss.
      # total_eval_loss += loss.item()

      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()


      # Calculate the accuracy for this batch of test sentences, and
      # accumulate it over all batches.




      pred_flat = np.argmax(logits, axis=1).flatten()
      labels_flat = label_ids.flatten()

      y_an.append(labels_flat.item())
      y_pred.append(pred_flat.item())
      # print(len(y_an))


    print('모델이 예측한 값은?? ',y_pred[0])



    # inverse transform
    result = {'inference' : y_pred[0]}
    logging.info('[hunmin log] result : {}'.format(result))

    return result



# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))



###########################################################################
## exec_train(tm) 호출 함수
###########################################################################



def model_build_and_compile(num_classes =3):
    #모델 구축

    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels = num_classes,
        output_attentions = False, # 모델이 어탠션 가중치를 반환하는지 여부.
        output_hidden_states = False, # 모델이 all hidden-state를 반환하는지 여부.
    )
    logging.info('[hunmin log] model.summary() :')
    print(model)



    return model


def get_input_mask_label(df, tokenizer):

  sentences = df.sentence.values
  labels = df.label.astype(int).values
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []


  for sent in sentences:
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )

      # Add the encoded sentence to the list.
      input_ids.append(encoded_dict['input_ids'])

      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)

  # Print sentence 0, now as a list of IDs.
  print('Original: ', sentences[0])
  print('Token IDs:', input_ids[0])
  return input_ids, attention_masks, labels




def get_input_mask_label_infer(text, tokenizer):

  sentences = np.array([text])

  labels = np.zeros((1, 1)).astype(int)
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []


  for sent in sentences:
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )

      # Add the encoded sentence to the list.
      input_ids.append(encoded_dict['input_ids'])

      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)

  # Print sentence 0, now as a list of IDs.
  print('Original: ', sentences[0])
  print('Token IDs:', input_ids[0])
  return input_ids, attention_masks, labels

