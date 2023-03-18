import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask

import pandas as pd
import math
from pcgrad import PCGrad


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        # sentiment layers
        self.sentiment_dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_linear = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        # paraphrase layers
        self.paraphrase_dropout_1 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_dropout_2 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_linear_1 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.paraphrase_linear_2 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.paraphrase_linear_interact = torch.nn.Linear(BERT_HIDDEN_SIZE, 1)
        # similarity layers
        self.similarity_dropout_1 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.similarity_dropout_2 = torch.nn.Dropout(config.hidden_dropout_prob)
        self.similarity_linear_1 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.similarity_linear_2 = torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.similarity_linear_interact = torch.nn.Linear(BERT_HIDDEN_SIZE, 1)



    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        outputs = self.bert(input_ids, attention_mask)  # potentially perform dropout on input_ids
        pooler_output = outputs['pooler_output']

        return pooler_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        output = self.forward(input_ids, attention_mask)
        output = self.sentiment_dropout(output) # dropout added
        sentiment_logits = self.sentiment_linear(output)

        return sentiment_logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        output_1 = self.paraphrase_dropout_1(output_1)
        output_2 = self.paraphrase_dropout_2(output_2)
        output_1 = self.paraphrase_linear_1(output_1)
        output_2 = self.paraphrase_linear_2(output_2)
        output = torch.mul(output_1, output_2) # want to update this to torch.mul(output_1, output_2), and drop the next row
        paraphrase_logit = self.paraphrase_linear_interact(output)
        
        return paraphrase_logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        output_1 = self.similarity_dropout_1(output_1)
        output_2 = self.similarity_dropout_2(output_2)
        if args.share_layers:
            print('NOTE: sharing layers between para and sts')
            output_1 = self.paraphrase_linear_1(output_1)
            output_2 = self.paraphrase_linear_2(output_2)
        else:
            output_1 = self.similarity_linear_1(output_1)
            output_2 = self.similarity_linear_2(output_2)
        output = torch.mul(output_1, output_2)
        similarity_logit = self.similarity_linear_interact(output)

        return similarity_logit




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # create a dataframe for training outcomes for each epoch (requires the pandas and openpyxl packages)
    df = pd.DataFrame(columns = ['epoch', 'train_acc_sst', 'train_acc_para', 'train_acc_sts', 'train_acc', \
                             'dev_acc_sst', 'dev_acc_para', 'dev_acc_sts', 'dev_acc'])
    

    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    # sst data
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # para data
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    # sts data
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # multitask model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        if args.training_method == 'single':
            # sst training
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
            
            # para training
            num_batches = 0
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'],
                                        batch['token_ids_2'], batch['attention_mask_1'],
                                        batch['attention_mask_2'], batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                logits = torch.sigmoid(logits) # sigmoid
                loss = F.l1_loss(logits.view(-1), b_labels) / args.batch_size # L1 loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                # temporary, for experimentation (train on about 8000 data points)
                if args.para_training_cut & num_batches * args.batch_size > 8000:
                    print('NOTE: para training cut.')
                    break
                            
            # sts training
            num_batches = 0
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'],
                                        batch['token_ids_2'], batch['attention_mask_1'],
                                        batch['attention_mask_2'], batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                logits = torch.sigmoid(logits) # sigmoid
                logits = logits.mul(5) # multiply by five to match labels
                loss = F.l1_loss(logits.view(-1), b_labels) / args.batch_size # L1 loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
        
        elif args.training_method in ["multi", 'multi_balanced']:
            # cnt_iter: number of iterations, set equal to the batch number of the smallest dataset
            # cnt_batch_*: the number of batches to train in each iteration for each dataset \
            # taking the ceiling as batch number for the larger datasets, they will need to be 
            # reset before the last batch is reached            
            cnt_iter, cnt_batch_sst, cnt_batch_para, cnt_batch_sts = 0, 0, 0, 0

            # construct tqdm objects and iterators for *_dataloader
            sst_dl, para_dl, sts_dl = tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE), \
            tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE), \
            tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
            sst_iter, para_iter, sts_iter = iter(sst_dl), iter(para_dl), iter(sts_dl)

            # find the number of iterations and the number of batches to train for each dataset during an iteration
            if len(sst_train_dataloader) < len(sts_train_dataloader): # sst is the smallest
                cnt_iter = len(sst_train_dataloader)
                cnt_batch_sst = 1
                cnt_batch_para = math.ceil((len(para_train_dataloader) / cnt_iter))
                cnt_batch_sts = math.ceil((len(sts_train_dataloader) / cnt_iter))
            else: # otherwise sts is the smallest
                cnt_iter = len(sts_train_dataloader)
                cnt_batch_sst = math.ceil((len(sst_train_dataloader) / cnt_iter))
                cnt_batch_para = math.ceil((len(para_train_dataloader) / cnt_iter))
                cnt_batch_sts = 1
            print(cnt_batch_sst, cnt_batch_para, cnt_batch_sts)

            # training by iteration
            num_batch_sst = 0
            num_batch_para = 0
            num_batch_sts = 0
            sst_len = len(sst_train_dataloader)
            para_len = len(para_train_dataloader)
            sts_len = len(sts_train_dataloader)
            sst_reset = False
            para_reset = False
            sts_reset = False

            for iteration in range(cnt_iter):
                loss = 0
                loss_sst = 0
                loss_para = 0
                loss_sts = 0

                # train on sst batch(es)
                if num_batch_sst + cnt_batch_sst > sst_len and sst_reset == False:
                    sst_dl = tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
                    sst_iter = iter(sst_dl)
                    sst_reset = True
                    print('NOTE: sst reset')
                for i in range(cnt_batch_sst):
                    batch = next(sst_iter)
                    b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                    b_ids = b_ids.to(device)
                    b_mask = b_mask.to(device)
                    b_labels = b_labels.to(device)

                    optimizer.zero_grad()
                    logits = model.predict_sentiment(b_ids, b_mask)
                    if args.training_method == 'multi_balanced': # normalize by observation count
                        loss_sst = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / (args.batch_size * cnt_batch_sst)
                        loss += loss_sst
                    else:
                        loss_sst = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                        loss += loss_sst
                num_batch_sst += cnt_batch_sst

                # train on para batch(es)
                if num_batch_para + cnt_batch_para > para_len and para_reset == False:
                    para_dl = tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
                    para_iter = iter(para_dl)
                    para_reset = True
                    print('NOTE: para reset')
                for i in range(cnt_batch_para):
                    batch = next(para_iter)
                    b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'],
                                        batch['token_ids_2'], batch['attention_mask_1'],
                                        batch['attention_mask_2'], batch['labels'])

                    b_ids_1 = b_ids_1.to(device)
                    b_ids_2 = b_ids_2.to(device)
                    b_mask_1 = b_mask_1.to(device)
                    b_mask_2 = b_mask_2.to(device)
                    b_labels = b_labels.to(device)

                    optimizer.zero_grad()
                    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    logits = torch.sigmoid(logits) # sigmoid
                    if args.training_method == 'multi_balanced': # normalize by observation count
                        loss_para = F.l1_loss(logits.view(-1), b_labels) / (args.batch_size * cnt_batch_para) # L1 loss
                        loss += loss_para
                    else:
                        loss_para = F.l1_loss(logits.view(-1), b_labels) / args.batch_size
                        loss += loss_para
                num_batch_para += cnt_batch_para
                
                # train on sts batch(es)
                if num_batch_sts + cnt_batch_sts > sts_len and sts_reset == False:
                    sts_dl = tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
                    sts_iter = iter(sts_dl)
                    sts_reset = True
                    print('NOTE: sts reset')
                for i in range(cnt_batch_sts):
                    batch = next(sts_iter)
                    b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'],
                                        batch['token_ids_2'], batch['attention_mask_1'],
                                        batch['attention_mask_2'], batch['labels'])

                    b_ids_1 = b_ids_1.to(device)
                    b_ids_2 = b_ids_2.to(device)
                    b_mask_1 = b_mask_1.to(device)
                    b_mask_2 = b_mask_2.to(device)
                    b_labels = b_labels.to(device)

                    optimizer.zero_grad()
                    logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                    logits = torch.sigmoid(logits) # sigmoid
                    logits = logits.mul(5) # multiply by five to match labels
                    if args.training_method == 'multi_balanced': # normalize by observation count
                        loss_sts = F.l1_loss(logits.view(-1), b_labels) / (args.batch_size * cnt_batch_sts) # L1 loss
                        loss += loss_sts
                    else:
                        loss_sts = F.l1_loss(logits.view(-1), b_labels) / args.batch_size
                        loss += loss_sts
                num_batch_sts += cnt_batch_sts

                # step
                if args.gradient_surgery:
                    pcg = PCGrad(optimizer)
                    losses = [loss_sst, loss_para, loss_sts]
                    pcg.pc_backward(losses) # calculate the gradient
                    pcg.step()  # apply gradient step
                    optimizer = pcg._optim
                else:
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        else:
            print('ERROR: invalid parameter value for --training_method')
            raise ValueError

        train_loss = train_loss / (num_batches)

        train_acc_para, _, _, train_acc_sst, _, _, train_acc_sts, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_acc_para, _, _, dev_acc_sst, _, _, dev_acc_sts, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        train_acc = np.average([train_acc_sst, train_acc_para, train_acc_sts])
        dev_acc = np.average([dev_acc_sst, dev_acc_para, dev_acc_sts])
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        df = df.append({'epoch' : epoch, 'train_acc_sst' :train_acc_sst, 'train_acc_para' : train_acc_para,\
                        'train_acc_sts' : train_acc_sts, 'train_acc' : train_acc, 'dev_acc_sst' : dev_acc_sst,\
                            'dev_acc_para' : dev_acc_para, 'dev_acc_sts' : dev_acc_sts, 'dev_acc' : dev_acc}, ignore_index = True)

    # Additional training
    if args.option == 'finetune' and args.additional_pretrain and args.additional_epoch > 0:
        print('Note: running additional training with Bert parameters fixed...')

        # create new optimizer with new learning rate
        lr = args.lr_additional
        optimizer = AdamW(model.parameters(), lr=lr)

        for param in model.bert.parameters():
            param.requires_grad = False
        
        for epoch in range(args.additional_epoch):
            model.train()
            train_loss = 0
            num_batches = 0

            # COPIED FROM "single" above, room for code improvement
            # sst training
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
            
            # para training
            num_batches = 0
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'],
                                        batch['token_ids_2'], batch['attention_mask_1'],
                                        batch['attention_mask_2'], batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                logits = torch.sigmoid(logits) # sigmoid
                loss = F.l1_loss(logits.view(-1), b_labels) / args.batch_size # L1 loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                # temporary, for experimentation (train on about 8000 data points)
                if args.para_training_cut & num_batches * args.batch_size > 8000:
                    print('NOTE: para training cut.')
                    break
                            
            # sts training
            num_batches = 0
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids_1, b_ids_2, b_mask_1, b_mask_2, b_labels = (batch['token_ids_1'],
                                        batch['token_ids_2'], batch['attention_mask_1'],
                                        batch['attention_mask_2'], batch['labels'])

                b_ids_1 = b_ids_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                logits = torch.sigmoid(logits) # sigmoid
                logits = logits.mul(5) # multiply by five to match labels
                loss = F.l1_loss(logits.view(-1), b_labels) / args.batch_size # L1 loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            train_acc_para, _, _, train_acc_sst, _, _, train_acc_sts, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
            dev_acc_para, _, _, dev_acc_sst, _, _, dev_acc_sts, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

            train_acc = np.average([train_acc_sst, train_acc_para, train_acc_sts])
            dev_acc = np.average([dev_acc_sst, dev_acc_para, dev_acc_sts])
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
            df = df.append({'epoch' : epoch, 'train_acc_sst' :train_acc_sst, 'train_acc_para' : train_acc_para,\
                        'train_acc_sts' : train_acc_sts, 'train_acc' : train_acc, 'dev_acc_sst' : dev_acc_sst,\
                            'dev_acc_para' : dev_acc_para, 'dev_acc_sts' : dev_acc_sts, 'dev_acc' : dev_acc}, ignore_index = True)

    filename = f"training_results_by_epoch_{args.epoch_results_filename}.xlsx"
    df.to_excel(filename)



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    
    # training results by epoch filename
    parser.add_argument("--epoch_results_filename", help='name for training results by epoch excel file', type=str, default = '')
    # share layers
    parser.add_argument("--share_layers", help = 'share layers between para and sts', action='store_true')
    # cut down on para training
    parser.add_argument("--para_training_cut", help = 'cut down on para training', action='store_true')
    # multitask training switch
    parser.add_argument("--training_method", help = 'single-task (single) or multi-task (multi, multi_balanced)', default='single')
    # gradient surgery switch
    parser.add_argument("--gradient_surgery", help = 'apply gradient surgery for multitask training', action='store_true')
    # additional pretrain
    parser.add_argument("--additional_pretrain", help = 'additional training with Bert parameters locked', action='store_true')
    # number of additional pretrain epochs
    parser.add_argument("--additional_epoch", help='number of epochs to run additional pre-training for', type=int, default=0)
    # additional pretrain learning rate
    parser.add_argument("--lr_additional", help='learning rate for additional pre-training', type=float, default=1e-3)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
