import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm, trange
import os
from transformers import *
from sklearn.metrics import f1_score
from model import Model
from transformers import BertTokenizer
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from pdb import set_trace

# logging.basicConfig(level=logging.ERROR)


train_left = pd.read_csv('./data/train/train.query.tsv', sep='\t', header=None)
train_left.columns = ['id', 'q1']
train_right = pd.read_csv('./data/train/train.reply.tsv', sep='\t', header=None)
train_right.columns = ['id', 'id_sub', 'q2', 'label']
df_train = train_left.merge(train_right, how='left')
df_train['q2'] = df_train['q2'].fillna('好的')
test_left = pd.read_csv('./data/test/test.query.tsv', sep='\t', header=None, encoding='gbk')
test_left.columns = ['id', 'q1']
test_right = pd.read_csv('./data/test/test.reply.tsv', sep='\t', header=None, encoding='gbk')
test_right.columns = ['id', 'id_sub', 'q2']
df_test = test_left.merge(test_right, how='left')


PATH = './'
BERT_PATH = './'
WEIGHT_PATH = './'
MAX_SEQUENCE_LENGTH = 100
input_categories = ['q1', 'q2']
output_categories = 'label'

print('train shape =', df_train.shape)
print('test shape =', df_test.shape)


def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       # truncation=True
                                       )

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, answer, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.q1, instance.q2

        ids_q, masks_q, segments_q = \
            _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30, 60):
        tres = i / 100
        y_pred_bin = (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t

def get_acc(y_true, y_pred):
    correct = 0
    total = 0
    for i, label in enumerate(y_true):
        pred = y_pred[i]

        correct += int(pred == label)
        total += 1
    return correct/total


class Example(object):
    def __init__(self,
                 qid,
                 question,
                 reply,
                 label=None):
        self.q_id = qid
        self.question = question
        self.reply = reply
        self.label = label


class Feature(object):
    def __init__(self,
                 q_id,
                 example_index,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 label=None):
        self.qid = q_id,
        self.example_index = example_index
        self.doc_tokens = doc_tokens,
        self.doc_input_ids = doc_input_ids,
        self.doc_input_mask = doc_input_mask,
        self.doc_segment_ids = doc_segment_ids,
        self.label = label


def read_examples(input_data, columns=None):
    examples = []
    for _, instance in tqdm(input_data.iterrows()):
        q_id, text_a, text_b, label = instance.id, instance.q1, instance.q2, instance.label
        examples.append(Example(qid=q_id,
                                question=text_a,
                                reply=text_b,
                                label=label))
    return examples


def convert_example2feature(examples, tokenizer, max_seq_length=512):
    features = []
    cnt_len = []
    for example_index, example in enumerate(tqdm(examples)):
        question_token = ["[CLS]"] + tokenizer.tokenize(example.question) + ["[SEP]"]
        # all_doc_tokens = []
        reply_token = tokenizer.tokenize(example.reply)
        all_doc_tokens = question_token + reply_token
        if len(all_doc_tokens) > max_seq_length - 1:
            all_doc_tokens = all_doc_tokens[:max_seq_length - 1]
        all_doc_tokens.append("[SEP]")
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(question_token)
        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))
        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_segment_ids.append(0)
            doc_input_mask.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        features.append(Feature(q_id=example.q_id,
                                example_index=example_index,
                                doc_tokens=all_doc_tokens,
                                doc_input_ids=doc_input_ids,
                                doc_segment_ids=doc_segment_ids,
                                doc_input_mask=doc_input_mask,
                                label=example.label))
        cnt_len.append(sum(doc_input_mask))
    print(max(cnt_len))
    cnt_len = np.array(cnt_len)

    return features

model_path = r"E:\DATA\bert_pretrained\chinese_wwm_ext_pytorch"
model_path = r"/DATA/disk1/baijinguo/BERT_Pretrained/chinese_wwm_ext_pytorch"
batch_size = 32
gradient_accumulation_steps = 1
global_step = 0
VERBOSE_STEP = 50
epochs = 5
total_loss = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained(model_path)

train_examples = read_examples(df_train)
train_features = convert_example2feature(train_examples, tokenizer)

all_input_ids = torch.tensor([f.doc_input_ids[0] for f in train_features], dtype=torch.long).to(device)
all_input_mask = torch.tensor([f.doc_input_mask[0] for f in train_features], dtype=torch.long).to(device)
all_segment_ids = torch.tensor([f.doc_segment_ids[0] for f in train_features], dtype=torch.long).to(device)
all_example_id = torch.tensor([f.example_index for f in train_features], dtype=torch.long).to(device)
all_labels = torch.tensor([f.label for f in train_features], dtype=torch.long).to(device)


max_len = int((all_input_mask > 0).long().sum(dim=1).max())
dataset = TensorDataset(all_input_ids[:, :max_len], all_input_mask[:, :max_len], all_segment_ids[:, :max_len],
                        all_example_id, all_labels)
num = len(all_input_ids)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(num*0.8), num-int(num*0.8)])

train_sampler = SequentialSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)
all_results = []
model = Model.from_pretrained(model_path)
model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=5*1e-5, eps=1e-8)


for epoch in trange(epochs):
    model.train()

    for batch in tqdm(train_dataloader):
        input_ids, input_mask, segment_ids, labels = batch[0], batch[1], batch[2], batch[4]
        loss, logits = model(input_ids, segment_ids, input_mask, labels=labels)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        total_loss += loss
        loss.backward()
        if (global_step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
        global_step += 1

        if global_step % VERBOSE_STEP == 0:
            print("-- In Epoch{} --{} step -- LOSS:{} ".format(epoch, global_step, total_loss/VERBOSE_STEP))
            total_loss = 0
    model.eval()
    pred = []
    true_label = []
    pred_logits = []
    test_loss = 0
    cnt = 0
    for batch in tqdm(val_dataloader):
        input_ids, input_mask, segment_ids, labels = batch[0], batch[1], batch[2], batch[4]
        loss, logits = model(input_ids, segment_ids, input_mask, labels=labels)
        test_loss += loss
        cnt += 1
        pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        pred_logits.extend(logits.detach().cpu().numpy()[:, 1].tolist())
        true_label.extend(labels.cpu().numpy().tolist())
    f1, t = search_f1(true_label, np.array(pred_logits))
    acc = get_acc(true_label, pred_logits)
    print("acc: {}".format(acc))
    print("loss: {}".format(test_loss/cnt))