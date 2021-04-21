import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

from model import Model
from utils.data_utils import SequenceLabelingDataset, glue_processor, prepare_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, data_raw, labels):
    model.eval()
    test_data = SequenceLabelingDataset(data_raw, annotated=False)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=test_data.collate_fn)
    preds = []
    epoch_pbar = tqdm(test_dataloader, desc="Prediction", disable=False)
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if b != None and not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, label_mask = batch
        with torch.no_grad():
            output = model(input_ids, segment_ids, input_mask)
        output = output.argmax(dim=2)
        output = output.tolist()
        label_mask = label_mask.tolist()
        # 一个函数　输入是output和label_mask，输出是preds
        out_label = align_predictions(output,label_mask,labels)
        preds = preds + out_label
        epoch_pbar.update(1)
    epoch_pbar.close()
    sentences = [data.words for data in data_raw]
    write_to_file(sentences, preds)

def align_predictions(preds,label_mask,id_to_label):
    res = []
    for pred,mask in zip(preds,label_mask):
        temp = []
        for p,m in zip(pred,mask):
            if m == 1:
                temp.append(id_to_label[p])
        s = ' '.join(temp)
        res.append(s)
    return res

def write_to_file(sentences, preds):
    import csv
    with open('preds.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('label', 'input_text'))
        for sentence, pred in zip(sentences, preds):
            writer.writerow((pred, sentence))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    test_examples = processor.get_predict_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)

    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.num_labels = len(labels)
    model = Model(model_config)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    predict(model, test_data_raw, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='sf', type=str)
    parser.add_argument("--data_dir", default='data/atis/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='outputs/model_best.bin', type=str)
    parser.add_argument("--max_seq_len", default=60, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
