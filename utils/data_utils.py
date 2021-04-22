import os
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import DataProcessor, logging
from typing import Optional, List

logger = logging.get_logger(__name__)

@dataclass
class InputExample:

    guid: str
    words: List[str]
    labels: Optional[List[str]]


class SfProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train/intent_seq.in")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "train/intent_seq.out")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "dev/intent_seq.out")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "predict")

    def get_labels(self,data_dir):
        """See base class."""
        slot_labels_list = self._read_tsv(os.path.join(data_dir, "vocab/slot_vocab"))
        slot_labels = [label[0] for label in slot_labels_list]
        return slot_labels

    def _create_examples(self, lines_in,lines_out, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i,(line,out) in enumerate(zip(lines_in,lines_out)):

            guid = "%s-%s" % (set_type, i)
            words = line[0][4:].strip()

            labels = None if set_type == "predict" else out[0].strip().split()[1:]
            examples.append(InputExample(guid=guid, words=words,  labels=labels))

        return examples

class TrainingInstance:
    def __init__(self,example,max_seq_len):
        self.words = example.words.split()
        self.labels = example.labels
        self.max_seq_len = max_seq_len

    def make_instance(self,tokenizer,label_map,pad_label_id=-100):
        tokens = []
        label_ids = []
        if self.labels:
            for word, label in zip(self.words, self.labels):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label_map[label]] + [pad_label_id] * (len(word_tokens) - 1))
        else:
            # 预测时，把需要预测的位置置为１
            for word in self.words:
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    label_ids.extend([1] + [0] * (len(word_tokens) - 1))

        # TODO 判断长度越界

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        self.label_ids = [pad_label_id] + label_ids + [pad_label_id]
        self.input_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.segment_id = [0] * len(self.input_ids)
        self.input_mask = [1] * len(self.input_ids)
        padding_length = self.max_seq_len-len(self.input_ids)

        if padding_length > 0:
            self.input_ids = self.input_ids + [0] * padding_length
            self.segment_id = self.segment_id + [0] * padding_length
            self.input_mask = self.input_mask + [0] * padding_length
            self.label_ids = self.label_ids + [pad_label_id] * padding_length



class SequenceLabelingDataset(Dataset):
    def __init__(self,data,annotated=True):
        self.data = data
        self.len = len(data)
        self.annotated = annotated

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx]
    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_ids for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        label_ids = torch.tensor([f.label_ids for f in batch],dtype=torch.long)
        return input_ids,segment_ids,input_mask,label_ids

def prepare_data(examples,max_seq_len,tokenizer,labels):
    label_map = {label:idx for idx,label in enumerate(labels)}
    data = []

    for example in examples:
        instance = TrainingInstance(example,max_seq_len)
        instance.make_instance(tokenizer,label_map)
        data.append(instance)
    return data


glue_processor = {
    'sf': SfProcessor()
}