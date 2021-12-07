import csv
import logging

# import numpy as np
import torch
from torch.utils.data.dataset import Dataset

# class RawDataset(torch.utils.data.IterableDataset):
#     r"""
#     An iterable dataset to save the data. This dataset supports multi-processing
#     to load the data.
#     Arguments:
#         iterator: the iterator to read data.
#         num_lines: the number of lines read by the individual iterator.
#     """
#     def __init__(self, iterator, num_lines):
#         super(Dataset, self).__init__()
#         self._num_lines = num_lines
#         self._iterator = iterator
#         self._setup = False

#     def _setup_iterator(self):
#         r"""
#         _setup_iterator() function assign the starting line and the number
#         of lines to read for the individual worker. Then, send them to the iterator
#         to load the data.
#         If worker info is not avaialble, it will read all the lines across epochs.
#         """
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info:
#             chunk = int(self._num_lines / worker_info.num_workers)
#             start = chunk * worker_info.id
#             read = chunk
#             if worker_info.id == worker_info.num_workers - 1:
#                 # The last worker needs to pick up some extra lines
#                 # if the number of lines aren't exactly divisible
#                 # by the number of workers.
#                 # Each epoch we loose an 'extra' number of lines.
#                 extra = self._num_lines % worker_info.num_workers
#                 read += extra
#         else:
#             start = 0
#             read = self._num_lines
#         self._iterator = self._iterator(start, read)

class QQPDataset(Dataset):
    def __init__(self, tokenizer, filename,
                 max_length=512, device='cuda',
                 is_inference=False, load_noise_data=False, is_toy=False):
        self.tokenizer = tokenizer
        self.filename = filename
        self.max_length = max_length
        self.device = device
        self.is_inference = bool(is_inference)
        self.is_toy = is_toy

        self.load_dataset(noised=load_noise_data)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_ids = self.input_ids[idx, :].to(self.device)
        samples = {
            'input_ids': input_ids,
        }
        if self.is_inference is False:
            samples['attention_mask'] = \
                    self.attention_mask[idx, :]
            samples['labels'] = self.labels[idx, :]
        return samples

    def load_dataset(self, noised=False):
        filename = self.filename
        if noised is True:
            filename += '.0'
        logging.info("Loading data from {}".format(filename))

        data = []
        with open(filename) as f:
            reader = csv.reader(f)
            for corrupted, sentence in reader:
                data.append([corrupted, sentence])
                if self.is_toy is True:
                    break

        tokens_list, labels_list = [], []
        for corrupted, sentence in data:
            tokens, labels = self.formatting(corrupted, sentence)
            tokens_list.append(tokens)
            labels_list.append(labels)
        sentences = [self.tokenizer.decode(tokens)
                     for tokens in tokens_list]
        encodings = self.tokenizer(
            sentences, return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_length)

        # print('SENTENCES: ' + sentences)
        # encodings = self.tokenizer(sentences)
        # print(encodings)

        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

        if self.is_inference is False:
            self.labels = torch.tensor(labels_list, dtype=torch.long)

    def formatting(self, input_text, target_text):
        input_tokens = self.tokenizer.encode(input_text)
        target_tokens = self.tokenizer.encode(target_text)

        tokens = [self.tokenizer.bos_token_id] + input_tokens \
            + [self.tokenizer.sep_token_id] + target_tokens \
            + [self.tokenizer.eos_token_id]

        labels = [-100] * (len(input_tokens) + 2) \
            + target_tokens + [self.tokenizer.eos_token_id] \
            + [-100] * (self.max_length - len(tokens))
        labels = labels[:self.max_length]
        return tokens, labels
