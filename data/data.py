from collections import Counter

import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, multi30k
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence

class Multi30kDataUtils:

    def __init__(self, root):
        train, valid = self.get_raw_dataset(root)

        self.de_tokenizer = get_tokenizer('spacy', language='de')
        self.en_tokenizer = get_tokenizer('spacy', language='en')

        self.en_vocab = self.build_vocab(train, self.en_tokenizer, 0)
        self.de_vocab = self.build_vocab(train, self.de_tokenizer, 1)

        self.train = self.preprocess(train, self.en_tokenizer, self.de_tokenizer, self.en_vocab, self.de_vocab)
        self.valid = self.preprocess(valid, self.en_tokenizer, self.de_tokenizer, self.en_vocab, self.de_vocab)

    @staticmethod
    def build_vocab(dataset, tokenizer, column):
        counter = Counter()
        for data in dataset:
            counter.update(tokenizer(data[column]))
        vocab = torchtext.vocab.vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    @staticmethod
    def get_raw_dataset(root):
        return Multi30k(root=root, language_pair=('en', 'de'), split=('train', 'valid'))

    @staticmethod
    def preprocess(dataset, en_tokenzer, de_tokenizer, en_vocab, de_vocab):
        processed_dataset = []
        for raw_en, raw_de in dataset:
            en_tensor = Multi30kDataUtils.preprocess_single(raw_en, en_tokenzer, en_vocab)
            de_tensor = Multi30kDataUtils.preprocess_single(raw_de, de_tokenizer, de_vocab)
            processed_dataset.append((en_tensor, de_tensor))
        return processed_dataset

    @staticmethod
    def preprocess_single(sequence, tokenizer, vocab):
        return torch.tensor([vocab[token] for token in tokenizer(sequence)], dtype=torch.int64)

    def generate_batch(self, is_padding):

        def generate_batch(data_batch):
            en_batch, de_batch = [], []
            for (en_item, de_item) in data_batch:
                en_batch.append(
                    torch.cat([torch.LongTensor([self.en_vocab['<bos>']]), en_item,
                               torch.LongTensor([self.en_vocab['<eos>']])], dim=0))
                de_batch.append(
                    torch.cat([torch.LongTensor([self.de_vocab['<bos>']]), de_item,
                               torch.LongTensor([self.de_vocab['<eos>']])], dim=0))
            if is_padding:
                en_batch = pad_sequence(en_batch, padding_value=self.en_vocab['<pad>'], batch_first=True)
                de_batch = pad_sequence(de_batch, padding_value=self.de_vocab['<pad>'], batch_first=True)

            return en_batch, de_batch

        return generate_batch

    def get_data_loader(self, split, batch_size, is_padding=True):
        if split == 'train':
            return DataLoader(self.train, batch_size=batch_size, shuffle=True, collate_fn=self.generate_batch(is_padding))
        elif split == 'valid':
            return DataLoader(self.valid, batch_size=batch_size, shuffle=True, collate_fn=self.generate_batch(is_padding))

    def get_vocab(self):
        return self.en_vocab, self.de_vocab

    def get_src_pad_idx(self):
        return self.en_vocab['<pad>']

    def get_trg_pad_idx(self):
        return self.de_vocab['<pad>']

    def get_trg_bos_idx(self):
        return self.de_vocab['<bos>']

    def get_trg_eos_idx(self):
        return self.de_vocab['<eos>']

    def get_tokenizer(self):
        return self.en_tokenizer, self.de_tokenizer

    def get_indexes_for_single(self, sequence, language):
        if language == 'src':
            tokenizer = self.en_tokenizer
            vocab = self.en_vocab
        elif language == 'trg':
            tokenizer = self.de_tokenizer
            vocab = self.de_vocab
        else:
            raise RuntimeError()

        return torch.cat([torch.LongTensor([vocab['<bos>']]), self.preprocess_single(sequence, tokenizer, vocab),
                          torch.LongTensor([vocab['<eos>']])], dim=0)

if __name__ == '__main__':
    loader = Multi30kDataUtils('/Users/fhaolin/work/data/multi30k')
    dataloader = loader.get_data_loader('train', 8)
    print(next(iter(dataloader)))