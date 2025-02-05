import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from tqdm import tqdm
import json
from nltk import word_tokenize
from univ.rift import tokenizers

try:
    import cPickle as pickle
except ImportError:
    import pickle


# Modified code taken from https://github.com/dongxinshuai/RIFT-NeurIPS2021
def process_snli_files(path: str, filename: str):
    """
    Converts the given data saved in a CSV file to the format expected when loading SNLI data.

    :param path: The path to the CSV file.
    :param filename: The name of the new file.
    """

    df = pd.read_csv(path, index_col=0)
    prems, hypos, one_hot_labels = df['premise'], df['hypos'], df['labels']
    all_prems, all_hypos, all_labels = [], [], []
    for idx, label in enumerate(one_hot_labels):
        all_prems.append(' '.join(word_tokenize(prems[idx])))
        all_hypos.append(' '.join(word_tokenize(hypos[idx])))
        all_labels.append(label)
    f = open(filename, 'wb')
    print(len(all_prems))
    print(all_prems[0], all_hypos[0], all_labels[0])
    saved = {}
    saved['train_perms'] = all_prems
    saved['train_hypos'] = all_hypos
    saved['train_labels'] = all_labels
    pickle.dump(saved, f)
    f.close()


def read_snli_files(opt, filetype):
    def label_switch(str):
        if str == "entailment":
            return [1, 0, 0]
        if str == "contradiction":
            return [0, 1, 0]
        if str == "neutral":
            return [0, 0, 1]
        raise NotImplementedError

    split = filetype
    totals = {'train': 550152, 'dev': 10000, 'test': 10000}
    all_prem = []
    all_hypo = []
    all_labels = []

    fn = os.path.join(opt['snli_file'].format(split))
    with open(fn) as f:
        for line in tqdm(f, total=totals[split]):
            example = json.loads(line)
            prem, hypo, gold_label = example['sentence1'], example['sentence2'], example['gold_label']
            try:
                one_hot_label = label_switch(gold_label)
                prem = ' '.join(word_tokenize(prem))
                hypo = ' '.join(word_tokenize(hypo))

                all_prem.append(prem)
                all_hypo.append(hypo)
                all_labels.append(one_hot_label)

            except:
                continue
    return all_prem, all_hypo, all_labels


def split_snli_files(opt):
    filename = opt['split_snli_files_path']
    if os.path.exists(filename):
        print('Read processed SNLI dataset')
        f = open(filename, 'rb')
        saved = pickle.load(f)
        f.close()
        train_perms = saved['train_perms']
        train_hypos = saved['train_hypos']
        train_labels = saved['train_labels']
        test_perms = saved['test_perms']
        test_hypos = saved['test_hypos']
        test_labels = saved['test_labels']
        dev_perms = saved['dev_perms']
        dev_hypos = saved['dev_hypos']
        dev_labels = saved['dev_labels']
    else:
        print('Processing SNLI dataset')
        train_perms, train_hypos, train_labels = read_snli_files(opt, 'train')
        dev_perms, dev_hypos, dev_labels = read_snli_files(opt, 'dev')
        test_perms, test_hypos, test_labels = read_snli_files(opt, 'test')
        f = open(filename, 'wb')
        saved = {}
        saved['train_perms'] = train_perms
        saved['train_hypos'] = train_hypos
        saved['train_labels'] = train_labels
        saved['test_perms'] = test_perms
        saved['test_hypos'] = test_hypos
        saved['test_labels'] = test_labels
        saved['dev_perms'] = dev_perms
        saved['dev_hypos'] = dev_hypos
        saved['dev_labels'] = dev_labels
        pickle.dump(saved, f)
        f.close()
    return train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels


class SnliData(torch.utils.data.Dataset):

    def __init__(self, opt, perm, hypo, y, tokenized_subs_dict, seq_max_len, tokenizer, given_class):
        self.opt = opt
        self.perm = perm.copy()
        self.hypo = hypo.copy()
        self.y = y.copy()
        self.tokenized_subs_dict = tokenized_subs_dict.copy()
        self.seq_max_len = seq_max_len
        self.tokenizer = tokenizer
        self.max_substitution_num = 10

        if given_class is not None:
            permperm = []
            hypohypo = []
            yy = []
            for i, label in enumerate(self.y):
                if self.y[i].argmax() == given_class:
                    permperm.append(self.perm[i])
                    hypohypo.append(self.hypo[i])
                    yy.append(self.y[i])
            self.perm = permperm
            self.hypo = hypohypo
            self.y = yy

    def transform(self, sent, label, text_subs, text_subs_mask, attention_mask):

        return torch.tensor(sent, dtype=torch.long), torch.tensor(label, dtype=torch.long), torch.tensor(text_subs,
                                                                                                         dtype=torch.long), torch.tensor(
            text_subs_mask, dtype=torch.float), torch.tensor(attention_mask, dtype=torch.long)

    def __getitem__(self, index):

        if self.opt['plm_type'] == "bert" or self.opt['plm_type'] == "distilbert":
            input_text = self.perm[index] + " [SEP] " + self.hypo[index]
        elif self.opt['plm_type'] == "roberta":
            input_text = self.perm[index] + "</s>" + self.hypo[index]
        elif self.opt['plm_type'] == 'gpt2':
            input_text = self.perm[index] + "<|endoftext|>" + self.hypo[index]

        encoded = self.tokenizer(input_text, self.seq_max_len)

        sent = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        text_subs = []
        text_subs_mask = []
        for token in sent:
            text_subs_mask.append([])
            text_subs.append([token])

        splited_words = self.hypo[index].split()

        h_start = len(self.perm[index].split()) + 2
        len_h = len(self.hypo[index].split())

        for i in range(h_start, min(self.seq_max_len - 1, h_start + len_h), 1):
            word = splited_words[i - h_start]
            if word in self.tokenized_subs_dict:
                text_subs[i].extend(self.tokenized_subs_dict[word])

        label = self.y[index].argmax()

        for i, x in enumerate(sent):
            text_subs_mask[i] = [1 for times in range(len(text_subs[i]))]

            while (len(text_subs[i]) < self.max_substitution_num):
                text_subs[i].append(0)
                text_subs_mask[i].append(0)

        return self.transform(sent, label, text_subs, text_subs_mask, attention_mask)

    def __len__(self):
        return len(self.y)


def make_given_y_iter_snli(opt, tokenized_subs_dict, tokenizer):
    train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(
        opt)

    seq_max_len = opt['snli_input_max_len']

    train_data_y0 = SnliData(opt, train_perms, train_hypos, np.array(train_labels), tokenized_subs_dict, seq_max_len,
                             tokenizer, given_class=0)
    train_loader_y0 = torch.utils.data.DataLoader(train_data_y0, opt['batch_size'] // opt['label_size'], shuffle=True,
                                                  num_workers=8)

    train_data_y1 = SnliData(opt, train_perms, train_hypos, np.array(train_labels), tokenized_subs_dict, seq_max_len,
                             tokenizer, given_class=1)
    train_loader_y1 = torch.utils.data.DataLoader(train_data_y1, opt['batch_size'] // opt['label_size'], shuffle=True,
                                                  num_workers=8)

    train_data_y2 = SnliData(opt, train_perms, train_hypos, np.array(train_labels), tokenized_subs_dict, seq_max_len,
                             tokenizer, given_class=2)
    train_loader_y2 = torch.utils.data.DataLoader(train_data_y2, opt['batch_size'] // opt['label_size'], shuffle=True,
                                                  num_workers=8)

    # from attack.attack_surface import LMConstrainedAttackSurface
    # attack_surface = LMConstrainedAttackSurface.from_files(opt.substitution_dict_path, opt.snli_lm_file_path)

    test_data = SnliData(opt, test_perms, test_hypos, np.array(test_labels), tokenized_subs_dict, seq_max_len,
                         tokenizer, given_class=None)
    test_loader = torch.utils.data.DataLoader(test_data, opt['test_batch_size'], shuffle=False, num_workers=8)

    return train_loader_y0, train_loader_y1, train_loader_y2, test_loader


def get_substitution_dict(file_path):
    import json
    with open(file_path) as f:
        subs_dict = json.load(f)
    return subs_dict


def get_data_iters(opt):
    tokenizer, substitution_tokenizer = tokenizers.get_tokenizers(opt['plm_type'])

    if opt['plm_type'] == 'bert':
        tokenized_subs_dict_path = opt['bert_tokenized_subs_dict_path']
    elif opt['plm_type'] == 'roberta':
        tokenized_subs_dict_path = opt['roberta_tokenized_subs_dict_path']
    elif opt['plm_type'] == 'gpt2':
        tokenized_subs_dict_path = opt['gpt2_tokenized_subs_dict_path']
    elif opt['plm_type'] == 'distilbert':
        tokenized_subs_dict_path = opt['distilbert_tokenized_subs_dict_path']
    else:
        raise NotImplementedError

    if os.path.exists(tokenized_subs_dict_path):
        f = open(tokenized_subs_dict_path, 'rb')
        saved = pickle.load(f)
        f.close()
        tokenized_subs_dict = saved["tokenized_subs_dict"]
    else:

        subs_dict = get_substitution_dict(opt['substitution_dict_path'])
        tokenized_subs_dict = {}  # key is text word, contents are tokenized substitution words

        # Tokenize syn data
        print("tokenizing substitution words")
        for key in subs_dict:
            if len(subs_dict[key]) != 0:
                # temp = tokenizer.encode_plus(subs_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']
                temp = substitution_tokenizer(subs_dict[key])['input_ids']
                temp = [x[0] for x in temp]
                tokenized_subs_dict[key] = temp

        print("done")

        filename = tokenized_subs_dict_path
        f = open(filename, 'wb')
        saved = {}
        saved['tokenized_subs_dict'] = tokenized_subs_dict
        pickle.dump(saved, f)
        f.close()

    opt['label_size'] = 3
    train_iter_y0, train_iter_y1, train_iter_y2, test_iter = make_given_y_iter_snli(opt, tokenized_subs_dict, tokenizer)
    return (train_iter_y0, train_iter_y1, train_iter_y2), test_iter
