import argparse
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchinfo import summary
from torchtext.data.metrics import bleu_score

import transformer.config as config
from transformer.data.data import Multi30kDataUtils
from transformer.model.transformer import Transformer


def load_configs():
    # Change to config.standard_configs if needs to adopt the configs from the original transformer
    configs = config.test_configs

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_save_path', required=False)
    parser.add_argument('-d', '--data_root_path', required=False)
    parser.add_argument('-dc', '--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.model_save_path is not None:
        configs['model_save_path'] = args.model_save_path
    if args.data_root_path is not None:
        configs['data_root_path'] = args.data_root_path

    configs['device'] = args.device
    return configs


def train(configs):
    datautil = Multi30kDataUtils(configs['data_root_path'])
    dataloader = datautil.get_data_loader('train', configs['batch_size'])

    if configs['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        

    net = Transformer(configs['head'], configs['d_model'], configs['sequence_max_len'],
                      configs['transformer_block_num'], datautil.get_src_pad_idx(),
                      datautil.get_trg_pad_idx(), len(datautil.get_vocab()[0]),
                      len(datautil.get_vocab()[1]), configs['ffn_hidden'],
                      configs['dropout'], device).to(device)
    print(net)
    summary(net,
            input_size=[(configs['batch_size'], configs['sequence_max_len']),
                        (configs['batch_size'], configs['sequence_max_len'])],
            dtypes=[torch.int64, torch.int64],
            col_names=("input_size", "output_size", "num_params"),
            depth=10)

    lr = configs['lr'] if 'lr' in configs else math.sqrt(configs['d_model']) * math.pow(configs['warmup_steps'], -1.5)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, betas=configs['adam_betas'],
                                 weight_decay=configs['weight_decay'], eps=configs['adam_eps'])

    criterion = nn.CrossEntropyLoss()

    step = 0
    cumulative_loss = []
    steps = []
    window_loss = 0
    for epoch in range(1, configs['epochs'] + 1):
        net.train()
        for src, trg in dataloader:
            src.to(device)
            trg.to(device)
            step += 1
            optimizer.zero_grad()
            out = net(src, trg[:, :-1])
            loss = criterion(out.transpose(1, 2), trg[:, 1:])
            loss.backward()
            optimizer.step()

            window_loss += loss.item()
            if step % configs['show_loss_steps'] == 0:
                cumulative_loss.append(window_loss / configs['show_loss_steps'])
                steps.append(step)
                show_loss(cumulative_loss, steps)
                window_loss = 0

            if 0 < configs['max_steps'] <= step:
                break

            adjust_optim(optimizer, step + 1, configs)

        valid_loss, bleu = evaluate(datautil, net, criterion, device)
        print('Epoch: {}, Validation loss: {}, BLEU score: {}'.format(epoch, valid_loss, bleu))

        if 0 < configs['max_steps'] <= step:
            break

    if 'model_save_path' in configs:
        data_params = {
            'src_pad_idx': datautil.get_src_pad_idx(),
            'trg_pad_idx': datautil.get_trg_pad_idx(),
            'src_vocab_size': len(datautil.get_vocab()[0]),
            'trg_vocab_size': len(datautil.get_vocab()[1])
        }
        torch.save({
            'model_state_dict': net.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': window_loss / configs['show_loss_steps'],
            'configs': configs,
            'data_params': data_params
        }, configs['model_save_path'])

        print('model saved')


def show_loss(loss, steps):
    print('loss: ', loss[-1])
    plt.plot(steps, loss)
    plt.title("Loss - train steps curve")
    plt.xlabel("Step num")
    plt.ylabel("Loss")
    plt.show()


def evaluate(datautil, net, criterion, device):
    cumulative_loss = 0
    predict_tokens = []
    trg_tokens = []
    dataloader = datautil.get_data_loader('valid', 16)
    src_vocab, trg_vocab = datautil.get_vocab()
    net.eval()

    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src.to(device)
            trg.to(device)
            predict = net(src, trg[:, :-1])

            predict_idx = torch.argmax(predict, dim=2).tolist()
            filtered_predict_idx = [list(filter(datautil.get_trg_pad_idx().__ne__, idxes)) for idxes in predict_idx]
            predict_tokens.extend(list(map(lambda idx:trg_vocab.lookup_tokens(idx), filtered_predict_idx)))

            trg_idx = trg[:, 1:].tolist()
            filtered_trg_idx = [list(filter(datautil.get_trg_pad_idx().__ne__, idxes)) for idxes in trg_idx]
            trg_tokens.extend(list(map(lambda idx:[[token] for token in trg_vocab.lookup_tokens(idx)], filtered_trg_idx)))

            loss = criterion(predict.transpose(1, 2), trg[:, 1:])
            cumulative_loss += loss.item()

    return cumulative_loss / i, bleu_score(predict_tokens, trg_tokens)


def adjust_optim(optimizer, step, configs):
    if 'lr' in configs:
        return

    optimizer.param_groups[0]['lr'] = (
            math.sqrt(configs['d_model']) * min(math.pow(step, -0.5), step * math.pow(configs['warmup_steps'], -1.5)))


if __name__ == '__main__':
    configs = load_configs()
    train(configs)

