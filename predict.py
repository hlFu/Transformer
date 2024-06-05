import argparse

import torch

from transformer.data.data import Multi30kDataUtils
from transformer.model.transformer import Transformer


def predict(model_state_path, sequence, device):
    print('Source:', sequence)
    state = torch.load(model_state_path)
    datautil = Multi30kDataUtils(state['configs']['data_root_path'])
    src = datautil.get_indexes_for_single(sequence, 'src').unsqueeze(0).to(device)
    print('Source indexes:', src)

    configs = state['configs']
    net = Transformer(configs['head'], configs['d_model'], configs['sequence_max_len'],
                      configs['transformer_block_num'], datautil.get_src_pad_idx(),
                      datautil.get_trg_pad_idx(), len(datautil.get_vocab()[0]),
                      len(datautil.get_vocab()[1]), configs['ffn_hidden'],
                      configs['dropout'], device).to(device)
    net.load_state_dict(state['model_state_dict'])
    net.eval()

    trg = torch.tensor([datautil.get_trg_bos_idx()], device=device).unsqueeze(0)

    predict_idxes = []
    for i in range(1, 128):
        output = net(src, trg)
        predict_next_idx = torch.argmax(output, dim=2)[0, -1]
        if predict_next_idx.item() == datautil.get_trg_pad_idx() or predict_next_idx.item() == datautil.get_trg_eos_idx():
            break
        predict_idxes.append(predict_next_idx.item())
        trg = torch.cat([trg, predict_next_idx.unsqueeze(0).unsqueeze(0)], dim=0)

    predict_tokens = datautil.get_vocab()[1].lookup_tokens(predict_idxes)

    print('Target: ', predict_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_save_path')
    parser.add_argument('-s', '--sequence')
    parser.add_argument('-dc', '--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    predict(args.model_save_path, args.sequence, device)
