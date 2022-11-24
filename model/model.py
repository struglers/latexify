
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from model.formula_encoder import FormulaEncoder
from model.coordinate_encoder import CoordinateEncoder
from model.symbol_encoder import SymbolEncoder
from model.ggnn import GatedGNN
from model.decoder import AttentionDecoder

INIT = 1e-2


class Im2LatexModel(nn.Module):
    def __init__(self, out_size, emb_size, dec_rnn_h,
                 enc_out_dim=512, add_pos_feat=False, dropout=0.):
        super(Im2LatexModel, self).__init__()

        self.formula_encoder = FormulaEncoder(add_pos_feat=add_pos_feat)
        self.coordinate_encoder = CoordinateEncoder()
        self.symbol_encoder = SymbolEncoder()
        self.ggnn = GatedGNN()
        self.attention_decoder = AttentionDecoder(out_size, emb_size, dec_rnn_h,
                                                  enc_out_dim=enc_out_dim,
                                                  dropout=dropout)

    def forward(self, formula_imgs, coordinates, symbols, edge_indices, seq_lens, formulas, epsilon=1.):
        """args:
        inputs:
        - formula_imgs: [N, 1, H, W]
        - coodinates: [N, 4, L]
        - symbols: [L', 1, 32, 32]
        - edge_indices: [2, 2*E']
        - seq_lens: [N]
        where
          N -> Batch size
          H -> Height of formula image
          W -> Width of formula image
          L -> Number of symbols in the image with the largest number of symbols
          L'-> Sum of all symbols in the batch
          E'-> Number of all edges in the batch of graphs
        formulas: [N, MAX_LEN]
        epsilon: probability of the current time step to
                 use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        formula_encodings = self.formula_encoder(formula_imgs)      # [N, H', D2]
        coord_embs = self.coordinate_encoder(coordinates, seq_lens) # [L', 256]
        symbol_embs = self.symbol_encoder(symbols)                  # [L', 256]
        concat_embs = torch.cat([coord_embs, symbol_embs], dim=1)   # [L', 512]
        symbol_encodings = self.ggnn(concat_embs, edge_indices)     # [L', 512]

        # pad and convert shape appropriately
        encodings_list = torch.split(symbol_encodings, seq_lens.tolist())
        symbol_encodings = pad_sequence(encodings_list, batch_first=True) # [N, L, 512], where L=max(seq_lens)

        encodings = torch.cat([formula_encodings, symbol_encodings], dim=1) # [N, H'+L, 512]
        logits = self.attention_decoder(encodings, formulas, epsilon=epsilon)

        return logits

