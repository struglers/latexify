
import torch.nn as nn

from .formula_encoder import FormulaEncoder
from .decoder import AttentionDecoder

INIT = 1e-2


class Im2LatexModel(nn.Module):
    def __init__(self, out_size, emb_size, dec_rnn_h,
                 enc_out_dim=512, add_pos_feat=False, dropout=0.):
        super(Im2LatexModel, self).__init__()

        self.formula_encoder = FormulaEncoder(add_pos_feat=add_pos_feat)
        self.attention_decoder = AttentionDecoder(out_size, emb_size, dec_rnn_h,
                                                  enc_out_dim=enc_out_dim,
                                                  dropout=dropout)

    def forward(self, formula_imgs, coordinates, symbols, edge_indices, formulas, epsilon=1.):
        """args:
        imgs: [B, C, H, W]
        formulas: [B, MAX_LEN]
        epsilon: probability of the current time step to
                 use the true previous token
        return:
        logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # TODO: Add Symbol Encoder, Coordinate Encoder and GGNN here
        encodings = self.formula_encoder(formula_imgs) # [N, H', D2]
        logits = self.attention_decoder(encodings, formulas, epsilon=epsilon)
        return logits

