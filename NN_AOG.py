from utils import *
from sym_net import SymbolNet

class NNAOG(nn.Module):
    def __init__(self):
        super(NNAOG, self).__init__()

        self.sym_net = SymbolNet()

    
    def forward(self, img_seq):
        batch_size = img_seq.shape[0]
        max_len = img_seq.shape[1]
        images = img_seq.reshape((-1, img_seq.shape[-3], img_seq.shape[-2], img_seq.shape[-1]))
        probs = self.sym_net(images)
        probs = F.log_softmax(probs, dim = -1)
        probs = probs.reshape((batch_size, max_len, -1))

        mask = torch.zeros_like(probs, device=probs.device)
        digit_pos_list = np.arange(0, max_len, 2)
        op_pos_list = np.arange(1, max_len, 2)
        mask[:, digit_pos_list[:, None], digit_idx_list] = 1.
        if len(op_pos_list) > 0:
            mask[:, op_pos_list[:, None], op_idx_list] = 1. 
        masked_probs = mask * torch.exp(probs)
        masked_probs[mask.bool()] += 1e-12

        return masked_probs
