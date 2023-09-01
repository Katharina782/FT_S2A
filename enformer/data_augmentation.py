import torch


def stochastic_reverse_complement(seq, training=None):
    if training:
        # sample from random uniform
        rc = seq[:, :, [3, 2, 1, 0]]
        reverse_bool = torch.randint(low=0, high=2, size=[1]).to(torch.bool)
        rc = torch.flip(rc, dims=[1])
        rc_seq = torch.where(reverse_bool, rc, seq)
        return rc_seq, reverse_bool

    else:
        return seq, False
    


class SwitchReverse():
    """
    Reverse predictions if the inputs were reverse complemented.
    """    
    def call(self, x_reverse):
        seq = x_reverse[0]
        # boolean wether the sequence was reverse complemented
        reverse = x_reverse[1]

        if len(seq.shape) == 3:
            return torch.where(reverse, torch.flip(seq, dims=[1]), seq)# torch.flip(seq, dims=[1]), seq)
        else:
            raise ValueError("Cannot recognize SwitchReverse iput dimensions %d" %len(seq.shape))


    
def shift_sequence(seq, shift, pad_value, shift_left=True):#, shift_right=False):
    pad = pad_value * torch.ones_like(seq[:, 0:torch.abs(shift), :])

    if shift_left:
        sliced_seq = seq[:, shift:, :]
        #print(sliced_seq.shape)
        return torch.cat([sliced_seq, pad], dim=1)
    else: 
        sliced_seq = seq[:, :-shift, :]
        #print(sliced_seq.shape)
        return torch.cat([pad, sliced_seq], dim=1)

class StochasticShift():
    def __init__(self, shift_max:int, pad_value=0):
        self.shift_max = shift_max
        self.pad_value = pad_value
    

    def call(self, seq, training=None):
        if training: 
            # symmetric shift left and right
            augment_shifts = torch.arange(start=-self.shift_max, end=self.shift_max + 1)
            # pick random shift
            shift_i = torch.randint(low=0, high=len(augment_shifts), size=[1])#, dtype=torch.int64)
            shift = augment_shifts[shift_i]
            if shift != 0:
                shift_left = shift < 0
                return shift_sequence(seq, torch.abs(shift), self.pad_value, shift_left=shift_left)
            else: 
                return seq
        else: 
            return seq
        

