import torch


def stochastic_reverse_complement(seq, training=None):
    if training:
        # sample from random uniform
        rc = seq[:, :, [3, 2, 1, 0]]
        reverse_bool = torch.randint(low=0, high=2, size=[1]).to(torch.bool)
        # flip the sequence to get the reverse strand
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


    
def shift_sequence(seq, shift, pad_value, shift_left=True):
    # padding the overhanging sequence
    pad = pad_value * torch.ones_like(seq[:, 0:torch.abs(shift), :])

    if shift_left:
        sliced_seq = seq[:, shift:, :] # get subset of sequence
        return torch.cat([sliced_seq, pad], dim=1) # pad the subset
    else: 
        sliced_seq = seq[:, :-shift, :]
        return torch.cat([pad, sliced_seq], dim=1)


# Shift the entire sequence by a number of positions left and right
# How many bp to shift the sequence left and right is sampled from a uniform distribution
# The  maximum and minimum number of positoins to shift is determined by shift_max
class StochasticShift():
    def __init__(self, shift_max:int, pad_value=0):
        self.shift_max = shift_max # maximum number of positions to shift
        self.pad_value = pad_value # pad the border with zeros or another value
    

    def call(self, seq, training=None):
        if training: 
            # symmetric shift left and right
            augment_shifts = torch.arange(start=-self.shift_max, end=self.shift_max + 1)
            # sample numbe of positions to shift form uniform distribution
            shift_i = torch.randint(low=0, high=len(augment_shifts), size=[1])
            shift = augment_shifts[shift_i]

            if shift != 0:
                shift_left = shift < 0
                return shift_sequence(seq, torch.abs(shift), self.pad_value, shift_left=shift_left)
            else: 
                return seq
        else: 
            return seq
        

