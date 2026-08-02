import torch

from torch import nn
from torch.nn import functional as F 

device = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################################################
class FSQLatent(nn.Module):

##############################################################################################
    def __init__(self, 
                 input_size : int,
                 output_size : int,
                 L : int,
                 activation : callable = None,
                 with_rescale : bool = True,
                 *args, 
                 **kwargs
                ):
        
        super(FSQLatent, self).__init__()

        # In the original paper, input and output are the same
        # because they use directly the ouput of a linear layer.
        # In our case, we are gonna use a projection to the inteded
        # output size directly in this class
        self.input_size = input_size
        self.output_size = output_size
        self.with_rescale = with_rescale

        # L can be 2 things: a single value (then all the output dimensions will use
        # the same value), or an array specifying the size L for each dimension
        # of the output dimension. In case it is an array, it's length must be the same
        # as the output size
        if type(L) == list:
            assert len(L) == self.output_size, "If we want to specify size for each dimension, L must be == to output size"
            self.Ls = L
        else:
            self.Ls = [L for _ in range(self.output_size)]
        
        self.Ls = torch.tensor(self.Ls).to(device)


        # Get the max to scale the Ls afterwards
        self.max_Ls = []
        for l in self.Ls:
            if l % 2 == 0:
                self.max_Ls.append(torch.round(l/2) - 1)
            else:
                self.max_Ls.append(torch.round(l/2))
        
        self.max_Ls = torch.tensor(self.max_Ls).to(device)

        # Linear projection to the desired output size
        # TODO: do we need any activation?
        self.activation = activation
        self.projection = nn.Linear(self.input_size, self.output_size)

##############################################################################################
    def round(self, x):
        # This method represents the round function 
        # We are gonna use the same function suggested by the paper
        # NB: we need to use the stop gradient operation
        
        x = x + (torch.round(x) - x).detach()
        return x
    
##############################################################################################
    def box_bound(self, x):
        # This method represents the bounding function f. We are gonna
        # use the same suggested by the paper.

        # There's a problem with even Ls. If it is even, L/2 is gonna be -L/2 to L/2,
        # but it will be L+1 values
        x = torch.round(self.Ls / 2) * F.tanh(x) 
        x = torch.minimum(x, self.max_Ls)

        x = self.round(x)

        return x
    
##############################################################################################
    def rescale(self, x):
        # The output will have random integer scale. Maybe it's better to renormalize the output
        # between -1 and 1

        # Each dimension should be [-L/2, L/2]

        x = x / torch.round(self.Ls/2)
        return x 

##############################################################################################
    def forward(self, x):

        # First, we project to the desired output size
        x = self.projection(x)
        if self.activation is not None:
            x = self.activation(x)
        
        # Then, we create the quantozed latent
        x = self.box_bound(x)

        if self.with_rescale:
            x = self.rescale(x)
        else:
            x = (x + self.Ls/2).long()

        return x

##############################################################################################
if __name__ == "__main__":

    batch_size = 32
    input_size = 256
    output_size = 3
    L = 8

    fsq_latent = FSQLatent(
        input_size=input_size,
        output_size=output_size,
        L=L,
        activation=None
    )

    dummy_input = torch.randn(batch_size, input_size)
    output = fsq_latent(dummy_input)
    print(f"Output of the FSQ layer: {output}")

    # Let's check the codebook size
    # the codebook size is jsut the prod of all Ls
    codebook_size = torch.prod(fsq_latent.Ls).detach()
    print(f"Codebook size is: {codebook_size}")
    