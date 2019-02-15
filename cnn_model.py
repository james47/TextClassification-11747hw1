import torch

class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)                 # batch_size x nwords x emb_size
        if len(emb.shape) < 3:
            emb = emb.unsqueeze(0)
        emb = emb.permute(0, 2, 1)                  # batch_size x emb_size x nwords
        h = self.conv_1d(emb)                       # batch_size x num_filters x nwords
        # Do max pooling
        h = h.max(dim=2)[0]                         # batch_size x num_filters
        h = self.relu(h)
        out = self.projection_layer(h)              # size(out) = batch_size x ntags
        return out
