from torch import nn

class Model2(nn.Module):
    def __init__(self):
        super().__init__()

class ViViT(nn.Module):
    def __init__(self):
        m1_enc_l = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=enc_n_heads) 
                   
        self.model1 = nn.TransformerEncoder(
            encoder_layer=m1_enc_l,
            num_layers=n_enc_layers, 
            norm=None)