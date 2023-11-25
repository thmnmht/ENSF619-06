import torch
import torch.nn as nn
from models.Autoformer import Model as Autoformer
from models.FEDformer import Model as FEDformer
from models.Informer import Model as Informer

class Model(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.aut = Autoformer(config)
        self.aut.load_state_dict(torch.load('checkpoints/custom_Autoformer_random_modes64_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth', map_location=torch.device('cpu')))
        self.fed = FEDformer(config)
        self.fed.load_state_dict(torch.load('checkpoints/custom_FEDformer_random_modes64_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth', map_location=torch.device('cpu')))
        self.inf = Informer(config)
        self.inf.load_state_dict(torch.load('checkpoints/custom_Informer_random_modes64_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_test_0/checkpoint.pth', map_location=torch.device('cpu')))
        # for model in [self.aut, self.fed, self.inf]:
        #     for param in model.parameters():
        #         param.requires_grad = False
        
        self.combine = nn.Conv1d(3, 1, 1)

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        aut_out = self.aut(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
        fed_out = self.fed(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
        inf_out = self.inf(x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask, dec_self_mask, dec_enc_mask)
        
        output = torch.cat((aut_out, fed_out, inf_out), dim=-1).transpose(1,2)
        output = self.combine(output).transpose(1,2)
        
        # output = torch.mean(torch.cat((aut_out, fed_out, inf_out), dim=-1), dim=-1).unsqueeze(-1)
        
        return output