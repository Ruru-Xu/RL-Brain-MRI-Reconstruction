import torch
from recons.cascade_network import build_reconstruction_model

def load_recon_model(recon_model_path):
    checkpoint = torch.load(recon_model_path)
    recon_model = build_reconstruction_model()
    recon_model.load_state_dict(checkpoint['model'])
    return recon_model

def test_recons(recon_model, test_img):
    with torch.no_grad():
        recon = fastmri.complex_abs(recon_model(test_img).permute(0, 2, 3, 1))
    return recon