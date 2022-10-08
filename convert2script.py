import torch
from models.SuperPointNet_gauss2_ssmall import SuperPointNet_gauss2_ssmall

model = SuperPointNet_gauss2_ssmall()

path = "logs/superpoint_coco_ssmall_ML22/checkpoints/superPointNet_180000_checkpoint.pth.tar"

model.load_state_dict(torch.load(path)["model_state_dict"])

# model.removeSem()  # if you don't want the semantic decoder

traced_script_module = torch.jit.script(model)
traced_script_module.save("SSP_small_180000_ML22.pt")
