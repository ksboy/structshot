import torch
# path = "./output/conll-2003/checkpointepoch=2.ckpt"
# model = torch.load(path)
# print(model["hyper_parameters"])
# model["hyper_parameters"]["labels"]="./data/conll-2003/labels.txt"
# torch.save(model, "./output/conll-2003/checkpointepoch=2.ckpt")

path = "./output/conll-2003/checkpointepoch=2.ckpt"
model = torch.load(path, map_location=torch.device('cpu'))
print(model.keys())
# model["hyper_parameters"]["labels"]="./data/conll-2003/labels.txt"
# torch.save(model, "./output/conll-2003/checkpointepoch=2.ckpt")