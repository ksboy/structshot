import torch
# path = "./output/conll-2003/checkpointepoch=2.ckpt"
# model = torch.load(path)
# print(model["hyper_parameters"])
# model["hyper_parameters"]["labels"]="./data/conll-2003/labels.txt"
# torch.save(model, "./output/conll-2003/checkpointepoch=2.ckpt")

# path = "./output/conll-2003/checkpointepoch=2.ckpt"
# model = torch.load(path, map_location=torch.device('cpu'))
# print(model.keys())
# # model["hyper_parameters"]["labels"]="./data/conll-2003/labels.txt"
# # torch.save(model, "./output/conll-2003/checkpointepoch=2.ckpt")


# a = torch.tenosr([[1,1,0,0,1,1,0], [1,1,0,0,1,1,0]])

entityList = []
state = 0
def fun(a, *y):
    for i in range(a.shape[0]):
        if i == 0:
            if a[i]==1: start = i
            continue
        if a[i]!=a[i-1]:
            if a[i]==1:
                start = i
            elif a[i]==0:
                end = i - 1 
                entityList.append([start, end])
    return entityList

