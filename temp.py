# import torch
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

# entityList = []
# state = 0
# def fun(a, *y):
#     for i in range(a.shape[0]):
#         if i == 0:
#             if a[i]==1: start = i
#             continue
#         if a[i]!=a[i-1]:
#             if a[i]==1:
#                 start = i
#             elif a[i]==0:
#                 end = i - 1 
#                 entityList.append([start, end])
#     return entityList

# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/")
# text = "其有履行能力而拒不履行生效法律文书确定义务:依法强制执行厦门市中级人民法院(2015)厦民初字第629号《民事判决书书》所作出的第一项判决,即:1.责令被申请人兰新光、兰春光、四平市三达科技有限公司立即向申请人姚明支付借款利息人民币玖佰伍拾贰万柒仟壹佰贰拾叁元(￥9,527,123.00元);2.责令被申请人兰新光、兰春光、四平市三达科技有限公司按照《中华人民共和国民事诉讼法》第二百五十三条以及《最高人民法院关于执行程序中计算迟延履行期间的债务利息适用法律若干问题的解释》等相关规定,加倍支付迟延履行期间的债务利息。"
# tokens = []
# for word in text:
#     word_tokens = tokenizer.tokenize(word)
#     if len(word_tokens)==0:
#         tokens.extend(['unused1'])
#         print(word, "==0")
#     elif len(word_tokens)>1:
#         tokens.extend(word_tokens)
#         print(word, ">1")
#     else:
#         tokens.extend(word_tokens)

# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(len(token_ids))


# from seqeval.metrics import classification_report
# report = classification_report([['B', 'I', 'O']],[['B', 'I', 'O']])
# print(report)


# words=['收', '购', '完', '成', '后', ',', '宁', '德', '时', '代', '将', '持', '有', 'P', 'i', 'l', 'b', 'a', 'r', 'a', '新', '发', '行', '的', '1', '.', '8', '3', '亿', '股', '普', '通', '股', ',', '占', '本', '次', '股', '份', '发', '行', '完', '成', '后', '总', '股', '本', '的', '8', '.', '5', '%', '。']
# labels=['O', 'O', 'O', 'O', 'O', 'O', '收购-sub-org', '收购-sub-org', '收购-sub-org', '收购-sub-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
# label='收购-sub-org'
# token_type_ids=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

import torch
a = torch.tensor([[1.0,2,3], [3,4,5]])
# res = torch.topk(a, 2, -1 ).values
res = torch.max(a, 1)[0]
print(res)
print(torch.sum(res).tolist())