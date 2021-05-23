import logging
import os
import json
from typing import List, TextIO, Union
from seqeval.metrics.sequence_labeling import get_entities
from utils_entity import InputExample, Split, EntityClassificationTask


logger = logging.getLogger(__name__)


class NER(EntityClassificationTask):
    def __init__(self, label_idx=-1):
        # in NER datasets, the last column is usually reserved for NER label
        self.label_idx = label_idx

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        file_path = os.path.join(data_dir, "{}.txt".format(mode))
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels_i, labels_c = [], []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        if mode!='train':
                            token_type_ids = [0] * len(words)
                            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, token_type_ids=token_type_ids, labels = labels_c, label = label))
                        else:
                            entities = get_entities(labels_i)
                            for _, start, end in entities:
                                token_type_ids = [0] * len(words)
                                _labels_c = ['O']*len(words)
                                label = labels_c[start]
                                for i in range(start, end+1):
                                    token_type_ids[i] = 1
                                    _labels_c[i] = labels_c[i]
                                    assert label == labels_c[i]
                                examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, token_type_ids=token_type_ids, labels=labels_c, label = label))
                        guid_index += 1
                        words = []
                        labels_i, labels_c = [], []
                else:
                    splits = line.split()
                    # if len(splits)!=4:
                    #     print(line)
                    # assert len(splits)==4
                    words.append(splits[0])
                    if len(splits) > 1:
                        label = splits[-1].replace("\n", "")
                        if len(label)==1:
                            labels_i.append("O")
                            labels_c.append("O")
                        else:
                            labels_i.append(label.split("-", 1)[0])
                            labels_c.append(label.split("-", 1)[1])
                    else:
                        # Examples could have no label for mode = "test"
                        labels_i.append("O")
                        labels_c.append("O")

            if words:
                if mode!='train':
                    token_type_ids = [0] * len(words)
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, token_type_ids=token_type_ids, labels = labels_c, label = label))
                else:
                    entities = get_entities(labels_i)
                    for _, start, end in entities:
                        token_type_ids = [0] * len(words)
                        _labels_c = ['O']*len(words)
                        label = labels_c[start]
                        for i in range(start, end+1):
                            token_type_ids[i] = 1
                            _labels_c[i] = labels_c[i]
                            assert label == labels_c[i]
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, token_type_ids=token_type_ids, labels = _labels_c, label = label))
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)
            else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

def remove_duplication(alist):
    res = []
    for item in alist:
        if item not in res:
            res.append(item)
    return res

class EE(NER):
    ## ccks格式
    def role_process_bio_ccks(self, input_file, add_event_type_to_role=False, is_predict=False):
        rows = open(input_file, encoding='utf-8').read().splitlines()
        results = []
        for row in rows:
            if len(row)==1: print(row)
            row = json.loads(row)
            labels = ['O']*len(row["content"])
            if is_predict: 
                results.append({"id":row["id"], "words":list(row["content"]), "labels":labels})
                continue
            for event in row["events"]:
                event_type = event["type"]
                for arg in event["mentions"]:
                    role = arg['role']
                    if role=="trigger": continue
                    if add_event_type_to_role: role = event_type + '-' + role
                    argument_start_index, argument_end_index = arg["span"]
                    labels[argument_start_index]= "B-{}".format(role)
                    for i in range(argument_start_index+1, argument_end_index):
                        labels[i]= "I-{}".format(role)
                    # if arg['alias']!=[]: print(arg['alias'])
            results.append({"guid":row["id"], "words":list(row["content"]), "labels":labels})
        # write_file(results,output_file)
        return results

    def read_examples_from_file(self, data_dir, mode: Union[Split, str],  task="role", dataset="ccks") -> List[InputExample]:
        file_path = os.path.join(data_dir, "{}.json".format(mode))
        if dataset=="ccks":
            if task=='trigger': items = self.trigger_process_bio_ccks(file_path)
            elif task=='role': items = self.role_process_bio_ccks(file_path, add_event_type_to_role=True)
        elif dataset=="lic":
            if task=='trigger': items = self.trigger_process_bio_lic(file_path)
            elif task=='role': items = self.role_process_bio_lic(file_path, add_event_type_to_role=True)
        return [InputExample(**item) for item in items]

    def get_labels(self, path: str, task='role', mode="ner", target_event_type='', add_event_type_to_role=True) -> List[str]:
        if not path:
            if mode=='ner':
                return ["O", "B-ENTITY", "I-ENTITY"]
            else:
                return ["O"]

        elif task=='trigger':
            labels = []
            rows = open(path, encoding='utf-8').read().splitlines()
            if mode == "ner": labels.append('O')
            for row in rows:
                row = json.loads(row)
                event_type = row["event_type"]
                if mode == "ner":
                    labels.append("B-{}".format(event_type))
                    labels.append("I-{}".format(event_type))
                else:
                    labels.append(event_type)
            return remove_duplication(labels)

        elif task=='role' and target_event_type=='':
            labels = []
            rows = open(path, encoding='utf-8').read().splitlines()
            if mode == "ner": labels.append('O')
            for row in rows:
                row = json.loads(row)
                event_type = row["event_type"]
                for role in row["role_list"]:
                    role_type = role['role'] if not add_event_type_to_role else event_type + '-' + role['role']
                    if mode == "ner":
                        labels.append("B-{}".format(role_type))
                        labels.append("I-{}".format(role_type))
                    else:
                        labels.append(role_type)
            return remove_duplication(labels)

        # 特定类型事件 [TASK] 中的角色
        elif task=='role' and target_event_type!='':
            labels = []
            rows = open(path, encoding='utf-8').read().splitlines()
            if mode == "ner": labels.append('O')
            for row in rows:
                row = json.loads(row)
                event_type = row["event_type"]
                if event_type!=target_event_type:
                    continue
                for role in row["role_list"]:
                    role_type = role['role'] if not add_event_type_to_role else event_type + '-' + role['role']
                    if mode == "ner":
                        labels.append("B-{}".format(role_type))
                        labels.append("I-{}".format(role_type))
                    else:
                        labels.append(role_type)
            return remove_duplication(labels)


