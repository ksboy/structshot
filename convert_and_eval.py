from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities
import sys
sys.path.append('/hy-nas/workspace/code_repo/ner')
from metrics import f1_score_identification, precision_score_identification, recall_score_identification, \
    accuracy_score_entity_classification, accuracy_score_token_classification, \
    f1_score_token_classification, precision_score_token_classification, recall_score_token_classification 

def convert(file_path, pred_file, output_file):
    preds = open(pred_file).read().splitlines()
    guid_index = 0
    results = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    entities = get_entities(labels)
                    for _, start, end in entities:
                        token_type_ids = [0] * len(words)
                        # if guid_index >= len(preds):
                        #     print(guid_index)
                        label = preds[guid_index] 
                        labels[start] = 'B-' + label
                        for i in range(start, end+1):
                            token_type_ids[i] = 1
                            labels[i] = 'I-' + label
                        guid_index += 1
                    results.append([words, labels])
                    words = []
                    labels = []
            else:
                splits = line.strip('\n').split('\t')
                # if len(splits)!=4:
                #     print(line)
                # assert len(splits)==4
                words.append(splits[0])
                if len(splits) > 1:
                    label = splits[-1].replace("\n", "")
                    labels.append(label)
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            entities = get_entities(labels)
            for _, start, end in entities:
                token_type_ids = [0] * len(words)
                label = preds[guid_index]
                labels[start] = 'B-' + label
                for i in range(start, end+1):
                    token_type_ids[i] = 1
                    labels[i] = 'I-' + label
                guid_index += 1
            results.append([words, labels])

    print(guid_index)
    writer = open(output_file, 'w')
    for words, labels in results:
        assert len(words) == len(labels)
        for word, label in zip(words, labels):
            writer.write(word + '\t' + label +  '\n')
        writer.write('\n')


def read_examples_from_file(file_path):
    results = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    results.append(labels)
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            results.append(labels)
        
    return results

if __name__ == "__main__":
    # convert
    convert("./data/FewFC-main/rearranged/few/trigger/identification.txt", "output/pred/ccks/trigger/entity/NNShot/test_preds.txt", "output/pred/ccks/trigger/entity/NNShot/pred_as_conll.txt")

    # eval
    preds = read_examples_from_file("output/pred/ccks/trigger/entity/NNShot/pred_as_conll.txt")
    labels = read_examples_from_file("./data/FewFC-main/rearranged/few/trigger/dev.txt")
    report = classification_report(labels, preds)
    print(report)
    results = {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision_i": precision_score_identification(labels, preds),
        "recall_i": recall_score_identification(labels, preds),
        "f1_i": f1_score_identification(labels, preds),
        "precision_c": precision_score_token_classification(labels, preds),
        "recall_c": recall_score_token_classification(labels, preds),
        "f1_c": f1_score_token_classification(labels, preds),
        "accuracy_token_c": accuracy_score_token_classification(labels, preds),
        "accuracy_entity_c": accuracy_score_entity_classification(labels, preds)
    }
    print(results)
