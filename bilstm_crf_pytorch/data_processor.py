import json
from collections import Counter

from config import train_path, dev_path, test_path
from vocabulary import Vocabulary, Labels


class CluenerProcessor:
    """Processor for the chinese ner data set."""
    def __init__(self, data_dir):
        self.vocab = Vocabulary()
        self.labels = Labels()
        self.data_dir = data_dir

    def get_vocab(self):
        vocab_path = self.data_dir / 'vocab.pkl'
        if vocab_path.exists():
            self.vocab.load_from_file(str(vocab_path))
        else:
            # TODO：也读取 dev 和 test？
            files = [train_path, dev_path, test_path]
            for file in files:
                with open(str(file), 'r') as fr:
                    for line in fr:
                        line = json.loads(line.strip())
                        text = line['text']
                        # 创建 char dict
                        self.vocab.update(list(text))
            self.vocab.build_vocab()
            self.vocab.save(vocab_path)

    def get_labels(self):
        files = [train_path]
        for file in files:
            with open(str(file), 'r') as fr:
                for line in fr:
                    line = json.loads(line.strip())
                    label = line.get('label') or {}
                    self.labels.update(label.keys())
        self.labels.build_label_ids('bios', add_special_label=True)
        print('labels', self.labels.label_counter.most_common())

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(str(train_path), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(str(dev_path), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(str(test_path), "test")

    def _create_examples(self, input_path, mode):
        examples = []
        with open(input_path, 'r') as f:
            idx = 0
            for line in f:
                json_d = {}
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label', None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                json_d['id'] = f"{mode}_{idx}"
                json_d['context'] = " ".join(words)
                json_d['tag'] = " ".join(labels)
                json_d['raw_context'] = "".join(words)
                idx += 1
                examples.append(json_d)
        return examples


if __name__ == '__main__':
    processor = CluenerProcessor('')
    samples = processor._create_examples(dev_path, 'dev')
    print(json.dumps(samples[:6], indent=2, ensure_ascii=False))
