import os.path
import re
from collections import Counter
import json


def normalize(content: str) -> str:
    """
    正则化操作：如将多个连续空格压缩为一个空格
    :param content: 待正则化文本
    :return: 正则化文本
    """
    return re.sub(r" +", " ", content)


def pre_tokenize(content: str) -> list:
    """
    预分词操作：根据分隔符分词，这里使用 split(" ") 方法进行分词
    :param content: 待分词文本
    :return: 单词列表
    """
    return content.split(" ")


def get_pairs(word: list) -> list:
    """
    获取所有词元对
    :param word: 字符组成的词列表，例如：["h", "e", "l", "l", "0"]
    :return: 所有词元对集合，例如：{('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', '0')}
    """
    pairs = []
    prev_char = word[0]
    for char in word[1:]:
        pair = (prev_char, char)
        if pair not in pairs:
            pairs.append(pair)
        prev_char = char
    return pairs


class BytesPairEncoderTrainer:
    """
    BPE Tokenizer Trainer Demo

    词表
    words = {
        word_0: [token_0, token_1, ..., token_m],
        word_1: [token_0, token_1, ..., token_m],
        ...
        word_n: [token_0, token_1, ..., token_m],
    }

    词元表
    alphabet = {
        token_0: count_0,
        token_1: count_1,
        ...
        token_m: count_1,
    }
    """

    def __init__(self):
        self.words = {}
        self.word_counts = {}
        self.alphabet = Counter()
        self.merges = []
        self.unk_token = "<unk>"
        self.word_prefix = "_"
        self.special_tokens = []

    def init(self, content: list) -> None:
        """
        初始化词表、词数量表和词元表
        :param content: 语料
        :return: None
        """
        for line in content:
            line = normalize(line)
            words = pre_tokenize(line)
            for word in words:
                word = self.word_prefix + word
                if word not in self.words:
                    self.words[word] = list(word)
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1
        self.alphabet = {}
        for word, chars in self.words.items():
            for ch in chars:
                self.alphabet[ch] = self.alphabet.get(ch, 0) + self.word_counts[word]

    def count_pairs(self):
        """
        统计每个词元的频次，并返回最高频次的词元和最高频次
        :return: top_pair, top_count
        """
        pairs_counter = {}
        for word, chars in self.words.items():
            for pair in get_pairs(chars):
                if pair not in pairs_counter:
                    pairs_counter[pair] = self.word_counts[word]
                else:
                    pairs_counter[pair] += self.word_counts[word]
        top_pair = max(pairs_counter, key=lambda k: pairs_counter[k])
        top_count = pairs_counter[top_pair]
        return top_pair, top_count

    def merge_pair(self, top_pair: tuple) -> None:
        """
        将语料中所有 top_pair 融合为新词元
        :param top_pair: 频次最高的词元对
        """
        for word, chars in self.words.items():
            word_pairs = get_pairs(chars)
            if top_pair not in word_pairs:
                continue
            for i, pair in enumerate(word_pairs):
                if pair == top_pair:
                    chars[i] = "".join(top_pair)
                    chars[i + 1] = None
            self.words[word] = [ch for ch in chars if ch]

    def do_train(self, content: list, steps=1000) -> None:
        """
        训练 BPE Tokenizer
        :param content: 语料
        :param steps: 迭代次数
        :return: None
        训练步骤:
            1. 拆分词为单个词元，建立最初词元表 alphabet
            2. 统计每个词元的频次，挑出频次最高的词元 top_pair 对并将组成的新词元加入到词元表
            3. 将语料中所有 top_pair 融合为新词元
            4. 重复操作，直至词元表中词元最高频次为 1
        """
        # 1. 拆分词为单个词元，建立最初词元表 alphabet
        self.init(content)
        for step in range(steps):
            # 统计每个词元的频次，挑出频次最高的词元 top_pair
            top_pair, top_count = self.count_pairs()
            print(f"step: {step + 1}, top_pair: {top_pair}, count: {top_count}")
            # 词元表中词元最高频次为 1
            if top_count == 1:
                break
            # 将语料中所有 top_pair 融合为新词元
            self.merge_pair(top_pair)
            # 记录merge操作
            self.merges.append(top_pair)
            # 更新词元表
            for raw_pair in top_pair:
                self.alphabet[raw_pair] -= top_count
            # 将组成的新词元加入到词元表
            self.alphabet["".join(top_pair)] = top_count

    def save(self, base_dir) -> None:
        """
        保存词元表和词元合成，生成 vocab.json 和 merges.txt 文件
        :param base_dir:
        :return: None
        """
        print("start save vocab.json and merges.txt")
        vocab = {}
        vocab_file = os.path.join(base_dir, "vocab.json")
        merges_file = os.path.join(base_dir, "merges.txt")
        i = 0
        vocab[self.unk_token] = i
        i += 1
        for special_token in self.special_tokens:
            vocab[special_token] = i
            i += 1
        for token, _ in self.alphabet.items():
            vocab[token] = i
            i += 1
        with open(vocab_file, "w", encoding="utf-8") as f1:
            f1.write(json.dumps(vocab))
        with open(merges_file, "w", encoding="utf-8") as f2:
            for pair in self.merges:
                f2.write(" ".join(pair) + "\n")


class BytesPairEncoder:
    """
    BPE Tokenizer Demo
    """

    def __init__(self):
        self.unk_token = "<unk>"
        self.word_prefix = "_"
        # token: token_id
        self.vocab = {}
        # token_id: token
        self.vocab_r = {}
        # [(old_token1, old_token2)] 两个待合并的词元
        self.merges = []
        self.bpe_pair_ranks = {}
        self.cache = {}

    def from_file(self, base_dir: str) -> None:
        """
        从文件中读取词表和合并操作
        :param base_dir: vocab.json 和 merges.txt 所在路径
        :return: None
        """
        with open(os.path.join(base_dir, "vocab.json"), "r", encoding="utf-8") as f1:
            self.vocab = json.load(f1)
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        with open(os.path.join(base_dir, "merges.txt"), "r", encoding="utf-8") as f2:
            for merge in f2.readlines():
                merge = merge.strip()
                self.merges.append(tuple(merge.split()))
        self.bpe_pair_ranks = dict(zip(self.merges, range(len(self.merges))))

    def bpe(self, word: str) -> list:
        """
        执行 BPE 算法
        1. 拆分词为单个字符词元
        2. 统计词元对的频数，找到频数最高的词元对
        3. 合并该词元对，生成新的词元，得到新的词元序列word
        :param word: 经过分词后的词
        :return: 词元列表
        """
        if word in self.cache:
            return self.cache[word]
        if len(word) == 1:
            return [word]

        word_tokens = list(word)
        pairs = get_pairs(word_tokens)

        while True:
            top_pair = min(pairs, key=lambda pair: self.bpe_pair_ranks.get(pair, float("inf")))
            if top_pair not in self.bpe_pair_ranks:
                break
            for i, pair in enumerate(pairs):
                if pair == top_pair:
                    word_tokens[i] = "".join(top_pair)
                    word_tokens[i + 1] = None
            word_tokens = [ch for ch in word_tokens if ch]
            if len(word_tokens) == 1:
                break
            else:
                pairs = get_pairs(word_tokens)
        self.cache[word] = word_tokens
        return word_tokens

    def tokenize(self, content: str) -> list:
        """
        将文本转换为词元
        :param content: 文本
        :return: 词元列表
        """
        content = normalize(content)
        words = pre_tokenize(content)
        bpe_tokens = []
        for word in words:
            word = self.word_prefix + word
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(word))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens: list) -> list:
        """
        将词元映射为 id
        :param tokens: 词元列表
        :return: id 列表
        """
        return [self.vocab[token] if token in self.vocab else self.vocab[self.unk_token] for token in tokens]

    def encode(self, content: str) -> list:
        """
        将文本编码为 id
        :param content: 文本
        :return: id 列表
        """
        return self.convert_tokens_to_ids(self.tokenize(content))

    def decode(self, ids: list) -> str:
        """
        将 id 列表解码为文本
        :param ids: id 列表
        :return: 文本
        """
        if len(ids) == 0:
            return ""
        tokens = [self.vocab_r[idx] for idx in ids]
        return "".join(tokens).replace(self.word_prefix, " ")[1:]

    def __call__(self, content):
        self.encode(content)


if __name__ == "__main__":
    text = [
        "New York can impart a great sense of forlornness or abandonment, yet it rarely seems dead or unresourceful.",
        "Through either shifting your location ten blocks or reducing your fortune by five dollars, "
        "you can experience rejuvenation - this chance for sudden rebirth is a quality of New York."
    ]
    # train
    bpe_trainer = BytesPairEncoderTrainer()
    bpe_trainer.do_train(text)
    bpe_trainer.save(".")

    # BPETokenizer
    bpe = BytesPairEncoder()
    bpe.from_file(".")
    encode_content = text[0]
    print(encode_content)
    input_ids = bpe.encode(encode_content)
    print(input_ids)
    decode_content = bpe.decode(input_ids)
    print(decode_content)

