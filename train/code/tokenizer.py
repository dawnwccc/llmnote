import re
from collections import Counter
import json


class BytesPairEncoder:
    """
    BPE Tokenizer Demo

    words = {
        word_0: [token_0, token_1, ..., token_m],
        word_1: [token_0, token_1, ..., token_m],
        ...
        word_n: [token_0, token_1, ..., token_m],
    }

    alphabet = {
        token_0: count_0,
        token_1: count_1,
        ...
        token_m: count_1,
    }
    """

    def __init__(self):
        self.unk_token = "<unk>"
        self.words = {}
        # word: count
        self.word_counts = {}
        self.alphabet = Counter()
        # token: token_id
        self.vocab = {}
        # token_id: token
        self.vocab_r = {}
        # ((old_token1, old_token2), new_token)
        self.merges = []

    def train(self, text, steps=1) -> None:
        """
        BPE Tokenizer训练接口
        :param text: 语料
        :param steps: 迭代次数
        """
        self.init(text)
        self.print_state()
        for step in range(steps):
            top_count = self.merge_pair()
            self.print_state()
            if top_count == 1:
                break
        print("-"*20 + " create vocab " + "-"*20)
        self.create_vocab()
        self.print_vocab_merges()

    def init(self, text):
        """
        初始化词表和字母表
        :param text:
        :return:
        """
        # 初始化词表
        for line in text:
            line = self.normalize(line)
            words = self.pre_tokenize(line)
            for word in words:
                if word not in self.words:
                    self.words[word] = [ch for ch in word]
                    self.word_counts[word] = 1
                else:
                    self.word_counts[word] += 1
        alphabet = {}
        for word, chars in self.words.items():
            for ch in chars:
                alphabet[ch] = alphabet.get(ch, 0) + self.word_counts[word]
        self.alphabet.update(alphabet)

    def print_state(self):
        """输出当前words, word_counts, alphabet"""
        print("=" * 40)
        print("-" * 20 + "> words <" + "-" * 20)
        for word, chars in self.words.items():
            print(f"{word} -> {chars}")
        print("-" * 20 + "> word_counts <" + "-" * 20)
        for word, count in self.word_counts.items():
            print(f"{word}: {count}")
        print("-" * 20 + "> alphabet <" + "-" * 20)
        for token, count in self.alphabet.items():
            print(f"{token}: {count}")
        print("=" * 40)

    def normalize(self, text):
        """
        正则化处理，去掉多余的空格
        :param text: 语料
        :return: 正则化后的语料
        """
        return re.sub(r"\s+", " ", text)

    def pre_tokenize(self, text):
        """
        预处理，这里采用 split 进行分词
        :param text: 正则化后的语料
        :return: 预处理后的语料
        """
        return text.split()

    def count_pair(self):
        """
        计算要合并的词元
        :return: 合并词元操作，词元数量
        """
        bigram_counter = {}
        for word, chars in self.words.items():
            for i in range(len(chars) - 1):
                bigram = ((chars[i], chars[i+1]), chars[i] + chars[i + 1])
                if bigram not in bigram_counter:
                    bigram_counter[bigram] = self.word_counts[word]
                else:
                    bigram_counter[bigram] += self.word_counts[word]
        top_bigram = max(bigram_counter, key=lambda k: bigram_counter[k])
        top_count = bigram_counter[top_bigram]
        return top_bigram, top_count

    def update_alphabet(self, token, count):
        """更新词元表"""
        if token in self.alphabet:
            self.alphabet[token] += count
        else:
            self.alphabet[token] = count

    def merge_pair(self):
        """
        合并词元
        :return: 词元数量
        """
        top_bigram, top_count = self.count_pair()
        print("-" * 12 + f" top bigram: {top_bigram}, count: {top_count} " + "-" * 12)
        if top_count == 1:
            return
        self.merges.append(top_bigram)
        for word, chars in self.words.items():
            is_merged = False
            for i in range(len(chars) - 1):
                if chars[i] + chars[i + 1] == top_bigram[1]:
                    self.update_alphabet(chars[i], -self.word_counts[word])
                    self.update_alphabet(chars[i + 1], -self.word_counts[word])
                    chars[i] = top_bigram[1]
                    chars[i + 1] = ""
                    is_merged = True
            if is_merged:
                self.words[word] = [ch for ch in chars if ch]
        self.update_alphabet(top_bigram[1], top_count)
        return top_count

    def create_vocab(self):
        """创建词表vocab和反转词表vocab_r"""
        self.vocab[self.unk_token] = 0
        self.vocab_r[0] = self.unk_token
        i = 1
        for token, _ in self.alphabet.items():
            self.vocab[token] = i
            self.vocab_r[i] = token
            i += 1

    def print_vocab_merges(self):
        """输出词表和合并操作"""
        print("="*40)
        print("-" * 20 + "> vocab <" + "-" * 20)
        for token, token_id in self.vocab.items():
            print(f"{token} -> {token_id}")
        print("-" * 20 + "> merges <" + "-" * 20)
        for merge in self.merges:
            print(f"{merge[0]} -> {merge[1]}")
        print("=" * 40)

    def save(self):
        """保存词表和合并操作"""
        pass

    def load(self, vocab_file, merges_file):
        """从文件中读取词表和合并操作"""
        pass

    def encode(self, text):
        """编码"""
        pass

    def decode(self, ids):
        """解码"""
        pass


if __name__ == "__main__":
    text = [
        "New York can impart a great sense of forlornness or abandonment, yet it rarely seems dead or unresourceful.\n"
        "Through either shifting your location ten blocks or reducing your fortune by five dollars, "
        "you can experience rejuvenation - this chance for sudden rebirth is a quality of New York."
    ]
    bpe = BytesPairEncoder()
    bpe.train(text, 100)
