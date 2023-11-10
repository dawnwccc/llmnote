from collections import Counter
import re


class BytePairEncoder:
    """
    corpus = {
        word_0: [token_0, token_1, ..., token_m],
        word_1: [token_0, token_1, ..., token_m],
        ...
        word_n: [token_0, token_1, ..., token_m],
    }

    vocab = {
        token_0: count_0,
        token_1: count_1,
        ...
        token_m: count_m
    }
    """

    def __init__(self):
        self.ws_token = "_"
        self.unk_token = "unk"

        self.corpus = {}
        self.word_count = {}
        self.vocab = Counter()

        self.id_tokens = {}
        self.token_ids = {}

    def train(self, text, steps=1):
        self.init_state(text)
        self.dump_init()
        for step in range(steps):
            print("=" * 12 + f" step: {step} " + "=" * 12)
            self.merge_pair()
            # self.dump_merge()
        print("=" * 12 + " dump final vocab " + "=" * 12)
        for token, count in sorted(self.vocab.items(), key=lambda x: x[1], reverse=True):
            print(f"{token} -> {count}")
        self.gen_id_token_map()
        print("=" * 12 + " dump final id_tokens " + "=" * 12)
        print(self.token_ids)
        print("=" * 40)

    def gen_id_token_map(self):
        self.id_tokens[0] = self.unk_token
        self.token_ids[self.unk_token] = 0

        idx = 1
        for token, _ in self.vocab.most_common():
            self.id_tokens[idx] = token
            self.token_ids[token] = idx
            idx += 1

    def segment(self, text):
        if len(text) == 1:
            return text if text in self.vocab else self.unk_token
        segments = [ch for ch in text]
        merge_rules = Counter()
        for i in range(len(segments) - 1):
            token_word = segments[i] + segments[i + 1]
            if token_word in self.vocab:
                merge_rules.update({(i, token_word): self.vocab[i]})
        while merge_rules:
            (i, token_word), _ = merge_rules.most_common(1)[0]
            if i >= len(segments) - 1 or segments[i] + segments[i + 1] != token_word:
                merge_rules.pop((i, token_word))
            for i in range(len(segments) - 1):
                if segments[i] + segments[i + 1] == token_word:
                    segments[i] = token_word
                    segments[i + 1] = ""
            segments = [seg for seg in segments if seg]
            if len(segments) <= 1:
                break
            for i in range(len(segments) - 1):
                token_word = segments[i] + segments[i + 1]
                if token_word in self.vocab:
                    merge_rules.update({(i, token_word): self.vocab[i]})
        return segments

    def encode(self, text):
        if not text: return
        text = self.preprocess(text)
        text = self.ws_token + re.sub(" ", self.ws_token, text.strip())
        seg_txt = self.segment(text)
        seg_ids = [self.token_ids[token] if token in self.token_ids else 0 for token in seg_txt]
        return (seg_txt, seg_ids)

    def decode(self, ids):
        text = "".join(self.id_tokens[idx] for idx in ids[1:]).replace(self.ws_token, " ")
        return text

    def init_state(self, content):
        # init corpus and word_count
        for line in content:
            # normalization
            sentence = self.preprocess(line.strip())
            # pre-tokenization
            self.process_sentence(sentence)
        alphabet = {}
        for word, chrs in self.corpus.items():
            for ch in chrs:
                alphabet[ch] = alphabet.get(ch, 0) + self.word_count[word]
        self.vocab.update(alphabet)

    def preprocess(self, text):
        # 将多个空白压缩为1个
        return re.sub(r"\s+", " ", text)

    def process_sentence(self, sentence):
        words = sentence.split()
        for word in words:
            word = self.ws_token + word
            if word not in self.corpus:
                self.corpus[word] = [ch for ch in word]
                self.word_count[word] = 1
            else:
                self.word_count[word] += 1

    def dump_init(self):
        print("=" * 12 + " dump initial state " + "=" * 12)
        print("--> dump corpus <--")
        for word, text in self.corpus.items():
            print(f"{word} -> {text}")
        print("-" * 20)
        print("--> dump word count <--")
        for word, count in self.word_count.items():
            print(f"{word} -> {count}")
        print("-" * 20)
        print("--> dump vocab <--")
        for token, count in self.vocab.items():
            print(f"{token} -> {count}")
        print("=" * 40)

    def update_vocab(self, symbol, count):
        if symbol in self.vocab:
            self.vocab[symbol] += count
        else:
            self.vocab[symbol] = count

    def dump_merge(self):
        print("=" * 40)
        print("--> dump vocab <--")
        for token, count in sorted(self.vocab.items(), key=lambda x: x[1], reverse=True):
            print(f"{token} -> {count}")
        print("-" * 40)
        print("--> dump corpus <--")
        for word, tokens in self.corpus.items():
            print(f"[{self.word_count[word]:3d}] * {word} -> {tokens}")
        print("=" * 40)

    def gen_bigrams(self):
        bigram_counter = Counter()
        for word, text in self.corpus.items():
            for i in range(len(text) - 1):
                bigram = text[i] + text[i + 1]
                bigram_counter[bigram] += self.word_count[word]
        return bigram_counter

    def merge_pair(self):
        top_bigram, top_count = self.gen_bigrams().most_common(1)[0]
        print(f"=> top_bigram: {top_bigram}, top_count: {top_count}")
        if top_count == 1:
            return
        for word, text in self.corpus.items():
            merged = False
            for i in range(len(text) - 1):
                if text[i] + text[i + 1] == top_bigram:
                    self.update_vocab(text[i], -self.word_count[word])
                    self.update_vocab(text[i + 1], -self.word_count[word])
                    text[i] = top_bigram
                    text[i + 1] = ""
                    merged = True
            if merged:
                self.corpus[word] = [token for token in text if token]
        self.update_vocab(top_bigram, top_count)


if __name__ == "__main__":
    content = [
        "bug " * 10 + "pug   " * 5 + "pun " * 12 + "bun " * 4 + "hug " * 5,
        "这是OpenAI团队前一段时间放出来的预印版论文。他们的目标是学习一个通用的表示，能够在大量任务上进行应用。",
        "这篇论文的亮点主要在于，他们利用了Transformer网络代替了LSTM作为语言模型来更好地捕获长距离语言结构",
        "然后在进行具体任务有监督微调时，使用了模型作为附属任务训练目标。",
        "ChatGPT是由OpenAI开发的一种大规模语言模型，其技术细节和研究成果已经在多个学术论文中进行了描述和发布。",
        "这些论文涵盖了ChatGPT的基本架构、训练和优化方法、性能评估和应用场景等方面，对于理解ChatGPT的内部机制和应用价值都非常有帮助。",
        "采用了基于Transformer架构的生成式预训练方法，将无标签的大规模文本数据用于预训练，从而在多个NLP任务上取得了优异的效果。"
    ]
    bpe = BytePairEncoder()
    bpe.train(content, 60)
    text = "你好，我是ChatGPT，OpenAI研发的基于Transformer架构的对话语言模型。"
    seg_text, seg_ids = bpe.encode(text)
    print(text)
    print(seg_text)
    print(seg_ids)
    dec_text = bpe.decode(seg_ids)
    print(dec_text)
