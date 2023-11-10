from datasets import load_dataset
from tokenizer import BytesPairEncoderTrainer, BytesPairEncoder

raw_datasets = load_dataset("code_search_net", "python", cache_dir="../../dataset")

# train
# training_corpus = raw_datasets["train"][:100]["whole_func_string"]
# bpe_trainer = BytesPairEncoderTrainer()
# bpe_trainer.do_train(training_corpus, 1000)
# bpe_trainer.save(".")

# test
test_corpus = raw_datasets["train"][101]["whole_func_string"]
bpe = BytesPairEncoder()
bpe.from_file(".")
input_ids = bpe.encode(test_corpus)
print(input_ids)
decode_text = bpe.decode(input_ids)
print(decode_text)