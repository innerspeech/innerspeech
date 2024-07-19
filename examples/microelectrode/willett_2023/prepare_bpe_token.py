import sentencepiece as spm

# spm.SentencePieceTrainer.train(input='test/botchan.txt', model_prefix='m', vocab_size=1000, user_defined_symbols=['SIL', 'UNK'])

s = spm.SentencePieceProcessor(model_file='data/977d4e24975b431ebb44f2dfcdea8778_tokenizer.model')

encoded = s.encode_as_pieces("this's is- a test")

print(encoded)

# ['▁this', '▁is', '▁a', '▁test']

#decode

decoded = s.decode_pieces(encoded)

print(decoded)


encoded_int = s.encode_as_ids('this is a test')

print(encoded_int)

# [12, 13, 10, 9]

#decode

decoded_int = s.decode_ids(encoded_int)

print(decoded_int)

# this is a test

print(s.decode([3, 36, 61, 160, 61, 20, 45, 266, 15, 459, 491]))