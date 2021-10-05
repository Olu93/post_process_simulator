# %%
from IPython.display import display
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from allennlp.data import Token, Vocabulary, TokenIndexer, Tokenizer
from allennlp.data.tokenizers import WhitespaceTokenizer

matplotlib.rcParams['figure.facecolor'] = 'w'




reader = PaymentRequestReader(True)
display(reader.original_data.head())
display(reader.data.head())
# %%
data = reader.data
data
# %%
reader.ext_traces

# %%
reader.vocab
# %%
# https://guide.allennlp.org/representing-text-as-features#6
import warnings
from typing import Dict

import torch
from allennlp.data import Token, Vocabulary, TokenIndexer, Tokenizer
from allennlp.data.fields import ListField, TextField
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,

)

from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
    
)
from allennlp.nn import util as nn_util


tokenizer: Tokenizer = WhitespaceTokenizer()

# Represents each token with an array of characters in a way that ELMo expects.
token_indexer: TokenIndexer = SingleIdTokenIndexer()

# Both ELMo and BERT do their own thing with vocabularies, so we don't need to add
# anything, but we do need to construct the vocab object so we can use it below.
# (And if you have any labels in your data that need indexing, you'll still need
# this.)

text = ["This is some awesome text .", "Some other text ."]
tokens = tokenizer.batch_tokenize(text)
vocab = Vocabulary()
vocab.add_tokens_to_namespace([str(tok) for sent in tokens for tok in sent])
print("ELMo tokens:", tokens)

INDEXER_NAME = "id_indexer"
text_field = TextField(tokens[0], {INDEXER_NAME: token_indexer})
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("ELMo tensors:", tensor_dict)

embedding = Embedding(num_embeddings=10, embedding_dim=3)
embedder = BasicTextFieldEmbedder(token_embedders={"id_indexer": embedding})
embedded_tokens = embedder(tensor_dict)
print("Using the TextFieldEmbedder:", embedded_tokens)