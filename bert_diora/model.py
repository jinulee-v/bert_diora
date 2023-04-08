from typing import *

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoConfig

from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer

torch.autograd.set_detect_anomaly(True)
import time

def flatten_tri_size(max_length):
    return (max_length + 1) * max_length // 2

def flatten_tri(begin, end, max_length):
    """
    Flattens the index (begin, length) given the max_length.
    max_length: 6 -> linear size: 21 (6 * 7 / 2)
    structure:
    20                (len=6)
    18 19             (len=5)
    15 16 17          (len=4)
    11 12 13 14       (len=3)
     6  7  8  9 10    (len=2)
     0  1  2  3  4  5 (len=1)
    """
    assert end <= max_length
    length = end - begin
    return (max_length + 1) * max_length // 2 - (max_length - length + 2) * (max_length - length + 1) // 2   \
           + begin

class BertDiora(nn.Module):
    """
    Re-implementation of DIORA using pretrained transformers (e.g. BERT).
    """

    def __init__(self, model_id: str='bert-base-uncased', freeze: bool=True, nltk_tok: str="ptb", device = torch.device('cpu')):
        super(BertDiora, self).__init__()

        # Retrieve size first
        config = AutoConfig.from_pretrained(model_id)
        self.size = config.hidden_size
        self.device = device
        
        # Compose function: MLP
        self.compose_mlp = nn.Sequential(
            nn.Linear(self.size*2, self.size),
            nn.ReLU(),
            nn.Linear(self.size, self.size),
            nn.ReLU()
        ).to(device)
        # Score function: bilinear. Following the DIORA paper, we set bias to False.
        self.bilinear = nn.Bilinear(self.size, self.size, 1, bias=False).to(device)
        # Root bias for outside pass
        self.root_bias = nn.Parameter(torch.zeros(self.size)).to(device)

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Load pretrained transformers
        # (after the initialization to prevent BERT being reset)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if freeze:
            self.model.requires_grad = False

        if nltk_tok == "ptb":
            self.nltk_tokenize = TreebankWordTokenizer().tokenize
            self.nltk_detokenize = TreebankWordDetokenizer().detokenize
        else: raise ValueError("nltk_tok must be in ['ptb'].")
    
    def get_nltk_embeddings(self, sentences):
        """
        Obtain embeddings for each NLTK-tokenized tokens, by mean-pooling the Sentencepiece-BERT style subword embeddings.
        """
        # Tokenize the sentences using the tokenizer
        tokenized_sentences = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        # Convert the tokenized sentences to input_ids and attention_masks
        input_ids = tokenized_sentences.input_ids.to(self.device)
        attention_masks = tokenized_sentences.attention_mask.to(self.device)

        # Generate embeddings using the transformer model
        outputs = self.model(input_ids, attention_mask=attention_masks)
        embeddings = outputs[0]

        # Get the list of NLTK tokens for each sentence
        nltk_tokens_list = [self.nltk_tokenize(sentence) for sentence in sentences]

        # Match the transformer subwords to the NLTK tokens
        index_map_list = []
        for nltk_tokens in nltk_tokens_list:
            token_ids_list = self.tokenizer.batch_encode_plus(nltk_tokens, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            index_map = []
            for i, token_id in enumerate(token_ids_list):
                # Keep track of the mapping between NLTK token and transformer token ids
                index_map += [i] * len(self.tokenizer.decode(token_id).split())
            index_map += [-1] * (embeddings.size(1) - len(index_map)) # Pad the index_map to max_len(subword)
            index_map_list.append(torch.tensor(index_map))

        # Compute mean-pooled embeddings for each NLTK token in each sentence
        result_embeddings = []
        for i in range(len(sentences)):
            nltk_tokens = nltk_tokens_list[i]
            index_map = index_map_list[i]
            sentence_embeddings = []
            for j, token in enumerate(nltk_tokens):
                # Get the embeddings corresponding to the transformer tokens for this NLTK token
                token_embeddings = embeddings[i, index_map == j]
                
                # Compute the mean-pooled embedding
                mean_pooled_embedding = torch.mean(token_embeddings, dim=0)
                
                # Append the mean-pooled embedding to the list of sentence embeddings
                sentence_embeddings.append(mean_pooled_embedding.unsqueeze(0))

            result_embeddings.append(torch.cat(sentence_embeddings, dim=0))
        
        return nn.utils.rnn.pad_sequence(result_embeddings), [sent_emb.size(0) for sent_emb in result_embeddings]


    def forward(self, sentences: List[str]):# Tokenize the sentence using NLTK
        """
        Compute inside / outside pass and obtain the loss value.
        """
        start_time = time.time() # DEBUG
        batch_size = len(sentences)

        word_embeddings, sent_lengths = self.get_nltk_embeddings(sentences) # seq_len * batch_size * size
        base_vecs = word_embeddings # To reduce dimension of word vectors, modify here
        max_len = max(sent_lengths)

        # Inside pass, batchified
        inside_vecs = torch.zeros([flatten_tri_size(max_len), batch_size, self.size], device=self.device)
        inside_scores = torch.zeros([flatten_tri_size(max_len), batch_size, 1], device=self.device)
        # base case for inside pass
        inside_vecs[:base_vecs.size(0)] = base_vecs

        for length in range(2, max_len+1):
            for begin in range(0, max_len+1-length):
                # span: begin ..(length).. end
                end = begin+length
                # Iterate through inside contexts
                context_list_left = [flatten_tri(begin, context, max_len) for context in range(begin+1, end)]
                context_list_right = [flatten_tri(context, end, max_len) for context in range(begin+1, end)]
                # left: [begin:context]; right: [context:end]
                left = inside_vecs[context_list_left] # n_span * batch_size * size
                right = inside_vecs[context_list_right]  # n_span * batch_size * size

                context_score = (
                    self.bilinear(left, right) \
                    + inside_scores[context_list_left] \
                    + inside_scores[context_list_right]
                ) # n_span * batch_size * 1
                context_vec = self.compose_mlp(
                    torch.cat([left, right], dim=2) # n_span * batch_size * (2*size)
                ) # n_span * batch_size * size

                # apply softmax normalization to context_score
                inside_score_weight = torch.softmax(context_score, dim=0) # n_span * batch_size * 1
                # Weighted mean of scores/vecotrs
                inside_score = torch.sum(inside_score_weight * context_score, dim=0) # batch_size * 1
                inside_vec = torch.sum(inside_score_weight * context_vec, dim=0) # batch_size * size
                # Update results
                inside_scores[flatten_tri(begin, end, max_len)] = inside_score
                inside_vecs[flatten_tri(begin, end, max_len)] = inside_vec

        # Outside pass batchified
        # Unlike the inside pass, outside pass should take consider of different sequence lengths
        outside_vecs = torch.zeros([flatten_tri_size(max_len), batch_size, self.size], device=self.device)
        outside_scores = torch.zeros([flatten_tri_size(max_len), batch_size, 1], device=self.device)

        # base case for outside pass
        for i in range(len(sent_lengths)):
            outside_vecs[flatten_tri(0, max_len, max_len), i] = self.root_bias # 1 * size

        for length in range(max_len-1, 0, -1):
            for begin in range(0, max_len+1-length):
                # span: begin ..(length).. end
                end = begin+length
                # Iterate through outside contexts
                # Left context first...
                context_list_parent = [flatten_tri(context, end, max_len) for context in range(0, begin)]
                context_list_sister = [flatten_tri(context, begin, max_len) for context in range(0, begin)]
                # then Right contexts.
                context_list_parent += [flatten_tri(begin, context, max_len) for context in range(end+1, max_len+1)]
                context_list_sister += [flatten_tri(end, context, max_len) for context in range(end+1, max_len+1)]

                # Masking for sequences shorter than max_len
                mask = torch.zeros(len(context_list_parent), batch_size, 1, requires_grad=False, device=self.device, dtype=torch.float32) # n_span * batch_size * 1
                for i, sent_len in enumerate(sent_lengths):
                    mask_list = [0 for context in range(0, begin)]
                    mask_list += [(0 if context<=sent_len else 1) for context in range(end+1, max_len+1)]
                    mask_list = torch.tensor(mask_list, dtype=torch.bool)
                    mask[mask_list, i, :] = -1e10 # mask out spans that exceed the sent_len; not -inf due to softmax-nan issues

                # parent: [context:end]; outside: [context:begin]
                parent = outside_vecs[context_list_parent] # n_span * batch_size * size
                sister = inside_vecs[context_list_sister] # n_span * batch_size * size
                context_score = (
                    self.bilinear(parent, sister)
                    + outside_scores[context_list_parent]
                    + inside_scores[context_list_sister]
                ) # n_span * batch_size * 1
                context_score += mask # mask spans that exceed sent_len
                context_vec =  self.compose_mlp(
                    torch.cat([parent, sister], dim=2) # n_span * batch_size * (2*size)
                ) # n_span * batch_size * size

                # apply softmax normalization to context_score
                outside_score_weight = torch.softmax(context_score, dim=0) # n_span * batch_size * 1
                outside_score_weight = torch.relu(outside_score_weight + mask)
                # torch.nan_to_num_(outside_score_weight, nan=0)
                if torch.any(torch.isnan(context_vec)):
                    # DEBUG
                    print(begin, end)
                    print(parent)
                    print(sister)
                    print(context_vec)
                    print(context_score)
                    print(outside_score_weight)
                    exit()
                # Weighted mean of scores/vecotrs
                outside_score = torch.sum(outside_score_weight * context_score, dim=0) # batch_size * 1
                outside_vec = torch.sum(outside_score_weight * context_vec, dim=0) # batch_size * size
                # Update results
                outside_scores[flatten_tri(begin, end, max_len)] = outside_score
                outside_vecs[flatten_tri(begin, end, max_len)] = outside_vec

                # base case for outside pass (that is not maximum length): set after outside pass computation to prevent override
                if begin == 0:
                    for i, sent_len in enumerate(sent_lengths):
                        if length == sent_len and sent_len != max_len:
                            outside_vecs[flatten_tri(0, sent_len, max_len), i] = self.root_bias # 1 * size
                            outside_scores[flatten_tri(0, sent_len, max_len), i] = 0 # 1 * size

        # Loss function
        # - The original DIORA applies max-margin loss for negative tokens.
        #   However, in our setting we use aggregated representations of subword tokens, thus cannot be directly extractable
        #   Therefore, we use cosine similarity between the recovered outside vector and original word vectors.
        # - Following the DIORA, we use token-wise micro-average, not sentence-wise macro-averaged loss.
        
        terminal_outside_vecs = outside_vecs[:max_len]
        assert terminal_outside_vecs.size() == base_vecs.size()
        loss = 1 - torch.cosine_similarity(terminal_outside_vecs, base_vecs, dim=2) # max_len * batch_size
        loss = torch.sum(loss) # single-element tensor
        loss /= sum(sent_lengths) # normalize

        print("Loss:", loss)
        print("Time(s):", time.time() - start_time)
        return loss
