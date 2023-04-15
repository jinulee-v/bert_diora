from typing import *

import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel

from nltk.tokenize.treebank import TreebankWordTokenizer

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

class BertDora(nn.Module):
    """
    Deep Outside-only Recursive Autoencoders.
    We only use the outside pass, starting from the sentence embedding and comparing the cosine similarity with [MASK] embeddings.
    (bottom-up) in_vectors = ReLU(MLP(mean(bert[i:j])))
    (top-down) out_vectors = [MASK] embeddings

    pool: 'mean' for mean-pooling, or 'bos' for BOS-token sampling([CLS] for bert). 
    """

    def __init__(self, model_id: str='bert-base-uncased', nltk_tok: str="ptb", device = torch.device('cpu'),
                 freeze: bool=True):
        super(BertDora, self).__init__()

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
        # Inside function: MLP (maps mean-pooled word vectors to size-dimension)
        self.inside_mlp = nn.Sequential(
            nn.Linear(self.size, self.size),
            nn.ReLU(),
        ).to(device)
        self.inside_score_mlp = nn.Sequential(
            nn.Linear(self.size, 1),
            nn.Sigmoid(),
        ).to(device)

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Load pretrained transformers
        # (after the initialization to prevent BERT being reset)
        self.model: PreTrainedModel = AutoModel.from_pretrained(model_id).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.freeze = freeze
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        if nltk_tok == "ptb":
            self.nltk_tokenize = TreebankWordTokenizer().tokenize
        else: raise ValueError("nltk_tok must be in ['ptb'].")
    
    def get_nltk_embeddings(self, sentences):
        """
        Obtain embeddings for each NLTK-tokenized tokens, by mean-pooling the Sentencepiece-BERT style subword embeddings.
        """
        # Get the list of NLTK tokens for each sentence
        nltk_tokens_list = [self.nltk_tokenize(sentence) for sentence in sentences]

        # Tokenize the sentences using the tokenizer
        tokenized_sentences = self.tokenizer([' '.join(sentence) for sentence in nltk_tokens_list], padding=True, truncation=True, return_tensors="pt")

        # Convert the tokenized sentences to input_ids and attention_masks
        input_ids = tokenized_sentences.input_ids.to(self.device)
        attention_masks = tokenized_sentences.attention_mask.to(self.device)

        # Generate embeddings using the transformer model
        outputs = self.model(input_ids, attention_mask=attention_masks)
        embeddings = outputs[0]

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

    def get_mask_embeddings(self, sentences):
        """
        Obtain embeddings for each NLTK-tokenized tokens, by mean-pooling the Sentencepiece-BERT style subword embeddings.
        """
        batch_size = len(sentences)

        # Get the list of NLTK tokens for each sentence
        nltk_tokens_list_base = [self.nltk_tokenize(sentence) for sentence in sentences]
        max_len = max(len(sent) for sent in nltk_tokens_list_base)

        result_embeddings = []
        for mask_idx in range(max_len):
            # Mask out i-th token
            nltk_tokens_list = nltk_tokens_list_base[:]
            for sent in nltk_tokens_list:
                sent[mask_idx] = self.mask_token # [MASK]

            # Tokenize the sentences using the tokenizer
            tokenized_sentences = self.tokenizer([' '.join(sentence) for sentence in nltk_tokens_list], padding=True, truncation=True, return_tensors="pt")
            mask_token_index = torch.argmax((tokenized_sentences == self.mask_token_id).to(torch.long))
            # reshape mask_token_index for torch.gather
            mask_token_index = mask_token_index.unsqueeze(0).unsqueeze(2).tile(batch_size, 1, self.size) # batch_size * 1 * size

            # Convert the tokenized sentences to input_ids and attention_masks
            input_ids = tokenized_sentences.input_ids.to(self.device)
            attention_masks = tokenized_sentences.attention_mask.to(self.device)

            # Generate embeddings using the transformer model
            if self.freeze:
                with torch.no_grad():
                    outputs = self.model(input_ids, attention_mask=attention_masks)
            else:
                outputs = self.model(input_ids, attention_mask=attention_masks)
            embeddings = outputs[0]

            # Compute [MASK] embeddings
            mask_embeddings = torch.gather(embeddings, mask_token_index) # batch_size * 1 * size
            result_embeddings.append(mask_embeddings)

        result_embeddings = torch.cat(result_embeddings, dim=1) # batch_size * length * size
        
        return result_embeddings.transpose(0, 1) # seq_len * batch_size * size


    def forward(self, sentences: List[str]):# Tokenize the sentence using NLTK
        """
        Compute inside / outside pass and obtain the loss value.
        """
        start_time = time.time() # DEBUG
        batch_size = len(sentences)

        word_embeddings, sent_lengths = self.get_nltk_embeddings(sentences) # seq_len * batch_size * size
        base_vecs = word_embeddings # To reduce dimension of word vectors, modify here
        max_len = max(sent_lengths)

        # Pseudo-inside pass (simulated)
        word_embeddings, sent_lengths = self.get_nltk_embeddings(sentences) # seq_len * batch_size * size
        inside_vecs = torch.zeros([flatten_tri_size(max_len), batch_size, self.size], device=self.device)
        # base case for inside pass
        inside_vecs[:base_vecs.size(0)] = word_embeddings
        # spans are mean-pooled
        for length in range(max_len-1, 0, -1):
            for begin in range(0, max_len+1-length):
                # span: begin ..(length).. end
                end = begin+length
                inside_vecs[flatten_tri(begin, end, max_len)] = torch.mean(word_embeddings[begin:end], dim=0)
        inside_vecs = self.inside_mlp(inside_vecs) # seq_len * batch_size * size
        inside_scores = self.inside_score_mlp(inside_vecs)

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

        return loss
