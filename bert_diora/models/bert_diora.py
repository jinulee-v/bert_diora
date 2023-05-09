from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig, PreTrainedModel

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk import Tree

torch.autograd.set_detect_anomaly(True)
import time

def flatten_tri_size(max_length):
    return (max_length + 1) * max_length // 2

def flatten_tri(begin, end, max_length):
    """
    Element-wise torch operation. begin: Tensor, length/max_length = integer.

    Flattens the index (begin, end) given the max_length.
    max_length: 6 -> linear size: 21 (6 * 7 / 2)
    structure:
    20                (len=6)
    18 19             (len=5)
    15 16 17          (len=4)
    11 12 13 14       (len=3)
     6  7  8  9 10    (len=2)
     0  1  2  3  4  5 (len=1)
    """
    length = end - begin
    return (max_length + 1) * max_length // 2 - (max_length - length + 2) * (max_length - length + 1) // 2   \
           + begin

class BertDiora(nn.Module):
    """
    Re-implementation of DIORA using pretrained transformers (e.g. BERT).
    """

    def __init__(self, model_id: str='bert-base-uncased', freeze: bool=True, nltk_tok: str="ptb", device = torch.device('cpu'), loss: str="cossim", loss_margin_k: int=20, loss_margin_lambda=1):
        super(BertDiora, self).__init__()

        # Retrieve size first
        config = AutoConfig.from_pretrained(model_id)
        self.size = config.hidden_size
        self.device = device
        
        self.word_linear = nn.Sequential(
            nn.Linear(self.size, self.size),
            nn.Tanh()
        )
        # Compose function: MLP
        self.compose_mlp = nn.Sequential(
            nn.Linear(self.size*2, self.size),
            nn.ReLU(),
            nn.Linear(self.size, self.size),
            nn.ReLU()
        )
        # Score function: bilinear. Following the DIORA paper, we set bias to False.
        self.bilinear = nn.Bilinear(self.size, self.size, 1, bias=False)
        # Root bias for outside pass
        self.root_bias = nn.Parameter(torch.zeros(self.size))

        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Load pretrained transformers
        # (after the initialization to prevent BERT being reset)
        self.model: PreTrainedModel = AutoModelWithLMHead.from_pretrained(model_id, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.freeze = freeze
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        if nltk_tok == "ptb":
            self.nltk_tokenize = TreebankWordTokenizer().tokenize
        else: raise ValueError("nltk_tok must be in ['ptb'].")

        self.loss = loss
        self.loss_margin_k = loss_margin_k
        self.loss_margin_lambda = loss_margin_lambda
        if not loss in ["cossim", "token_ce", "token_margin"]:
            raise ValueError("loss must be in ['cossim', 'token_ce', 'token_margin']")
    
    def get_nltk_embeddings(self, sentences):
        """
        Obtain embeddings for each NLTK-tokenized tokens, by mean-pooling the Sentencepiece-BERT style subword embeddings.
        """
        # Get the list of NLTK tokens for each sentence
        nltk_tokens_list = [sentence.split() for sentence in sentences]

        # Tokenize the sentences using the tokenizer
        tokenized_sentences = self.tokenizer([' '.join(sentence) for sentence in nltk_tokens_list], padding=True, truncation=True, return_tensors="pt")

        # Convert the tokenized sentences to input_ids and attention_masks
        input_ids = tokenized_sentences.input_ids.to(self.device)
        attention_masks = tokenized_sentences.attention_mask.to(self.device)

        # Generate embeddings using the transformer model
        outputs = self.model(input_ids, attention_mask=attention_masks, output_hidden_states=True).hidden_states
        embeddings = outputs[-1]

        # Match the transformer subwords to the NLTK tokens
        index_map_list = []
        single_span_tokens_list = []
        for nltk_tokens in nltk_tokens_list:
            token_ids_list = self.tokenizer.batch_encode_plus(nltk_tokens, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            index_map = []
            single_span_tokens = []
            for i, token_id in enumerate(token_ids_list):
                # Keep track of the mapping between NLTK token and transformer token ids
                nltk_token = self.tokenizer.decode(token_id).split()
                index_map += [i] * len(nltk_token)
                if len(token_id) == 1:
                    single_span_tokens.append(token_id[0])
                else:
                    single_span_tokens.append(-1)

            index_map += [-1] * (embeddings.size(1) - len(index_map)) # Pad the index_map to max_len(subword)
            index_map_list.append(torch.tensor(index_map))
            single_span_tokens_list.append(torch.tensor(single_span_tokens))

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

        return nn.utils.rnn.pad_sequence(result_embeddings), \
               [sent_emb.size(0) for sent_emb in result_embeddings], \
               nn.utils.rnn.pad_sequence(single_span_tokens_list).to(self.device)

    def _inside(self, inside_vecs, inside_scores, max_len, batch_size, argmax=False):

        # Argmax instead of weighted mean pooling, as in CYK parsing / S-DIORA.
        # in such use cases, we record and return the backpointer for each step.
        if argmax:
            backpointer = torch.zeros((flatten_tri_size(max_len), batch_size, 2), dtype=torch.long)

        for length in range(2, max_len+1):
            num_spans = max_len+1-length
            num_contexts = length - 1

            begin = torch.arange(0, num_spans, 1, device=self.device)
            end = begin + length
            context_list_left = flatten_tri(
                begin.unsqueeze(1), # num_spans * 1
                begin.unsqueeze(1) + torch.arange(1, num_contexts+1, 1, device=self.device).unsqueeze(0), # num_spans * num_contexts
                max_len
            ).reshape(-1) # 1-D tensor of num_spans * num_contexts
            context_list_right = flatten_tri(
                begin.unsqueeze(0) + torch.arange(1, num_contexts+1, 1, device=self.device).unsqueeze(1), # num_contexts * num_spans
                end.unsqueeze(0), # 1 * num_spans
                max_len
            ).transpose(0, 1).reshape(-1) # 1-D tensor of num_spans * num_contexts
            left = inside_vecs[context_list_left]
            right = inside_vecs[context_list_right]

            # Calculate inside vector/scores
            context_score = (
                self.bilinear(left, right) \
                + inside_scores[context_list_left] \
                + inside_scores[context_list_right]
            ).reshape(num_spans, num_contexts, batch_size, 1)
            context_vec = self.compose_mlp(
                torch.cat([left, right], dim=2) # n_span * batch_size * (2*size)
            ).reshape(num_spans, num_contexts, batch_size, self.size)
            
            if not argmax:
                # Weighted mean pooling of scores/vectors
                # - apply softmax normalization to context_score
                inside_score_weight = torch.softmax(context_score, dim=1) # num_spans * num_contexts * batch_size * 1
                # - Weighted mean of scores/vecotrs
                inside_score = torch.sum(inside_score_weight * context_score, dim=1) # num_spans * batch_size * 1
                inside_vec = torch.sum(inside_score_weight * context_vec, dim=1) # num_spans * batch_size * self.size
            else:
                # argmax pooling of scores/vectors
                argmax_index = torch.argmax(context_score, dim=1, keepdim=True) # num_spans * 1 * batch_size * 1
                inside_score = torch.gather(context_score, dim=1, index=argmax_index).squeeze(1)
                inside_vec = torch.gather(context_vec, dim=1, index=argmax_index).squeeze(1)
                # backtracking
                argmax_index_backpointer = argmax_index.squeeze(3) # num_spans * 1 * batch_size
                context_list_left = context_list_left.reshape(num_spans, num_contexts, 1).tile(1, 1, batch_size) # num_spans * num_contexts * batch_size
                context_list_right = context_list_right.reshape(num_spans, num_contexts, 1).tile(1, 1, batch_size) # num_spans * num_contexts * batch_size
                # print(torch.gather(context_list_left, dim=1, index=argmax_index_backpointer).squeeze(1))
                # print(torch.gather(context_list_right, dim=1, index=argmax_index_backpointer).squeeze(1))
                # print("=====================")
                backpointer[flatten_tri(begin, end, max_len), :, 0] = \
                    torch.gather(context_list_left, dim=1, index=argmax_index_backpointer).squeeze(1)
                backpointer[flatten_tri(begin, end, max_len), :, 1] = \
                    torch.gather(context_list_right, dim=1, index=argmax_index_backpointer).squeeze(1)
            
            # Update results
            inside_vec = F.normalize(inside_vec, dim=2) # normalize to unit vector
            inside_scores[flatten_tri(begin, end, max_len)] = inside_score
            inside_vecs[flatten_tri(begin, end, max_len)] = inside_vec

        if argmax:
            return backpointer

    def _outside(self, outside_vecs, outside_scores, inside_vecs, inside_scores, max_len, batch_size):
        for length in range(max_len-1, 0, -1):
            # We make two diagonal matrices, and then sum up.
            # For the following chart,
            #   a
            #   b c
            #   d e f
            # to batchify outside for [d, e, f]:
            # - we make two matrices for parents/sisters(left/right sisters),
            # - zero-mask them with upper/lower triangle matrices,
            # - sum the two.
            # (Left: close to far; Right: far to close indexing)
            # Parents   Left  Right   Sisters   Left  Right
            #        d: - -   a b            d: - -   c e
            #        e: b -   - c            e: d -   - f
            #        f: c a   - -            f: b e   - -
            # 
            num_spans = max_len+1-length
            num_contexts = max_len - length

            begin = torch.arange(0, num_spans, 1, device=self.device)
            end = begin + length
            context_list_parent = (
                torch.tril(flatten_tri( # Left parents
                    begin.unsqueeze(0) - torch.arange(1, num_contexts+1, 1, device=self.device).unsqueeze(1), # num_contexts * num_spans
                    end.unsqueeze(0), # 1 * num_spans
                    max_len
                ).transpose(0, 1), diagonal=-1) +
                torch.triu(flatten_tri( # Right parents
                    begin.unsqueeze(1), # num_spans * 1
                    end.unsqueeze(1) + torch.arange(num_contexts, 0, -1, device=self.device).unsqueeze(0), # num_spans * num_contexts
                    max_len
                ), diagonal=0)
            ).reshape(-1) # 1-D tensor of num_spans * num_contexts
            context_list_sister = (
                torch.tril(flatten_tri( # Left sisters
                    begin.unsqueeze(0) - torch.arange(1, num_contexts+1, 1, device=self.device).unsqueeze(1), # num_contexts * num_spans
                    begin.unsqueeze(0), # 1 * num_spans
                    max_len
                ).transpose(0, 1), diagonal=-1) +
                torch.triu(flatten_tri( # Right sisters
                    end.unsqueeze(1), # num_spans * 1
                    end.unsqueeze(1) + torch.arange(num_contexts, 0, -1, device=self.device).unsqueeze(0), # num_spans * num_contexts
                    max_len
                ), diagonal=0)
            ).reshape(-1) # 1-D tensor of num_spans * num_contexts
            parents = outside_vecs[context_list_parent]
            sisters = inside_vecs[context_list_sister]

            # Calculate outside vector/scores
            context_score = (
                self.bilinear(parents, sisters) \
                + outside_scores[context_list_parent] \
                + inside_scores[context_list_sister]
            ).reshape(num_spans, num_contexts, batch_size, 1)
            context_vec = self.compose_mlp(
                torch.cat([parents, sisters], dim=2) # n_span * batch_size * (2*size)
            ).reshape(num_spans, num_contexts, batch_size, self.size)
            
            # apply softmax normalization to context_score
            outside_score_weight = torch.softmax(context_score, dim=1) # num_spans * length-1 * batch_size * 1
            # Weighted mean of scores/vecotrs
            outside_score = torch.sum(outside_score_weight * context_score, dim=1) # num_spans * batch_size * 1
            outside_vec = torch.sum(outside_score_weight * context_vec, dim=1) # num_spans * batch_size * self.size
            outside_vec = F.normalize(outside_vec, dim=2) # normalize to unit vector
            # Update results
            outside_scores[flatten_tri(begin, end, max_len)] = outside_score
            outside_vecs[flatten_tri(begin, end, max_len)] = outside_vec

    def forward(self, sentences: List[str]):# Tokenize the sentence using NLTK
        """
        Compute inside / outside pass and obtain the loss value.
        """
        start_time = time.time() # DEBUG
        batch_size = len(sentences)

        word_embeddings, sent_lengths, single_tokens = self.get_nltk_embeddings(sentences) # seq_len * batch_size * size
        base_vecs = word_embeddings # To reduce dimension of word vectors, modify here
        max_len = max(sent_lengths)

        ##### Inside pass, batchified #####
        inside_vecs = torch.zeros([flatten_tri_size(max_len), batch_size, self.size], device=self.device)
        inside_scores = torch.zeros([flatten_tri_size(max_len), batch_size, 1], device=self.device)
        # base case for inside pass
        inside_vecs[:base_vecs.size(0)] = F.normalize(self.word_linear(base_vecs), dim=2)
        self._inside(inside_vecs, inside_scores, max_len, batch_size)

        ##### Outside pass batchified #####
        # Unlike the inside pass, outside pass should take consider of different sequence lengths
        outside_vecs = torch.zeros([flatten_tri_size(max_len), batch_size, self.size], device=self.device)
        outside_scores = torch.zeros([flatten_tri_size(max_len), batch_size, 1], device=self.device)
        # base case for outside pass
        for i in range(len(sent_lengths)):
            outside_vecs[flatten_tri(0, max_len, max_len), i] = F.normalize(self.root_bias, dim=0) # 1 * size
        self._outside(outside_vecs, outside_scores, inside_vecs, inside_scores, max_len, batch_size)

        ##### Loss function #####
        # - The original DIORA applies max-margin loss for negative tokens.
        #   However, in our setting we use aggregated representations of subword tokens, thus cannot be directly extractable
        #   Therefore, we use cosine similarity between the recovered outside vector and original word vectors.
        # - Following the DIORA, we use token-wise micro-average, not sentence-wise macro-averaged loss.
        
        terminal_outside_vecs = outside_vecs[:max_len]
        assert terminal_outside_vecs.size() == base_vecs.size()
        
        single_tokens = single_tokens.unsqueeze(2) # max_len * batch_size * 1, resize to apply torch.gather()
        single_tokens_mask = single_tokens != -1
        single_tokens = single_tokens * single_tokens_mask # To remove all -1 indices

        if self.loss == "cossim":
            # Cossine similarity loss
            loss = 1 - torch.cosine_similarity(terminal_outside_vecs, base_vecs, dim=2) # max_len * batch_size
            loss = torch.sum(loss) # single-element tensor
            loss /= sum(sent_lengths) # normalize
        elif self.loss == "token_ce":
            # Token cross-entropy, recycling the LM_head of the transformer
            token_probs = torch.log_softmax(self.model.cls(terminal_outside_vecs), dim=2)
            # Extract token probs, but only when a NLTK token maps to a single BERT token
            token_probs = torch.gather(token_probs, 2, single_tokens)
            token_probs = (- token_probs) * single_tokens_mask
            # compute loss
            loss = torch.sum(token_probs) / torch.sum(single_tokens_mask)
        elif self.loss == "token_margin":
            # Token max_margin, recycling the LM_head of the transformer
            token_probs = torch.log_softmax(self.model.cls(terminal_outside_vecs), dim=2)
            margin_tokens = torch.randint(0, self.tokenizer.vocab_size, (max_len, batch_size, self.loss_margin_k)).to(self.device)
            margin_tokens_mask = single_tokens_mask * (margin_tokens != single_tokens).to(torch.long)
            # Extract token probs, but only when a NLTK token maps to a single BERT token
            margin_probs = torch.gather(token_probs, 2, margin_tokens)
            token_probs = torch.gather(token_probs, 2, single_tokens)
            margin = (self.loss_margin_lambda - token_probs + margin_probs) * margin_tokens_mask
            margin = torch.relu(margin)
            # compute loss
            loss = torch.sum(margin) / torch.sum(margin_tokens_mask)
        return loss

    def parse(self, sentences: List[str]):
        batch_size = len(sentences)
        # Get the list of NLTK tokens for each sentence
        nltk_tokens_list = [self.nltk_tokenize(sentence) for sentence in sentences]

        word_embeddings, sent_lengths, _ = self.get_nltk_embeddings(sentences) # seq_len * batch_size * size
        base_vecs = word_embeddings # To reduce dimension of word vectors, modify here
        max_len = max(sent_lengths)

        # CYK algorithm (very much similar to the inside pass), batchified
        inside_vecs = torch.zeros([flatten_tri_size(max_len), batch_size, self.size], device=self.device)
        inside_scores = torch.zeros([flatten_tri_size(max_len), batch_size, 1], device=self.device)
        # base case for inside pass
        inside_vecs[:base_vecs.size(0)] = F.normalize(base_vecs, dim=2)

        # Fill the inside vectors first
        backpointer = self._inside(inside_vecs, inside_scores, max_len, batch_size, argmax=True)

        trees = []
        # Reveal the parse tree by backtracking
        for i, length in enumerate(sent_lengths):
            backtrack_ptr = backpointer[:, i, :].cpu()
            tokens = nltk_tokens_list[i]
            def backtrack(idx):
                if idx < max_len:
                    # Punctuation labels for proper removal
                    label = "_"
                    if tokens[idx] in ".,?!":
                        label = tokens[idx]
                    elif tokens[idx] in ":;—…":
                        label = ":"
                    elif tokens[idx] in "‘“":
                        label = "''"
                    elif tokens[idx] in "\'\"":
                        label = "``"
                    elif tokens[idx] in "-":
                        label = "HYPH"
                    return Tree(label, [tokens[idx]]) # Base case
                else:
                    return Tree('_', [backtrack(backtrack_ptr[idx][0]), backtrack(backtrack_ptr[idx][1])])
            trees.append(backtrack(flatten_tri(0, length, max_len)))
        return trees
