
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import nltk
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False) # *2 for SwiGLU
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states_left, hidden_states_right = self.fc1(hidden_states).chunk(2, dim=-1)
        hidden_states = hidden_states_left * torch.nn.functional.silu(hidden_states_right)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config, hidden_size):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = hidden_size
        self.gamma = nn.Parameter(torch.ones(self.hidden_size))
        nn.init.ones_(self.gamma)

    def forward(self, hidden_states):
        # batch, seq, embed
        rms = torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        return hidden_states * rms * self.gamma


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # TODO: set up W_q, W_k, W_v, W_o here
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False) # for W_q, W_k, W_v
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # for W_o
        # TODO: set up normalizers here
        self.rms_norm_q = A2RMSNorm(config, self.hidden_size // self.num_attention_heads)
        self.rms_norm_k = A2RMSNorm(config, self.hidden_size // self.num_attention_heads)

    def forward(self, hidden_states, rope_rotations):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        batch_size, seq_length, _ = hidden_states.size()
        head_dim = self.hidden_size // self.num_attention_heads
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1) 
        # batch, seq, num_heads, head_dim
        q = q.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2) # batch, num_heads, seq, head_dim
        k = k.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2) # batch, num_heads, seq, head_dim
        v = v.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2) # batch, num_heads, seq, head_dim
        q = self.rms_norm_q(q)
        k = self.rms_norm_k(k)
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5) # batch, num_heads, seq, seq
        attn_scores = attn_scores.masked_fill(
            torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool(), float('-inf')
        )
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_scores, v) # batch, num_heads, seq, head_dim
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size) # batch, seq, hidden
        attn_output = self.o_proj(attn_output) # batch, seq, hidden
        return attn_output


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.masked_attn = A2Attention(config)
        self.mlp = A2MLP(config)
        self.rms_norm_attn = A2RMSNorm(config, config.hidden_size)
        self.rms_norm_mlp = A2RMSNorm(config, config.hidden_size)

    def forward(self, hidden_states, rope_rotations):
        attn_output = self.masked_attn(hidden_states, rope_rotations)
        hidden_states = self.rms_norm_attn(attn_output) + hidden_states
        mlp_output = self.mlp(hidden_states)   
        hidden_states = self.rms_norm_mlp(mlp_output) + hidden_states
        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        self.num_layer = config.num_hidden_layers
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # TODO: put all transformer decoder layers in a ModuleList.
        self.layers = nn.ModuleList([A2DecoderLayer(config) for _ in range(self.num_layer)])
        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids, labels=None):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers
        embedding = self.embedding(input_ids)
        for i in range(self.num_layer):
            embedding = self.layers[i](embedding, rope_rotations)   
        unembedding = self.unembedding(embedding)
        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        # TODO: Compute the loss as in Assignment 1 if labels is not None.
        return unembedding


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(dataset=self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
        #       labels = input_ids with padding replaced by -100
	    #       put input_ids and labels onto the GPU (or whatever device you use)
        #       apply the model to input_ids and labels
        #       get the loss from the model output
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()
        train_loss_list = []
        val_loss_list = []
        for epoch in tqdm(range(args.num_train_epochs)):
            tmp_train_loss = 0
            tmp_val_loss = 0
            for batch in train_loader:
                input_ids = self.tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
                labels = input_ids.clone()
                input_ids = input_ids[:, :-1]
                labels = labels[:, 1:]
                labels[labels == self.tokenizer.pad_token_id] = -100
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = loss_func(outputs.logits.reshape(-1, self.model.config.vocab_size), labels.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tmp_train_loss += loss.item()
            train_loss_list.append(tmp_train_loss / len(train_loader))
            for batch in val_loader:
                input_ids = self.tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
                labels = input_ids.clone()
                input_ids = input_ids[:, :-1]
                labels = labels[:, 1:]
                labels[labels == self.tokenizer.pad_token_id] = -100
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = loss_func(outputs.logits.reshape(-1, self.model.config.vocab_size), labels.reshape(-1))
                    tmp_val_loss += loss.item()
            tmp_val_loss /= len(val_loader)
            val_loss_list.append(tmp_val_loss)

        plt.plot(train_loss_list, label='train loss') 
        plt.plot(val_loss_list, label='val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
            

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    def val(self):
        total_loss = 0
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        val_loader = DataLoader(dataset=self.eval_dataset, batch_size=self.args.per_device_eval_batch_size, shuffle=False)
        for batch in tqdm(val_loader):
            input_ids = self.tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt')['input_ids'].to(self.select_device())
            labels = input_ids.clone()
            input_ids = input_ids[:, :-1]
            labels = labels[:, 1:]
            labels[labels == self.tokenizer.pad_token_id] = -100
            with torch.no_grad():
                input_ids = input_ids[:, :-1]
                labels = labels[:, 1:]
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = loss_func(outputs.logits.reshape(-1, self.model.config.vocab_size), labels.reshape(-1))
                total_loss += loss.item()
        total_loss /= len(val_loader)
        print('Validation loss:', total_loss)
        perplexity = np.exp2(total_loss)
        return perplexity


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, pad_token_id, unk_token_id, bos_token_id, eos_token_id, 
                 model_max_length, str_to_id, id_to_str):
        # TODO: store all values you need in order to implement __call__ below.
        self.pad_token_id = pad_token_id     # Compulsory attribute.
        self.unk_token_id = unk_token_id     # Compulsory attribute.
        self.bos_token_id = bos_token_id     # Compulsory attribute.
        self.eos_token_id = eos_token_id     # Compulsory attribute.
        self.model_max_length = model_max_length # Needed for truncation.
        self.str_to_id = str_to_id  # Needed for mapping tokens to integer values.  
        self.id_to_str = id_to_str  # Needed for mapping integer values to tokens.

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        input_ids = []
        max_len = 0
        for text in texts:
            # text = '<BOS> ' + text + ' <EOS>'
                # text = text[:self.model_max_length]
            tokenized = [self.str_to_id.get(token, self.unk_token_id) for token in lowercase_tokenizer(text)]
            tokenized = [self.bos_token_id] + tokenized + [self.eos_token_id]
            input_ids.append(tokenized[:self.model_max_length])
            max_len = max(max_len, len(tokenized))
        
        mask_tensor = torch.zeros((len(texts), max_len), dtype=torch.long)
        if padding:
            for i in range(len(input_ids)):
                mask_tensor[i, :len(input_ids[i])] = 1
                input_ids[i] = input_ids[i] + [self.pad_token_id] * (max_len - len(input_ids[i]))


        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        return BatchEncoding({'input_ids': input_ids, 'attention_mask': mask_tensor})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_id)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


def topk_sampling(model, prompt, max_len, topk, tokenizer):
    """Generate text autoregressively using top-k sampling.
       
       Args:
         model: The language model to use for generation.
         prompt: The initial text prompt to start the generation from.
         max_len: The maximum length of the generated sequence (including the prompt).
         topk: The number of highest-probability tokens to keep for sampling at each step.
         tokenizer: The tokenizer to use for encoding the prompt and decoding the generated tokens.

       Returns:
         The generated text (including the prompt).
    """
    model.eval()
    input_ids = torch.tensor(tokenizer(prompt, return_tensors='pt')['input_ids'])[:, :-1] # remove the <EOS> token
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_len - input_ids.shape[1]):
            outputs = model(generated)
            next_token_logits = outputs[:, -1, :]
            topk_logits, topk_indices = torch.topk(next_token_logits, topk)
            probabilities = torch.softmax(topk_logits, dim=-1)
            next_token = topk_indices[0, torch.multinomial(probabilities, num_samples=1)[0]]
            print(next_token)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    text = [tokenizer.id_to_str[idx.item()] for idx in generated[0]]
    return text


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    tokenizer = A1Tokenizer.from_file('tokenizer.pkl')
    config = A2ModelConfig(
        vocab_size=len(tokenizer.str_to_id),
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_hidden_layers=4,
        rope_theta=100000.0,
        rms_norm_eps=torch.finfo(torch.float32).eps
    )
    test_sentence = ['This is a test.']
    test_tensor = torch.tensor(tokenizer(test_sentence, return_tensors='pt')['input_ids'])
    
    model = A2Transformer(config)
    output = model(test_tensor)
    print(output.shape)

    print(topk_sampling(model, ['The meaning of life is'], max_len=10, topk=5, tokenizer=tokenizer))
