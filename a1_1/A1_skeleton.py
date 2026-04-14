
import torch, nltk, pickle
nltk.download('punkt_tab')
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
import matplotlib.pyplot as plt
from transformers import TrainingArguments
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Subset
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """
    with open(train_file, 'r') as f:
        texts = f.readlines()
    tokenized_texts = [tokenize_fun(text) for text in texts]
    counter = Counter()
    for text in tokenized_texts:
        counter.update(text)
    most_common = counter.most_common(max_voc_size)
    vocab = [pad_token, unk_token, bos_token, eos_token] + [token for token, _ in most_common]
    stoi = {token: i for i, token in enumerate(vocab)}
    itos = {i: token for i, token in enumerate(vocab)}
    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    return A1Tokenizer(
        pad_token_id=stoi[pad_token],
        unk_token_id=stoi[unk_token],
        bos_token_id=stoi[bos_token],
        eos_token_id=stoi[eos_token],
        model_max_length=model_max_length,
        str_to_id=stoi,
        id_to_str=itos
    )

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


###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=0, embedding_size=128, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    _tied_weights_keys = []
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_size)
        self.rnn = nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, num_layers=2)
        self.unembedding = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        if not hasattr(self, "all_tied_weights_keys"):
            # Compatibility with newer transformers versions expecting this mapping.
            self.all_tied_weights_keys = {}
        self.post_init()

    def _init_weights(self, module):
        """Required by some transformers versions when post_init() is used."""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)


    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)
        loss = None
        if labels is not None:
            loss = self.loss_func(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

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


"""
Test:
tokenizer = A1Tokenizer.from_file('tokenizer.pkl')
test_texts = ['This is a test.', 'Another test.']
print(tokenizer(test_texts, padding=True, return_tensors='pt'))
"""

def plot_embeddings_pca(emb, inv_voc, words):
    vectors = np.vstack([emb.weight[inv_voc[w]].cpu().detach().numpy() for w in words])
    vectors -= vectors.mean(axis=0)
    twodim = TruncatedSVD(n_components=2).fit_transform(vectors)
    plt.figure(figsize=(5,5))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.02, y, word)
    plt.axis('off')
    plt.savefig('embedding_pca.png')

    
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    dataset = load_dataset('text', data_files={'train': 'train.txt', 'val': 'val.txt'})
    dataset = dataset.filter(lambda x: x['text'].strip() != '') # filter out empty lines
    # tokenizer = build_tokenizer('train.txt', max_voc_size=10000, model_max_length=128)
    # tokenizer.save('tokenizer.pkl')
    tokenizer = A1Tokenizer.from_file('tokenizer.pkl')
    
    for sec in ['train', 'val']:
        dataset[sec] = Subset(dataset[sec], range(1000))

    model = A1RNNModel(A1RNNModelConfig(vocab_size=len(tokenizer), embedding_size=128, hidden_size=256))
    model = model.from_pretrained('/Users/wangzi/Downloads/trainer_output')
    trainer = A1Trainer(
        model=model,
        args=TrainingArguments(
            optim='adamw_torch',
            eval_strategy='epoch',
            use_cpu=True,
            learning_rate=1e-3,
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=256,
            output_dir='trainer_output',
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer
    )
    

    # trainer.train()
    # print('Perplexity:', trainer.val())

    test_text = ["She lives in San"]
    test_input_ids = tokenizer(test_text, truncation=True, padding=True, return_tensors='pt')['input_ids'].to('cpu')
    print(test_input_ids.shape)
    with torch.no_grad():
        model.eval()
        output = model(input_ids=test_input_ids[:, :-1])
    print(output.logits.shape)
    topk_idx = output.logits[0, -1].topk(10).indices
    print([tokenizer.id_to_str[idx.item()] for idx in topk_idx])
    


    plot_embeddings_pca(model.embedding, tokenizer.str_to_id, ['sweden', 'denmark', 'europe', 'africa', 'london', 'stockholm', 'large', 'small', 'great', 'black', '3', '7', '10', 'seven', 'three', 'ten', '1984', '2005', '2010'])
    # idx = output.logits.argmax(dim=-1)[0, -1].item()
    # print(tokenizer.id_to_str[idx])