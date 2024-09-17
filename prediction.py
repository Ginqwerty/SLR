# Import Library
import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import os
import time
from tqdm import tqdm

############################################################
# PositionWise Feed-Forward Networks #######################
class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# Residual Connection and Layer Normalization
class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

############################################################
# Encoder ##################################################
class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder_NoEmbedding(d2l.Encoder):
    def __init__(self, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.ffn_num_hiddens = ffn_num_hiddens
        self.projection = nn.Linear(768, num_hiddens) # set the input_dim as 768
        self.input_dim = None  # Placeholder for input dimension
        # self.projection = None  # Initialize projection as None
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block" + str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        device = X.device  # Ensure to get the device from X
        print("the X's device is :", device)

        # Skip embedding if you're using features
        if self.input_dim is None:
            self.input_dim = X.shape[-1]  # Set input_dim from input tensor
        #     self.projection = nn.Linear(self.input_dim, self.num_hiddens).to(device)  # Reinitialize projection

        # print(f"Input shape: {X.shape}")  # Print input shape
        # print(f"X values before projection: {X}")
        # print(f"X dtype before projection: {X.dtype}")
        # print("num_hiddens: ", self.num_hiddens)
        # print("input_dim: ", self.input_dim)
        # print("ffn_num_hiddens: ", self.ffn_num_hiddens)
        # Before moving to device

        # projection = nn.Linear(self.input_dim, self.num_hiddens)
        X = self.projection(X).to(device)  # Project the feature dimension to num_hiddens
        # print(f"Input shape after projection: {X.shape}")  # Print input shape after projection

        # After moving to device
        # print(f"After projection layer, the X's device: {X.device}")

        X = self.pos_encoding(X * math.sqrt(self.num_hiddens)).to(device)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X.to(device), valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

############################################################
# Decoder ##################################################
class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

############################################################
# Sign to Eng Class ########################################
'''
    Modifications made according to d2l's MTFraEng to adapt the translation of Sign to a specific language
'''
class signEng(d2l.DataModule):
    """The dataset with ViT features and English sentences."""
    def _load_features_and_sentences(self, features_dir, sentences_file):
        """Load frame features from .pt files in a directory and sentences from a text file."""
        features = []

        # Load frame features from .pt files
        for file_name in sorted(os.listdir(features_dir)):
            if file_name.endswith('.pt'):
                feature_path = os.path.join(features_dir, file_name)
                feature = torch.load(feature_path, weights_only=True)
                features.append(feature)

        # Load English sentences from text file as a single string
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = f.read().strip()  # Read as a single string

        return features, sentences

    def _preprocess(self, text):
        """Process text to handle sentence pairs and punctuation."""
        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        # print("no_space: ", no_space)
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
              for i, char in enumerate(text.lower())]
        # print("out: ", out[:50])
        # processed_text = ''.join(out)
        return ''.join(out)

    def _tokenize(self, text, max_examples=None):
        """Tokenize English sentences for decoder input."""
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples: break
            #print("line: ", line)
            tgt.append([t for t in f'{line} <eos>'.split(' ') if t])
            #break
        return src, tgt

    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128, features_dir='features', sentences_file='sentences.txt'):
        """Initialize with given batch size and other parameters."""
        super(signEng, self).__init__()
        self.save_hyperparameters()
        self.features_dir = features_dir
        self.sentences_file = sentences_file
        features, sentences = self._load_features_and_sentences(features_dir, sentences_file)
        print(f"Features length: {len(features)}")
        print(f"sentences length: {len(sentences)}")
        # 先不call build arrays
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(features, sentences)

    def _build_arrays(self, features, sentences, src_vocab=None, tgt_vocab=None):
        """Build arrays from frame features and English sentences."""
        def _build_array(sentences, vocab, is_tgt=False):
            # print("input sentences before padding: ", sentences)
            '''
            # The _build_array function defines an anonymous function of pad_or_trim that takes a sentence seq and the target length t as arguments.
            # If the length of the sentence is greater than t, it will truncate the first t elements of the sentence.
            # lt is padded after a sentence if its length is less than t until the length of the sentence reaches t
            # print("sentences: ", sentences)
            '''
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
            sentences = [pad_or_trim(s, 9) for s in sentences]
            print("sentences.shape: ", len(sentences))
            #print("sentences[1]: ", sentences[1])

            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab = d2l.Vocab(sentences, min_freq=1)
                # for idx, token in enumerate(vocab.idx_to_token):
                #     if idx < 10:  # 查看前10个词汇
                #         print(f"Index {idx}: {token}")

            array = d2l.tensor([vocab[s] for s in sentences])

            # Debug: print array shape
            #print(f"Array shape: {array.shape}")
            #print("Array: ", array)
            valid_len = d2l.reduce_sum( #有效长度（valid_len）表示句子中非 <pad> 元素的数量
                d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
            return array, vocab, valid_len

        # Padding features to the same length
        #print("features: ", features)
        max_len = max(len(f) for f in features)
        src_valid_len = []
        #print("max_len: ", max_len)
        feature_dim = features[0].size(1)  # Dimension of the feature vectors
        #print("feature_dim: ", feature_dim)
        features_padded = []
        for f in features:
            src_valid_len.append(len(f))
            if len(f) < max_len:
                padding = torch.zeros((max_len - len(f), feature_dim), dtype=f.dtype, device=f.device)
                padded_feature = torch.cat([f, padding], dim=0)
            else:
                padded_feature = f
            features_padded.append(padded_feature)
        features_tensor = torch.stack(features_padded)
        print("Stacked features_tensor.shape: ", features_tensor.shape)
        #print(features_tensor)

        # print("sentences: ", sentences)
        src, tgt = self._tokenize(self._preprocess(sentences))
        src_valid_len_tensor = torch.tensor(src_valid_len, dtype=torch.int64)

        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
        src_array, src_vocab, src_valid_len = features_tensor, None, src_valid_len_tensor
       
        # 检查是否是相同类型
        #if src_valid_len.dtype == _.dtype:
        #    print("Both Tensors have the same data type")
        #else:
        #    print("Two Tensors have the different data type")
        #    print("src_valid_len.dtype: ", src_valid_len.dtype)
        #    print("_.dtype: ", _.dtype)
	
 	# Check src and tgt arraies have same length
        print("src_array_length: ", len(src_array))
        print("tgt_array_length: ", len(tgt_array))
        return ((src_array, tgt_array[:, :-1], src_valid_len, tgt_array[:, 1:]),
                src_vocab, tgt_vocab)

    def get_dataloader(self, train):
        """Get dataloader with frame features for encoder and English sentences for decoder."""
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)

    def check_get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        dataloader = self.get_tensorloader(self.arrays, train, idx)
        for batch in dataloader:
            print(f"Batch: {batch}")
            break  # Inspect the first batch

    def build(self, src_sentences, tgt_sentences):
        """Build arrays with frame features and English sentences."""
        # features, sentences = self._load_features_and_sentences(self.features_dir, self.sentences_file)
        sentences_str = "\n".join(tgt_sentences).strip()
        arrays, _, _ = self._build_arrays(features, sentences)
        return arrays

############################################################
# re-define Trainer (Store training loss value) ############
class Trainer(d2l.HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""
    # def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    #     self.save_hyperparameters()
    #     assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        # Debugging outputs
        print(f"Number of training batches: {len(self.train_dataloader)}")
        print(f"Number of validation batches: {len(self.val_dataloader)}")
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train()
        num_batches = len(self.train_dataloader)
        start_time = time.time()  # 记录开始时间
        # 使用tqdm显示进度条
        with tqdm(total=num_batches, desc=f"Epoch [{self.epoch + 1}/{self.max_epochs}]", unit="batch") as pbar:
            for batch in self.train_dataloader:
                loss = self.model.training_step(self.prepare_batch(batch))
                self.train_losses.append(loss.item())  # 记录训练损失
                #print("training loss.item(): ", loss.item())
                self.optim.zero_grad()
                with torch.no_grad():
                    loss.backward()
                    if self.gradient_clip_val > 0:  # To be discussed later
                        self.clip_gradients(self.gradient_clip_val, self.model)
                    self.optim.step()
                torch.cuda.empty_cache()  # Clear cache
                self.train_batch_idx += 1
                pbar.update(1)  # 更新进度条
        
        # 记录整个epoch的时间
        epoch_time = time.time() - start_time
        print(f"Epoch [{self.epoch + 1}/{self.max_epochs}] completed in {epoch_time:.2f} seconds.")

        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
        torch.cuda.empty_cache()  # Clear cache

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """Defined in :numref:`sec_use_gpu`"""
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
        self.train_losses = []  # 用于记录训练损失
        self.val_losses = []    # 用于记录验证损失
    

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        if self.gpus:
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
            # Check if any tensor is not on GPU
            #for tensor in batch:
            #    if tensor.is_cuda:
            #        print(f"Tensor {tensor.shape} is on GPU")
            #    else:
            #        print(f"Tensor {tensor.shape} is not on GPU")
        return batch
    
    def prepare_model(self, model):
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model
        #for param in model.parameters():
        #    print(param.device) # Print the device of each parameter

    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

def load_testing_features_and_sentences(features_dir, sentences_file):
    """Load frame features from .pt files in a directory and sentences from a text file."""
    features = []

    # Load frame features from .pt files
    for file_name in sorted(os.listdir(features_dir)):
        if file_name.endswith('.pt'):
            feature_path = os.path.join(features_dir, file_name)
            feature = torch.load(feature_path, weights_only=True)
            features.append(feature)

    # Load English sentences from text file as a single string
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()   # Read as a single string

    return features, sentences

############################################################
# Prediction Step ############################################
'''
Build Model 
'''
batch_size = 8
num_steps = 9
num_train = 512
num_val = 128
#features_dir = '/home/streetparking/SLR/NewPheonixSampleFeatures'
features_dir = '/home/streetparking/SLR/paddedTrainingVideoFeaturesGPU'
#features_dir = '/home/streetparking/SLR/paddedDevingVideoFeaturesGPU'

#sentences_file = '/home/streetparking/SLR/germen_sentences.txt'
sentences_file = '/home/streetparking/SLR/trainingTranslation.txt'
#sentences_file = '/home/streetparking/SLR/devingTranslation.txt'
 
signdata = signEng(batch_size=batch_size, num_steps=num_steps, num_train=num_train, num_val=num_val,
                features_dir=features_dir, sentences_file=sentences_file)
num_hiddens, num_blks, dropout = 256, 4, 0.2 # Should Adjust based on the performance
ffn_num_hiddens, num_heads = 64, 4
encoder = TransformerEncoder_NoEmbedding(
    num_hiddens, ffn_num_hiddens, num_heads,
    num_blks, dropout)
decoder = TransformerDecoder(
    len(signdata.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
    num_blks, dropout)
signmodel = d2l.Seq2Seq(encoder, decoder, tgt_pad=signdata.tgt_vocab['<pad>'],
                    lr=0.001)

#trainer = d2l.Trainer(max_epochs=20, gradient_clip_val=1, num_gpus=1)
# trainer = Trainer(max_epochs=1, gradient_clip_val=1, num_gpus=1) #Using self-Training which could store loss value

'''
Predict
'''
save_path = '/home/streetparking/SLR/savedModel/0904_256_6_02_128_4.pth'
signmodel.load_state_dict(torch.load(save_path, weights_only=True))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
signmodel.to(device)
for name, param in signmodel.named_parameters():
    print(f"Parameter: {name}, Device: {param.device}")
    break

# print('check device: ', device)

testing_features_dir = '/home/streetparking/SLR/NewPheonixSampleFeatures'
testing_translation_file = '/home/streetparking/SLR/germen_sentences.txt'

#testing_features_dir = '/home/streetparking/SLR/paddedTestingVideoFeaturesGPU'
#testing_translation_file = '/home/streetparking/SLR/testingTranslation.txt'

#testing_features_dir = '/home/streetparking/SLR/paddedTestingVideoFeaturesGPU'
#testing_translation_file = '/home/streetparking/SLR/testingTranslation.txt'

loaded_signs, loaded_gers = load_testing_features_and_sentences(testing_features_dir, testing_translation_file)
preds, _ = signmodel.predict_step(
    signdata.build(loaded_signs, loaded_gers), d2l.try_gpu(), signdata.num_steps)
for sign, ger, p in zip(loaded_signs, loaded_gers, preds):
    translation = []
    for token in signdata.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{sign} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), ger, k=2):.3f}')

#features, sentences = load_testing_features_and_sentences(testing_features_dir, testing_translation_file)
#print("loaded sentences: ", sentences[0])
# sign1 = features[0].to(device) # make sure the prediction feature are all in GPU
# sign2 = features[1].to(device)
# sign3 = features[2].to(device)
# sign4 = features[3].to(device)
# sign5 = features[4].to(device)

#print("sign 1 feature: ", sign1)
# signs = [sign1, sign2, sign3, sign4, sign5]
#signs = [features[0], features[1], features[2], features[3], features[4]]
# engs = ['liebe zuschauer guten abend', 'heftiger wintereinbruch gestern in nordirland schottland', 'schwere überschwemmungen in den usa', 'weiterhin warm am östlichen mittelmeer und auch richtung westliches mittelmeer ganz west und nordeuropa bleibt kühl', 'und sehr kühl wird auch die kommende nacht']
#engs = [sentences[0], sentences[1], sentences[2], sentences[3], sentences[4]]
# preds, _ = signmodel.predict_step(
#     # signdata.build(signs, engs), d2l.cpu(), signdata.num_steps)
#    signdata.build(signs, engs), d2l.try_gpu(), signdata.num_steps)
# for sign, eng, p in zip(signs, engs, preds):
#    translation = []
#    for token in signdata.tgt_vocab.to_tokens(p):
#        if token == '<eos>':
#            break
#        translation.append(token)
#    print(f'{sign} => {translation}, bleu,'
#          f'{d2l.bleu(" ".join(translation), eng, k=2):.3f}')

