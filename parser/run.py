import os
import torch
import nltk
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from torch.autograd import Variable


from .tag import Tag
from .mask import OuterMask, RelationMask, VariableMask
from .utils import readpretrain, get_from_ix, getRequiredOutput
from .encoder import EncoderRNN
from .decoder import AttnDecoderRNN

class Parser(object):
    def __init__(self):
        self.word_to_ix = {}
        self.lemma_to_ix = {}
        self.ix_to_lemma = []
        self.ix_to_word = []
    
    def load_model(self):
        pretrain_file = os.path.join('data', 'sskip.100.vectors')
        tag_info_file = os.path.join('data', 'tag.info')
        word_list_file = os.path.join('data', 'word.list')
        lemma_list_file = os.path.join('data', 'lemma.list')
        model_file = os.path.join('data', 'model')
        UNK = "<UNK>"

        # Get the bag of words
        for line in open(word_list_file):
            line = line.strip()
            self.word_to_ix[line] = len(self.ix_to_word)
            self.ix_to_word.append(line)
        
        for line in open(lemma_list_file):
            line = line.strip()
            self.lemma_to_ix[line] = len(self.ix_to_lemma)
            self.ix_to_lemma.append(line)
        
        self.tags_info = Tag(tag_info_file, self.ix_to_lemma)
        SOS = self.tags_info.SOS
        EOS = self.tags_info.EOS
        self.outer_mask_pool = OuterMask(self.tags_info)
        self.rel_mask_pool = RelationMask(self.tags_info)
        self.var_mask_pool = VariableMask(self.tags_info)
        
        
        self.pretrain_to_ix = {UNK:0}
        self.pretrain_embeddings = [ [0. for i in range(100)] ] # for UNK 
        pretrain_data = readpretrain(pretrain_file)
       
        for one in pretrain_data:
            self.pretrain_to_ix[one[0]] = len(self.pretrain_to_ix)
            self.pretrain_embeddings.append([float(a) for a in one[1:]])
        
        
        print("pretrain dict size:{}".format(len(self.pretrain_to_ix)))
        print("word dict size: {}".format(len(self.word_to_ix)))
        print("lemma dict size: {}".format(len(self.lemma_to_ix)))
        print("global tag (w/o variables) dict size: {}".format(self.tags_info.k_rel_start))
        print("global tag (w variables) dict size: {}".format(self.tags_info.tag_size))

        WORD_EMBEDDING_DIM = 64
        PRETRAIN_EMBEDDING_DIM = 100
        LEMMA_EMBEDDING_DIM = 32
        TAG_DIM = 128
        INPUT_DIM = 100
        ENCODER_HIDDEN_DIM = 256
        DECODER_INPUT_DIM = 128
        ATTENTION_HIDDEN_DIM = 256

        self.encoder = EncoderRNN(len(self.word_to_ix), WORD_EMBEDDING_DIM, len(self.pretrain_to_ix), PRETRAIN_EMBEDDING_DIM, torch.FloatTensor(self.pretrain_embeddings), len(self.lemma_to_ix), LEMMA_EMBEDDING_DIM, INPUT_DIM, ENCODER_HIDDEN_DIM, n_layers=2, dropout_p=0.1)
        self.decoder = AttnDecoderRNN(self.outer_mask_pool, self.rel_mask_pool, self.var_mask_pool, self.tags_info, TAG_DIM, DECODER_INPUT_DIM, ENCODER_HIDDEN_DIM, ATTENTION_HIDDEN_DIM, n_layers=1, dropout_p=0.1)
    
        check_point = torch.load(model_file, map_location='cpu')
        self.encoder.load_state_dict(check_point["encoder"])
        self.decoder.load_state_dict(check_point["decoder"])
        return

    def _get_lemmas(self, tokens):
        lemmatizer = WordNetLemmatizer()
        pos_tokens = [nltk.pos_tag(tokens)]
        lemmas = []
        for pos in pos_tokens[0]:
            word, pos_tag = pos
            lemmas.append(lemmatizer.lemmatize(word.lower(),self._get_wordnet_pos(pos_tag)))
            lemmas[-1] = lemmas[-1].encode("utf8")
        return lemmas

    def _preprocess(self,tokens):
        for i in range(len(tokens)):
            if tokens[i] == "(":
                tokens[i] = "-LRB-"
            elif tokens[i] == ")":
                tokens[i] = "-RRB-"
        return tokens 

    def _get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _decode(self, sentence_variable):
        encoder = self.encoder
        decoder = self.decoder
        encoder_hidden = encoder.initHidden()
        encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden, train=False)
        
        ####### struct
        decoder_hidden1 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
        with torch.no_grad():
            decoder_input1 = Variable(torch.LongTensor([0]))
        decoder_output1, hidden_rep1, decoder_hidden1 = decoder(None, decoder_input1, decoder_hidden1, encoder_output, least=None, train=False, mask_variable=None, opt=1)
        structs = decoder_output1.view(-1).data.tolist()

        ####### relation
        decoder_hidden2 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
        decoder.rel_mask_pool.reset(sentence_variable[0].size(0))
        decoder.total_rel = 0
        relations = []
        positions = []
        hidden_rep2_list = []
        for i in range(len(structs)):
            if structs[i] == 5 or structs[i] == 6: # prev output, and hidden_rep1[i+1] is the input representation of prev output.
                least = False
                if structs[i] == 5 or (structs[i] == 6 and structs[i+1] == 4):
                    least = True
                decoder.rel_mask_pool.set_sdrs(structs[i] == 5)
                decoder_output2, hidden_rep2, decoder_hidden2, position = decoder(sentence_variable, hidden_rep1[i+1], decoder_hidden2, encoder_output, least=least, train=False, mask_variable=None, opt=2)
                relations.append(decoder_output2.view(-1).data.tolist())
                positions.append(position)
                hidden_rep2_list.append(hidden_rep2)
        ####### variable
        decoder_hidden3 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
        #p_max
        p_max = 0
        for tok in structs:
            if tok >= decoder.tags_info.p_rel_start and tok < decoder.tags_info.k_tag_start:
                p_max += 1
        #user_k
        user_k = []
        stack = []
        for tok in structs:
            if tok == 4:
                if stack[-1][0] == 5:
                    user_k.append(stack[-1][1])
                stack.pop()
            else:
                if tok >= decoder.tags_info.k_rel_start and tok < decoder.tags_info.p_rel_start:
                    stack[-1][1].append(tok - decoder.tags_info.k_rel_start)
                stack.append([tok,[]])
        decoder.var_mask_pool.reset(p_max, k_use=True)
        structs_p = 0
        user_k_p = 0
        struct_rel_tokens = []
        var_tokens = []
        for i in range(len(structs)):
            if structs[i] == 1: # EOS
                continue
            decoder.var_mask_pool.update(structs[i])
            struct_rel_tokens.append((structs[i],-1))
            if structs[i] == 5 or structs[i] == 6:
                if structs[i] == 5:
                    assert len(user_k[user_k_p]) >= 2
                    decoder.var_mask_pool.set_k(user_k[user_k_p])
                    user_k_p += 1

                for j in range(len(relations[structs_p])):
                    if relations[structs_p][j] == 1: # EOS
                        continue
                    decoder.var_mask_pool.update(relations[structs_p][j])
                    struct_rel_tokens.append((relations[structs_p][j], positions[structs_p][j]))
                    decoder_output3, decoder_hidden3= decoder(None, hidden_rep2_list[structs_p][j+1], decoder_hidden3, encoder_output, least=None, train=False, mask_variable=None, opt=3)
                    var_tokens.append(decoder_output3.view(-1).data.tolist())
                    decoder.var_mask_pool.update(4)
                    struct_rel_tokens.append((4, -1))
                structs_p += 1
        assert structs_p == len(relations)

        return struct_rel_tokens, var_tokens

    def test(self, sent):
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        tokens = tokenizer.tokenize(sent)
        tokens = self._preprocess(tokens)
        lemmas = self._get_lemmas(tokens)
        pretrains = [str(tok.lower()) for tok in tokens]
        print(tokens)
        print(lemmas)
        print(pretrains)

        instance = []
        with torch.no_grad():
            instance.append(Variable(torch.LongTensor([get_from_ix(tok, self.word_to_ix, 0) for tok in tokens])))
            instance.append(Variable(torch.LongTensor([get_from_ix(tok, self.pretrain_to_ix, 0) for tok in pretrains])))
            instance.append(Variable(torch.LongTensor([get_from_ix(tok.decode('ascii'), self.lemma_to_ix, 0) for tok in lemmas])))
        print(instance)
        structs, tokens = self._decode(instance)
        print(structs)
        print(tokens)
        output = getRequiredOutput(structs, tokens, sent)
        return output


if __name__ == '__main__':
    p = Parser()
    p.load_model()
    out = p.test('Hello how are you.')
