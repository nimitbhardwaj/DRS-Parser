import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

class AttnDecoderRNN(nn.Module):
    def __init__(self, outer_mask_pool, rel_mask_pool, var_mask_pool, tags_info, tag_dim, input_dim, feat_dim, encoder_hidden_dim, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.outer_mask_pool = outer_mask_pool
        self.rel_mask_pool = rel_mask_pool
        self.var_mask_pool = var_mask_pool
        self.total_rel = 0

        self.tags_info = tags_info
        self.tag_size = tags_info.tag_size
        self.all_tag_size = tags_info.all_tag_size

        self.tag_dim = tag_dim
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.hidden_dim = encoder_hidden_dim * 2

        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        self.tag_embeds = nn.Embedding(self.tags_info.all_tag_size, self.tag_dim)

        self.struct2rel = nn.Linear(self.hidden_dim, self.tag_dim)
        self.rel2var = nn.Linear(self.hidden_dim, self.tag_dim)

        self.lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)

        self.feat = nn.Linear(self.hidden_dim + self.tag_dim, self.feat_dim)
        self.feat_tanh = nn.Tanh()

        self.out = nn.Linear(self.feat_dim, self.tag_size)

        self.selective_matrix = Variable(torch.randn(1, self.hidden_dim, self.hidden_dim))

    def forward(self, sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable, opt):
        if opt == 1:
            return self.forward_1(inputs, hidden, encoder_output, train, mask_variable)
        elif opt == 2:
            return self.forward_2(sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable)
        elif opt == 3:
            return self.forward_3(inputs, hidden, encoder_output, train, mask_variable)
        else:
            assert False, "unrecognized option"

    def forward_1(self, input, hidden, encoder_output, train, mask_variable):
        self.lstm.dropout = 0.0
        tokens = []
        self.outer_mask_pool.reset()
        hidden_rep = []
        while True:
            mask = self.outer_mask_pool.get_step_mask()
            with torch.no_grad():
                mask_variable = Variable(torch.FloatTensor(mask), requires_grad = False).unsqueeze(0)
            
            embedded = self.tag_embeds(input).view(1, 1, -1)
            output, hidden = self.lstm(embedded, hidden)
            
            hidden_rep.append(output)

            attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

            global_score = self.out(feat_hiddens)

            score = global_score + (mask_variable - 1) * 1e10

            _, input = torch.max(score,1)
            idx = input.view(-1).data.tolist()[0]

            tokens.append(idx)
            self.outer_mask_pool.update(-2, idx)

            if idx == self.tags_info.tag_to_ix[self.tags_info.EOS]:
                break
        with torch.no_grad():
            x = Variable(torch.LongTensor(tokens)), torch.cat(hidden_rep,0), hidden
        return x

    def forward_2(self, sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable):
        self.lstm.dropout = 0.0
        tokens = []
        positions = []
        rel = 0
        hidden_reps = []

        mask_variable_true = Variable(torch.FloatTensor(self.rel_mask_pool.get_step_mask(True)), requires_grad = False)
        mask_variable_false = Variable(torch.FloatTensor(self.rel_mask_pool.get_step_mask(False)), requires_grad = False)
        embedded = self.struct2rel(inputs).view(1, 1,-1)

        while True:
            output, hidden = self.lstm(embedded, hidden)
            hidden_reps.append(output)
            selective_score = torch.bmm(torch.bmm(output, self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)
            attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

            global_score = self.out(feat_hiddens)

            total_score = torch.cat((global_score, selective_score), 1)

            if least:
                output = total_score + (mask_variable_true - 1) * 1e10
                least = False
            else:
                output = total_score + (mask_variable_false - 1) * 1e10

            _, input = torch.max(output,1)
            idx = input.view(-1).data.tolist()[0]

            if idx >= self.tags_info.tag_size:
                ttype = idx - self.tags_info.tag_size
                idx = sentence_variable[2][ttype].view(-1).data.tolist()[0]
                idx += self.tags_info.tag_size
                tokens.append(idx)
                positions.append(ttype)
                with torch.no_grad():
                    input = Variable(torch.LongTensor([idx]))
            else:
                tokens.append(idx)
            
            positions.append(-1)

            if idx == self.tags_info.tag_to_ix[self.tags_info.EOS]:
                break
            elif rel > 61 or self.total_rel > 121:
                embedded = self.tag_embeds(input).view(1, 1, -1)
                output, hidden = self.lstm(embedded, hidden)
                hidden_reps.append(output)
                break
            rel += 1
            self.total_rel += 1
            embedded = self.tag_embeds(input).view(1, 1, -1)
        with torch.no_grad():
            x = Variable(torch.LongTensor(tokens)), torch.cat(hidden_reps,0), hidden, positions
        return x

    def forward_3(self, inputs, hidden, encoder_output, train, mask_variable):
        self.lstm.dropout = 0.0
        tokens = []
        embedded = self.rel2var(inputs).view(1, 1,-1)
        while True:
            output, hidden = self.lstm(embedded, hidden)

            attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

            global_score = self.out(feat_hiddens)

            mask = self.var_mask_pool.get_step_mask()
            with torch.no_grad():
                mask_variable = Variable(torch.FloatTensor(mask))

            score = global_score + (mask_variable - 1) * 1e10

            _, input = torch.max(score, 1)
            embedded = self.tag_embeds(input).view(1, 1, -1)

            idx = input.view(-1).data.tolist()[0]
            assert idx < self.tags_info.tag_size
            if idx == self.tags_info.tag_to_ix[self.tags_info.EOS]:
                break
            tokens.append(idx)
            self.var_mask_pool.update(idx)  
        with torch.no_grad():
            x = Variable(torch.LongTensor(tokens)), hidden
        return x