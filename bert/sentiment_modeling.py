from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from bert.modeling import BertModel, BERTLayerNorm
from bert.dynamic_rnn import DynamicLSTM
cuda=torch.device('cuda')

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()

def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):

    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)

    context_outputs = flatten_emb_by_sentence(input, input_mask)  # [<N*L, D]
    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]    # [N*M, JR, D]

    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):

    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def distant_cross_entropy(logits, positions, mask=None, threshold=None):

    sigmoid = nn.Sigmoid()
    positions = positions.to(dtype=logits.dtype)
    mask = mask.to(dtype=logits.dtype)
    probs = (1 - positions) + (2 * positions - 1) * sigmoid(logits)
    log_probs = torch.log(probs) * mask
    loss = -1 * torch.mean(torch.sum(log_probs, dim=-1) / torch.sum(mask, dim=-1))
    aspect_num = torch.sum((sigmoid(logits) > threshold ).to(dtype=logits.dtype) * mask, -1)
    return loss, aspect_num

def pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(0)
    return sequence

class MIM(nn.Module):
    def __init__(self, config,args):
        super(BertForJointSpanExtractAndClassification, self).__init__()
        self.bert = BertModel(config)
        self.ate = DynamicLSTM(config.hidden_size, args.hidden_size, num_layers=1, batch_first=True, bidirectional=True, rnn_type='GRU')
        self.atc= DynamicLSTM(config.hidden_size, args.hidden_size, num_layers=1, batch_first=True, bidirectional=True, rnn_type='GRU')
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.unary_affine = nn.Linear(2*args.hidden_size, 1)
        self.start_outputs = nn.Linear(2*args.hidden_size, 1)
        self.end_outputs = nn.Linear(4*args.hidden_size, 1)
        self.dense = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(2*args.hidden_size, 5)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
            elif isinstance(module, DynamicLSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        param.chunk(4)[1].fill_(1)
        self.apply(init_weights)

    def forward(self, mode, attention_mask, input_ids=None, token_type_ids=None, start_positions=None, end_positions=None, aspect_num=None,
                span_aspect_num=None, span_starts=None, span_ends=None, polarity_labels=None, label_masks=None, sequence_input=None,
                weight_kl=None, window_size = None,n_best_size = None, logit_threshold =None ):
        if mode == 'train':
            assert input_ids is not None and token_type_ids is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]
            batch_size, seq_len, _ = sequence_output.size()

            assert start_positions is not None and end_positions is not None and window_size is not None and n_best_size is not  None \
                   and logit_threshold  is not None
            temp = torch.ones(batch_size,dtype=torch.int64,device=cuda)
            L_tensor = seq_len*temp  #
            sequence_output_ate, (_, _) = self.ate(sequence_output, L_tensor )  # [batch_size,seq_len, d]
            sequence_output_atc, (_, _) = self.atc(sequence_output, L_tensor )  # [batch_size,seq_len,d]

            start_logits = self.start_outputs(sequence_output_ate)   # [batch_size, seq_len, 1]
            start_logits = start_logits.squeeze(-1)
            start_loss ,start_aspect_num = distant_cross_entropy(start_logits, start_positions,attention_mask,logit_threshold )

            mask = [[0] * seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    if i >= j and i - j < window_size:
                        mask[i][j] = 1
            mask = torch.tensor(mask, dtype=torch.float, device=cuda)
            mask_temp = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
            mask_temp = mask_temp.unsqueeze(-1)  # [batch_size, seq_len, seq_len,1]
            span_matrix = sequence_output_ate.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch_size ,seq_len ,seq_len,d]
            window_embedding = torch.mean(mask_temp * span_matrix, dim=-2)  # [batch_size ,seq_len, d]
            end_sentence = torch.cat([sequence_output_ate, window_embedding], dim=-1)  # [batch_size ,seq_len,2*d]

            end_logits = self.end_outputs(end_sentence)  # [batch_size, seq_len ,1]
            end_logits = end_logits.squeeze(-1)
            end_loss,end_aspect_num = distant_cross_entropy(end_logits, end_positions,attention_mask,logit_threshold )

            ae_loss = (start_loss + end_loss) / 2

            aspect_score = start_logits + end_logits
            aspect_score = aspect_score.unsqueeze(1).expand(-1, n_best_size, -1).reshape(n_best_size* batch_size, -1)

            assert span_starts is not None and span_ends is not None and polarity_labels is not None \
                   and label_masks is not None and  weight_kl is not None
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_output_atc,
                                                             attention_mask)  # [N*M, JR, d], [N*M, JR]

            sequence_output_atc = sequence_output_atc.unsqueeze(1).expand(-1, span_starts.size(1), -1, -1)
            sequence_output_atc = sequence_output_atc.reshape(span_output.size(0), seq_len, -1)  # [N*M, L,  d]
            interaction_mat = torch.matmul(sequence_output_atc, torch.transpose(span_output, 1, 2))  #[N*M, L, JR]
            alpha = torch.nn.functional.softmax(interaction_mat, dim=1)  #[N*M, L, JR]
            beta = torch.nn.functional.softmax(interaction_mat, dim=2) #[N*M, L, JR]
            beta_avg = beta.mean(dim=1, keepdim=True)   #[N*M, 1, JR]
            gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # [N*M, L, 1]
            span_pooled_output = torch.matmul(torch.transpose(sequence_output_atc, 1, 2), gamma).squeeze(-1)  # [N*M, d]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            ac_loss_fct = CrossEntropyLoss(reduction='none')
            flat_polarity_labels = flatten(polarity_labels)
            flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()

            loss_KL = torch.nn.KLDivLoss(reduction='mean')
            KL_1= loss_KL(gamma.squeeze(-1).softmax(dim=-1).log(), aspect_score.softmax(dim=-1))
            KL_2= loss_KL(aspect_score.softmax(dim=-1).log(), gamma.squeeze(-1).softmax(dim=-1))
            KL= -2 / (2 + KL_1 + KL_2)

            return ae_loss + ac_loss + weight_kl * KL

        elif mode == 'extract_inference':
            assert input_ids is not None and token_type_ids is not None and window_size is not None and logit_threshold is not None
            all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
            sequence_output = all_encoder_layers[-1]
            batch_size, seq_len, hid_size = sequence_output.size()

            temp = torch.ones(batch_size, dtype=torch.int64, device=cuda)
            L_tensor = seq_len * temp  #
            sequence_output_ate, (_, _) = self.ate(sequence_output, L_tensor)  # [batch_size,seq_len, d]
            sequence_output_atc, (_, _) = self.atc(sequence_output, L_tensor)  # [batch_size,seq_len,d]

            start_logits = self.start_outputs(sequence_output_ate)  # [N, L, 1]
            start_logits = start_logits.squeeze(-1)

            mask = [[0] * seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    if i >= j and i - j < window_size:
                        mask[i][j] = 1
            mask = torch.tensor(mask, dtype=torch.float, device=cuda)
            mask_temp = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]
            mask_temp = mask_temp.unsqueeze(-1)  # [batch_size, seq_len, seq_len,1]
            span_matrix = sequence_output_ate.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch_size ,seq_len ,seq_len,d]
            window_embedding = torch.mean(mask_temp * span_matrix, dim=-2)  # [batch_size ,seq_len, d]
            end_sentence = torch.cat([sequence_output_ate, window_embedding], dim=-1)  # [batch_size ,seq_len,2*d]
            end_logits = self.end_outputs(end_sentence)  # [batch_size, seq_len ,1]
            end_logits = end_logits.squeeze(-1)

            sigmoid = torch.nn.Sigmoid()
            start_logits = sigmoid(start_logits)
            end_logits = sigmoid(end_logits)

            start_target_num = torch.sum((sigmoid(start_logits) > logit_threshold).to(dtype=start_logits.dtype) * attention_mask.to(
                dtype=start_logits.dtype), -1)
            end_target_num = torch.sum(
                (sigmoid(end_logits) > logit_threshold).to(dtype=end_logits.dtype) * attention_mask.to(dtype=end_logits.dtype), -1)
            target_num_prediction = (start_target_num + end_target_num) / 2
            return start_logits, end_logits, target_num_prediction,sequence_output_atc

        elif mode == 'classify_inference':
            assert span_starts is not None and span_ends is not None and sequence_input is not None
            batch_size, seq_len, hid_size = sequence_input .size()
            span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                             attention_mask)  # [N*M, JR, d], [N*M, JR]

            sequence_output_atc = sequence_input.unsqueeze(1).expand(-1, span_starts.size(1), -1, -1)
            sequence_output_atc = sequence_output_atc.reshape(-1, seq_len, span_output.size(2))  # [N*M, L,  d]
            interaction_mat = torch.matmul(sequence_output_atc, torch.transpose(span_output, 1, 2))  # [N*M, L, JR]
            alpha = torch.nn.functional.softmax(interaction_mat, dim=1)  # [N*M, L, JR]
            beta = torch.nn.functional.softmax(interaction_mat, dim=2) #[N*M, L, JR]
            beta_avg = beta.mean(dim=1, keepdim=True)   #[N*M, 1, JR]
            gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # [N*M, L, 1]
            span_pooled_output = torch.matmul(torch.transpose(sequence_output_atc, 1, 2), gamma).squeeze(-1)  # [N*M, d]

            span_pooled_output = self.dense(span_pooled_output)
            span_pooled_output = self.activation(span_pooled_output)
            span_pooled_output = self.dropout(span_pooled_output)
            ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

            return reconstruct(ac_logits, span_starts)


