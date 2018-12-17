__author__ = 'jwj'
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class LSTM_cat_cat(nn.Module):

    def __init__(self, type, dim_input_course, dim_input_grade, dim_input_major, nb_lstm_layers, nb_lstm_units, batch_size, drp_rate):
        super(LSTM_cat_cat, self).__init__()
        self.type = type
        self.dim_input_grade_major = dim_input_grade + dim_input_major
        self.dim_input_course= dim_input_course 
        self.dim_input_major = dim_input_major
        self.dim_output = dim_input_grade
        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.batch_size = batch_size
        if self.type == 'LSTM':
            self.lstm = nn.LSTM(input_size=self.dim_input_grade_major, hidden_size=nb_lstm_units, num_layers=nb_lstm_layers, batch_first=True)
        elif self.type == 'RNN':
            self.lstm = nn.RNN(input_size=self.dim_input_grade_major, hidden_size=nb_lstm_units, num_layers=nb_lstm_layers, batch_first=True)
        elif self.type == 'GRU':
            self.lstm = nn.GRU(input_size=self.dim_input_grade_major, hidden_size=nb_lstm_units, num_layers=nb_lstm_layers, batch_first=True)
        self.hidden = self.init_hidden()
        if drp_rate > 0:
            self.drp = True
            self.dropout = nn.Dropout(p=drp_rate)
        else:
            self.drp = False
        self.course_to_dense = nn.Linear(dim_input_course, nb_lstm_units)
        self.hidden_to_output = nn.Linear(nb_lstm_units * 2, self.dim_output)

    def init_hidden(self):
        if self.type == 'LSTM':
            hidden_a = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
            hidden_b = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
            hidden_layer = [Variable(hidden_a), Variable(hidden_b)]
        elif self.type == 'RNN' or 'GRU':
            hidden_layer = Variable(torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units))
        return hidden_layer

    def forward(self, batch_data, batch_data_length):
        seq_len = batch_data.size()[1]

        batch_grade_major = batch_data[:, :, self.dim_input_course:].contiguous()

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        pack_input = torch.nn.utils.rnn.pack_padded_sequence(batch_grade_major, batch_data_length, batch_first=True)
        lstm_out, self.hidden = self.lstm(pack_input, self.hidden)

        # undo the packing operation
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # reshape the data so it goes into the linear layer
        lstm_out = lstm_out.contiguous()
        lstm_out = lstm_out.view(-1, lstm_out.shape[2])

        batch_course = batch_data[:, :, :self.dim_input_course].contiguous()
        dense_out = self.course_to_dense(batch_course)
        dense_out = dense_out.view(-1, dense_out.shape[2])

        linear_in = torch.cat((lstm_out, dense_out), dim=1)

        # run through linear layer
        # if dropout
        if self.drp:
            dropout = self.dropout(linear_in)
            output = self.hidden_to_output(dropout)
        else:
            output = self.hidden_to_output(linear_in)

        output = output.view(self.batch_size, seq_len, self.dim_output)
        return output

    # masked loss, softmax on course
    def loss(self, output, label, weight1, weight2):
        # first mask the padded items in the output vector
        trans_label = torch.sum(label, dim=2)
        trans_label = trans_label.view(-1)
        real_label = torch.nonzero(trans_label)
        real_label = real_label.view(-1)
        trans_output = output.view(-1, self.dim_output)
        mask_output = trans_output[real_label]

        # second mask the unenrolled courses in output vector
        trans_label1 = label.view(-1, self.dim_output)
        trans_label1 = trans_label1[real_label]
        trans_label1 = trans_label1.view(-1, 4)
        real_label1 = torch.sum(trans_label1, dim=1)
        real_label1 = torch.nonzero(real_label1)
        real_label1 = real_label1.view(-1)
        mask_output = mask_output.view(-1, 4)
        mask_output = mask_output[real_label1]
        mask_label = trans_label1[real_label1]

        # identify whether ABCDF or Credit/No credit
        mask_label1 = mask_label[:, :2]
        mask_output1 = mask_output[:, :2]
        l1 = torch.sum(mask_label1, dim=1)
        l1 = torch.nonzero(l1).view(-1)

        mask_output1 = mask_output1[l1]
        mask_label1 = torch.nonzero(mask_label1)[:, 1].contiguous().view(-1)
        cross_entropy1 = nn.CrossEntropyLoss(weight=0.5/weight1)
        loss1 = cross_entropy1(mask_output1, mask_label1)

        mask_label2 = mask_label[:, 2:]
        mask_output2 = mask_output[:, 2:]
        l2 = torch.sum(mask_label2, dim=1)
        l2 = torch.nonzero(l2).view(-1)
        mask_output2 = mask_output2[l2]
        mask_label2 = torch.nonzero(mask_label2)[:, 1].contiguous().view(-1)
        cross_entropy2 = nn.CrossEntropyLoss(weight=0.5/weight2)
        loss2 = cross_entropy2(mask_output2, mask_label2)

        return loss1+loss2

        # masked loss, softmax on course
    def weighted_loss_by_minibatch(self, output, label):

        # first mask the padded items in the output vector
        trans_label = torch.sum(label, dim=2)
        trans_label = trans_label.view(-1)
        real_label = torch.nonzero(trans_label)
        real_label = real_label.view(-1)
        trans_output = output.view(-1, self.dim_output)
        mask_output = trans_output[real_label]

        # second mask the unenrolled courses in output vector
        trans_label1 = label.view(-1, self.dim_output)
        trans_label1 = trans_label1[real_label]
        trans_label1 = trans_label1.view(-1, 7)
        real_label1 = torch.sum(trans_label1, dim=1)
        real_label1 = torch.nonzero(real_label1)
        real_label1 = real_label1.view(-1)
        mask_output = mask_output.view(-1, 7)
        mask_output = mask_output[real_label1]
        mask_label = trans_label1[real_label1]

        # identify whether ABCDF or Credit/No credit
        mask_label1 = mask_label[:, :5]
        mask_output1 = mask_output[:, :5]
        l1 = torch.sum(mask_label1, dim=1)
        l1 = torch.nonzero(l1).view(-1)
        # calculate weight for ABCDF classes
        count = torch.sum(mask_label1[l1], dim=0)
        count = count / torch.sum(count)
        count = 0.2 / count
        print(count)

        mask_output1 = mask_output1[l1]
        mask_label1 = torch.nonzero(mask_label1)[:, 1].contiguous().view(-1)
        cross_entropy1 = nn.CrossEntropyLoss(weight=count)
        loss1 = cross_entropy1(mask_output1, mask_label1)

        mask_label2 = mask_label[:, 5:]
        mask_output2 = mask_output[:, 5:]
        l2 = torch.sum(mask_label2, dim=1)
        l2 = torch.nonzero(l2).view(-1)
        # calculate weight for credit/uncredit classes
        count = torch.sum(mask_label2[l2], dim=0)
        count = count / torch.sum(count)
        count = 0.5 / count
        print(count)

        mask_output2 = mask_output2[l2]
        mask_label2 = torch.nonzero(mask_label2)[:, 1].contiguous().view(-1)
        cross_entropy2 = nn.CrossEntropyLoss(weight=count)
        loss2 = cross_entropy2(mask_output2, mask_label2)

        return loss1+loss2

    # only calculate the loss in the last semester of each sequence
    def vali_loss(self, output, out_len, label, weight1, weight2):

        # first get the last semester of each sequence
        output_last = output[range(output.shape[0]), out_len - 1]
        label_last = label[range(label.shape[0]), out_len - 1]

        # second mask the unenrolled courses in output vector
        tran_label = label_last.contiguous().view(-1, 4)
        tran_label1 = torch.sum(tran_label, dim=1)
        masked_label = torch.nonzero(tran_label1).view(-1)
        real_label = tran_label[masked_label]

        tran_output = output_last.contiguous().view(-1, 4)
        real_output = tran_output[masked_label]

        # identify whether ABCDF or Credit/No credit
        real_label1 = real_label[:, :2]
        real_output1 = real_output[:, :2]
        l1 = torch.sum(real_label1, dim=1)
        l1 = torch.nonzero(l1).view(-1)

        real_output1 = real_output1[l1]
        real_label1 = np.nonzero(real_label1)[:, 1].contiguous().view(-1)
        cross_entropy1 = nn.CrossEntropyLoss(weight=0.5/weight1)
        loss1 = cross_entropy1(real_output1, real_label1)

        real_label2 = real_label[:, 2:]
        real_output2 = real_output[:, 2:]
        l2 = torch.sum(real_label2, dim=1)
        l2 = torch.nonzero(l2).view(-1)

        real_output2 = real_output2[l2]
        real_label2 = np.nonzero(real_label2)[:, 1].contiguous().view(-1)
        cross_entropy2 = nn.CrossEntropyLoss(weight=0.5/weight2)
        loss2 = cross_entropy2(real_output2, real_label2)

        return loss1+loss2

 # only calculate the loss in the last semester of each sequence
    def vali_weighted_loss_by_minibatch(self, output, out_len, label):

        # first get the last semester of each sequence
        output_last = output[range(output.shape[0]), out_len - 1]
        label_last = label[range(label.shape[0]), out_len - 1]

        # second mask the unenrolled courses in output vector
        tran_label = label_last.contiguous().view(-1, 7)
        tran_label1 = torch.sum(tran_label, dim=1)
        masked_label = torch.nonzero(tran_label1).view(-1)
        real_label = tran_label[masked_label]

        tran_output = output_last.contiguous().view(-1, 7)
        real_output = tran_output[masked_label]

        # identify whether ABCDF or Credit/No credit
        real_label1 = real_label[:, :5]
        real_output1 = real_output[:, :5]
        l1 = torch.sum(real_label1, dim=1)
        l1 = torch.nonzero(l1).view(-1)
        # calculate weight for ABCDF classes
        count = torch.sum(real_label1[l1], dim=0)
        count = count / torch.sum(count)
        count = 0.2 / count
        print(count)

        real_output1 = real_output1[l1]
        real_label1 = np.nonzero(real_label1)[:, 1].contiguous().view(-1)
        cross_entropy1 = nn.CrossEntropyLoss(weight=count)
        loss1 = cross_entropy1(real_output1, real_label1)

        real_label2 = real_label[:, 5:]
        real_output2 = real_output[:, 5:]
        l2 = torch.sum(real_label2, dim=1)
        l2 = torch.nonzero(l2).view(-1)
        # calculate weight for credit/uncredit classes
        count = torch.sum(real_label2[l2], dim=0)
        count = count / torch.sum(count)
        count = 0.5 / count
        print(count)

        real_output2 = real_output2[l2]
        real_label2 = np.nonzero(real_label2)[:, 1].contiguous().view(-1)
        cross_entropy2 = nn.CrossEntropyLoss(weight=count)
        loss2 = cross_entropy2(real_output2, real_label2)

        return loss1+loss2



