__author__ = 'jwj'
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import myLSTM as LSTM
import pickle
import numpy as np
import torch.utils.data as Data
from data_process import process_data, get_data_from_condense_seq
from metrics import accuracy, sensitivity
from utils import *


# for validation loss in early stopping
def evaluate_loss(model, loader, vali_data, batchsize, dim_input_course, dim_input_grade, dim_input_major):

    model.eval()
    summ = []
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        processed_data = process_data(batch_x.numpy(), vali_data, batchsize, dim_input_course, dim_input_grade, dim_input_major)
        padded_input = Variable(torch.Tensor(processed_data[0]), requires_grad=False).cuda()
        seq_len = processed_data[1]
        padded_label = Variable(torch.Tensor(processed_data[2]), requires_grad=False).cuda()

        # clear hidden states
        model.hidden = model.init_hidden()
        model.hidden[0] = model.hidden[0].cuda()
        model.hidden[1] = model.hidden[1].cuda()
        # compute output
        y_pred = model(padded_input, seq_len).cuda()
        # only compute the loss for testing period
        loss = model.vali_loss(y_pred, seq_len, padded_label).cuda()
        summ.append(loss.item())

    average_loss = np.average(summ)
    return average_loss


# for validation
def evaluate_metrics(model, loader, vali_data, batchsize, dim_input_course, dim_input_grade, dim_input_major):

    model.eval()
    summ1 = 0  # letter grades
    summ2 = 0  # credit/uncredit

    len1 = len2 = 0
    tp = np.zeros(2)
    tn = np.zeros(2)
    true = np.zeros(2)
    false = np.zeros(2)
    predict_true = np.zeros(2)
    predict_false = np.zeros(2)
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        processed_data = process_data(batch_x.numpy(), vali_data, batchsize, dim_input_course, dim_input_grade, dim_input_major)
        padded_input = Variable(torch.Tensor(processed_data[0]), requires_grad=False).cuda()
        seq_len = processed_data[1]
        padded_label = Variable(torch.Tensor(processed_data[2]), requires_grad=False).cuda()

        # clear hidden states
        model.hidden = model.init_hidden()
        model.hidden[0] = model.hidden[0].cuda()
        model.hidden[1] = model.hidden[1].cuda()
        # compute output
        y_pred = model(padded_input, seq_len)

        # only compute the accuracy for testing period
        accura = accuracy(y_pred, seq_len, padded_label)
        len1 += accura[3]
        len2 += accura[4]
        summ1 += (accura[0] * accura[3])
        summ2 += (accura[1] * accura[4])

        print('accuracy on letter grades', accura[0], 'accuracy on credit/uncredit', accura[1], 'total', accura[2])

        # compute tp, fp, fn, tn
        sen = sensitivity(y_pred, seq_len, padded_label)
        tp += sen[0]
        tn += sen[1]
        true += sen[2]
        false += sen[3]
        predict_true += sen[4]
        predict_false += sen[5]

    average_metric1 = summ1 / len1
    average_metric2 = summ2 / len2
    average_metric = (summ1 + summ2) / (len1 + len2)

    print("number of letter grades ", len1, "num of credit/no credit: ", len2)
    print("Accuracy--", "letter grades:", average_metric1, "credit/no credit", average_metric2, "overall", average_metric)

    tpr = tp / true
    fpr = (predict_true - tp) / false
    fnr = (predict_false - tn) / true
    tnr = tn / false

    precision_B = (tn / predict_false)[0]
    f_value_B = 2 / (1 / tnr[0] + 1 / precision_B)
    precision_uncredit = (tn / predict_false)[-1]
    f_value_uncredit = 2 / (1 / tnr[-1] + 1 / precision_uncredit)
    f_value = np.append(f_value_B, f_value_uncredit)
    print("Further evaluation on prediction of letter grade and credit/no credit: ", "[on letter grade, on credit/no credit]")
    print("tpr: ", tpr)
    print("fpr: ", fpr)
    print("fnr: ", fnr)
    print("tnr: ", tnr)
    print('F: ', f_value, 'average F:', np.average(f_value))


if __name__ == '__main__':
    args = parse_args()
    batchsize = args.batch_size
    model_name = args.evaluated_model_path
    model = torch.load(model_name)

    evaluated_semester = args.evaluated_semester

    f = open(args.course_id_path, 'rb')
    course_id = pickle.load(f)['course_id']
    f = open(args.major_id_path, 'rb')
    major_id = pickle.load(f)['major_id']
    f.close()
    dim_input_course = len(course_id)
    dim_input_grade = len(course_id) * 4
    dim_input_major = len(major_id)

    vali_data = get_data_from_condense_seq(evaluated_semester)[1]
    vali_data_index = torch.IntTensor(np.array(range(vali_data.shape[0])))
    torch_vali_data_index = Data.TensorDataset(vali_data_index, vali_data_index)
    vali_loader = Data.DataLoader(dataset=torch_vali_data_index, batch_size=batchsize, shuffle=True, num_workers=2, drop_last=True)

    print('Evaluation results of ', model_name)
    evaluate_metrics(model, vali_loader, vali_data, batchsize, dim_input_course, dim_input_grade, dim_input_major)

