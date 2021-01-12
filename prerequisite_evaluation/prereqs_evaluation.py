__author__ = 'jwj'
from utils import *
import sys
sys.path.insert(0, '../grade_prediction')
import myLSTM as LSTM
import torch
import pickle
import pandas as pd
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
import csv


# condense all input data: courses_id concatenated with courses with grade == A and B
def get_input_gradedata(k):  # k: id of output course

    input_data = np.zeros((len(course_id), 1, dim_input), int)
    for i in range(len(course_id)):  # >B
        input_data[i, 0, k] = 1
        input_data[i, 0, len(course_id) + 4 * i] = 1
    return input_data


def cal_output(model, loader, input_data, batchsize):

    model.eval()
    y_pred_all = torch.FloatTensor(np.zeros((len(loader) * batchsize, 1, len(course_id) * 4)))
    len_padded = 0
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        padded_input = input_data[batch_x]
        if len(batch_x) < batchsize:
            len_padded = batchsize - len(batch_x)
            padded_one = np.ones((len_padded, 1, dim_input), int)
            padded_input = np.concatenate((padded_input, padded_one), axis=0)
        seq_len = [1 for i in range(batchsize)]
        padded_input = Variable(torch.Tensor(padded_input), requires_grad=False)
        # clear hidden states
        model.hidden = model.init_hidden()
        # compute output (batch_size, 1, dim_output)
        y_pred = model(padded_input, seq_len)
        # print(y_pred)
        y_pred_all[range(step * batchsize, (step + 1) * batchsize)] = y_pred.data
    if len_padded > 0:
        y_pred_all = y_pred_all[:-len_padded]
    y_pred_temp = y_pred_all.view(y_pred_all.size()[0], 1, len(course_id), -1)
    y_pred_temp1 = torch.exp(y_pred_temp[:, :, :, :2])
    y_pred_all = torch.exp(y_pred_temp) / torch.sum(y_pred_temp1, dim=3)[:, :, :, None]
    y_pred_all = y_pred_all.contiguous().view(-1, 1, len(course_id) * 4)
    return y_pred_all


def level(num):
    if num <= 99:
        level = 0
    elif num <= 199:
        level = 1
    else:
        level = 2
    return level


def evaluate_prereqs(i, N):  # i is the target course id
    prob = np.zeros(len(course_id))
    input_data = get_input_gradedata(i)
    input_data_index = torch.IntTensor(np.array(range(len(input_data))))
    input_torch_data_index = Data.TensorDataset(input_data_index, input_data_index)
    input_loader = Data.DataLoader(dataset=input_torch_data_index, batch_size=batchsize, shuffle=False, num_workers=2, drop_last=False)
    output = cal_output(model, input_loader, input_data, batchsize)
    t_course = id_course[i]
    t_num = t_course.split(' ')[1]
    t_num = int(''.join(list(filter(str.isdigit, t_num))))
    t_level = level(t_num)
    #print(t_course, t_level)

    for j in range(len(course_id)):
        c_num = id_course[j].split(' ')[1]
        c_num = int(''.join(list(filter(str.isdigit, c_num))))
        c_level = level(c_num)
        if c_level <= t_level:
            prob[j] = output[j, 0, i * 4]
    prob[i] = 0
    if use_filter:
        rank = np.argsort(-prob)
        related_sub = target_prereqs_sub_filter[i]
        related_course = [j for j in range(len(course_id)) if course_id_sub_id[j] in related_sub and prob[j]!=0]
        related_course_ranked = list(set(rank).intersection(set(related_course)))
        related_course_ranked.sort(key=list(rank).index)
        unrelated_course_ranked = list(set(rank)-set(related_course_ranked))
        unrelated_course_ranked.sort(key=list(rank).index)
        rank = related_course_ranked + unrelated_course_ranked
        rank = rank[:N]
        print('Top N predictions:')
        for i in rank:
            print(id_course[int(i)])

    else:
            rank = np.argsort(-prob)[:N]

    for k in data_prereqs_list:
        if k[1] in rank:
            print('correct predictions: ', id_course[k[0]], id_course[k[1]])
            with open(file_name, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([id_course[k[0]], id_course[k[1]]])


if __name__ == '__main__':

    args = parse_arguments()
    #target_id = sys.argv[1]  # input target course ID
    file_name = args.results_path + str(args.target_course_id)+'.tsv'
    use_filter =True
    target_prereqs_sub_filter = pickle.load(open('target_prereqs_filter.pkl', 'rb'))
    course_id_sub_id = pickle.load(open('course_id_sub_id.pkl', 'rb'))

    f = open('target_id.pkl', 'rb')
    course_id_target = int(pickle.load(f)['id_target'][int(args.target_course_id)])
    
    batchsize = 32

    model_name = args.evaluated_model_path
    # use cpu to load the model
    model = torch.load(model_name, map_location=lambda storage, loc: storage)

    f = open(args.course_id_path, 'rb')
    course = pickle.load(f)
    course_id = course['course_id']
    id_course = course['id_course']
    f = open(args.major_id_path,'rb')
    major = pickle.load(f)
    major_id = major['major_id']
    dim_input = len(course_id) + len(course_id) * 4 + len(major_id)  # input: target course + course_grade
    f.close()

    data_prereqs = pd.read_csv(args.prereqs_path, header=0)
    data_prereqs['prereqs'] = data_prereqs['prereqs'].map(course_id)
    data_prereqs['target'] = data_prereqs['target'].map(course_id)
    data_prereqs.dropna(inplace=True)

    data_prereqs = data_prereqs.loc[data_prereqs['target'] == course_id_target]
    print(data_prereqs)

    data_prereqs_list = []
    for i, j in zip(data_prereqs['target'], data_prereqs['prereqs']):
        data_prereqs_list.append((int(i), int(j)))

    evaluate_prereqs(course_id_target, args.topN)


