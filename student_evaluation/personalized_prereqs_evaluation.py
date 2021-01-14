__author__ = 'jwj'
import pandas as pd
import numpy as np
import pickle
from utils import *
import sys
sys.path.insert(0, '../grade_prediction')
import myLSTM
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import csv


# select students with enrollment records in Spring 2020 (validation period) and at least one period before Spring 2020
def select_evaluation_set(course_id, grade_cutoff):
    if grade_cutoff == 'A':
        good_grades = [1]  # A
        bad_grades = [2, 3, 4, 5]  # BCDF
    elif grade_cutoff == 'B':
        good_grades = [1, 2]  # AB
        bad_grades = [3, 4, 5]  # CDF
    else:
        print('Please set grade cutoff to A or B!')
        exit()
    f = open(args.input_path, 'rb')
    data = pickle.load(f)['stu_sem_major_grade_condense']
    data = np.array(data)
    vali_period = data[:, args.evaluated_semester]  # goal course in the evaluated semester
    stu_0 = []
    stu_1 = []
    for i in range(len(data)):
        if vali_period[i] != {} and data[i, args.evaluated_semester-1] == {}:  # have courses in spring 2016
            stu_sem_course = np.array(vali_period[i]['course_grade'])[:, 0]  # get all the courses he selected in spring 2016
            if course_id in stu_sem_course:  # if target course is in
                # check whether have at least two semesters before spring 2016
                num = len([1 for j in data[i, :args.evaluated_semester] if j!= {}])
                if num >= 2:
                    where = np.where(stu_sem_course == course_id)[0]  # get the grade
                    grade = np.array(vali_period[i]['course_grade'])[where, 1]
                    if grade in good_grades:
                        stu_0.append(i)
                    elif grade in bad_grades:
                        stu_1.append(i)
    print('number of well-performing students: ', len(data[stu_0]), '; number of under-performing students: ', len(data[stu_1]))
    if len(data[stu_0])==0 or len(data[stu_1])==0:
        print('Not enough students to evaluate, please choose another course.')
        exit()
    return data[stu_0, :args.evaluated_semester+1], data[stu_1, :args.evaluated_semester+1]


# condense all input data: courses_id concatenated with courses with grade == A and B
def get_input_gradedata(k, candidates):  # k: id of output course

    input_data = np.zeros((len(candidates), 1, dim_input_course+dim_input_grade+dim_input_major), int)
    s = 0
    dic_inputid_courseid = dict()
    for i in candidates:
        input_data[s, 0, k] = 1
        input_data[s, 0, len(course_id) + 4 * i] = 1
        dic_inputid_courseid[s] = i
        s += 1
    return input_data, dic_inputid_courseid


def process_data(data, dim_input_course, dim_input_grade, dim_input_major):
    length = len([2 for i in data if i != {}])
    if args.grade_cutoff == 'A':
        cutoff = 1
    elif args.grade_cutoff == 'B':
        cutoff = 2
    else:
        print("Please set grade cutoff to A or B!")
    input_pad = np.zeros((1, length, dim_input_course+dim_input_grade+dim_input_major), int)  # padded input
    sem_flag = 0
    stu_course_grade = []
    for k in data:
        if k != {}:
            major_name = [id_major[i] for i in k['major']]
            input_pad[0, sem_flag, dim_input_course+dim_input_grade+np.array(k['major'])] = 1
            stu_course_grade.append('sem '+str(sem_flag)+': ')
            stu_course_grade.append('major: ' + str(major_name) + ': ')
            for s in k['course_grade']:
                stu_course_grade.append(id_course[s[0]]+'-'+id_grade[s[1]]+' ')
                if s[1] <= cutoff:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4] = 1
                elif s[1] <= 5:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4 + 1] = 1
                elif s[1] == 6:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4 + 2] = 1
                elif s[1] == 7:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4 + 3] = 1
                if sem_flag != 0:
                    input_pad[0, sem_flag-1, s[0]] = 1
            sem_flag += 1
    # write the student's grade histories
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(stu_course_grade)

    # return the input for RNN, length, and the courses the student has not already taken 
    courses = np.where(np.sum(input_pad[0, :, :dim_input_course], axis=0) == 0)[0]
    return input_pad, [length], courses


def cal_hiddenstates(data_per, k, model):  # the recommended semester is k
    # filter: only consider courses in that semester
    rec_courses = sem_courses[k]

    model.eval()
    processed_data = process_data(data_per[:k], dim_input_course, dim_input_grade, dim_input_major)
    padded_input = Variable(torch.Tensor(processed_data[0]), requires_grad=False)
    seq_len = processed_data[1]
    # clear hidden states
    model.batch_size = 1
    model.hidden = model.init_hidden()
    # compute output
    y_pred = model(padded_input, seq_len)
    y_last_period = y_pred[0, -1]
    y_last_period = y_last_period.contiguous().view(dim_input_course, 4)
    y_B_exp = torch.exp(y_last_period[:, :2])
    y_B = y_B_exp / torch.sum(y_B_exp, dim=1)[:, None]
    y_B = y_B.data.numpy()[:, 0]

    # filter: only consider courses with grade predicted >= cutoff (A or B)
    where = np.where(y_B > 0.5)[0]
    candidate = set(rec_courses).intersection(set(where))
    # filter: don't consider courses student has already taken
    candidate = candidate.intersection(set(processed_data[2]))
    new_hidden = model.hidden
    hidden_shape0 = new_hidden[0].data.size()[0]
    hidden_shape2 = new_hidden[0].data.size()[2]
    new_hidden_a = new_hidden[0].data.expand(hidden_shape0, len(candidate), hidden_shape2)
    new_hidden_b = new_hidden[1].data.expand(hidden_shape0, len(candidate), hidden_shape2)
    new_hidden = (Variable(new_hidden_a), Variable(new_hidden_b))
    return new_hidden, candidate


def cal_output(model, hidden, loader, input_data, batchsize):

    model.eval()
    y_pred_all = torch.FloatTensor(np.zeros((len(loader) * batchsize, 1, len(course_id) * 4)))
    model.batch_size = batchsize
    for step, (batch_x, batch_y) in enumerate(loader):  # batch_x: index of batch data
        padded_input = input_data[batch_x]
        seq_len = [1 for i in range(batchsize)]
        padded_input = Variable(torch.Tensor(padded_input), requires_grad=False)
        # clear hidden states
        model.hidden = hidden
        # compute output (batch_size, 1, dim_output)
        y_pred = model(padded_input, seq_len)
        y_pred_all[range(step * batchsize, (step + 1) * batchsize)] = y_pred.data
    y_pred_temp = y_pred_all.view(y_pred_all.size()[0], 1, len(course_id), 4)
    y_pred_temp1 = torch.exp(y_pred_temp[:, :, :, :2])
    y_pred_all = y_pred_temp1 / torch.sum(y_pred_temp1, dim=3)[:, :, :, None]
    y_pred_all = y_pred_all.contiguous().view(y_pred_all.size()[0], 1, len(course_id) * 2)
    return y_pred_all


def level(num):
    if num <= 99:
        level = 0
    elif num <= 199:
        level = 1
    else:
        level = 2
    return level


def recommend_evaluation(model, hidden, candidicates, real_courses):  # candicates course id
    model.eval()
    batchsize = len(candidicates)
    prob = np.zeros(batchsize)
    input_data_return = get_input_gradedata(course_id[args.target_course], candidicates)
    input_data = input_data_return[0]
    dic_inputid_courseid = input_data_return[1]
    input_data_index = torch.IntTensor(np.array(range(len(input_data))))
    input_torch_data_index = Data.TensorDataset(input_data_index, input_data_index)
    input_loader = Data.DataLoader(dataset=input_torch_data_index, batch_size=batchsize, shuffle=False, num_workers=2, drop_last=False)
    output = cal_output(model, hidden, input_loader, input_data, batchsize)

    # filter: not recommend upper level course
    split = args.target_course.split(' ')
    t_num = split[1]
    t_sub = ' '.join(split[:-1])
    t_num = int(''.join(list(filter(str.isdigit, t_num))))
    t_level = level(t_num)
    for i, j in zip(dic_inputid_courseid.keys(), dic_inputid_courseid.values()):
        c_num = id_course[j].split(' ')[0]
        c_num = int(''.join(list(filter(str.isdigit, c_num))))
        c_level = level(c_num)
        if c_level <= t_level:
            prob[i] = output[i, 0, course_id[args.target_course] * 2]
    
    dic_courseid_inputid = dict(zip(dic_inputid_courseid.values(), dic_inputid_courseid.keys()))

    # filter: related subject
    rank = np.argsort(-prob)
    related_sub = target_prereqs_sub_filter[subject_id[t_sub]]
    related_course = [j for j in dic_inputid_courseid.values() if course_id_sub_id[j] in related_sub and prob[dic_courseid_inputid[j]]!=0 and j != course_id[args.target_course]]
    related_course = [dic_courseid_inputid[i] for i in related_course]
    related_course_ranked = list(set(rank).intersection(set(related_course)))
    related_course_ranked.sort(key=list(rank).index)
    unrelated_course_ranked = list(set(rank)-set(related_course_ranked))
    unrelated_course_ranked.sort(key=list(rank).index)
    rank = related_course_ranked + unrelated_course_ranked
    rank = rank[:N]

    course_rank = [dic_inputid_courseid[i] for i in rank]
    correct = list(set(course_rank).intersection(set(real_courses)))
    correct.sort(key=course_rank.index)
    correct_courseNM = [id_course[i] for i in correct]
    real_courses = [id_course[i] for i in real_courses]
    predict_courses = [id_course[i] for i in course_rank]
    label = ['label: ', real_courses]
    predict = ['predict: ', predict_courses]
    correct_c =['correct: ', correct_courseNM]
    with open(args.results_path+file_name, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(label)
        writer.writerow(predict)
        writer.writerow(correct_c)
        writer.writerow([])
    if correct!=[]:
        return 1
    else:
        return 0


if __name__ == '__main__':
    args = parse_arguments()
    print('goal course: ', args.target_course)
    file_name = args.target_course+'.tsv'
    N = args.topN

    target_prereqs_sub_filter = pickle.load(open(args.target_sub_pre_sub_filter, 'rb'))
    subject_id = pickle.load(open(args.subject_id_path, 'rb'))['subject_id']
    course_id_sub_id = pickle.load(open(args.course_id_sub_id, 'rb'))

    f = open(args.course_id_path, 'rb')
    course = pickle.load(f)
    course_id = course['course_id']
    id_course = course['id_course']
    f = open(args.major_id_path, 'rb')
    major = pickle.load(f)
    major_id = major['major_id']
    id_major = major['id_major']
    dim_input_major = len(major_id)
    f = open(args.grade_id_path, 'rb')
    grade = pickle.load(f)
    id_grade = grade['id_grade']
    dim_input_course = len(course_id)
    dim_input_grade = len(course_id) * 4
    model_name = args.evaluated_model_path
    model = torch.load(model_name, map_location=lambda storage, loc: storage)
    model.eval()
    f = open('sem_courses.pkl', 'rb')
    sem_courses = pickle.load(f)

    # divide students into well-performing and under-performing by a grade cutoff (A or B)
    grade_cutoff = args.grade_cutoff
    eval_set_positive, eval_set_negetive = select_evaluation_set(course_id[args.target_course], grade_cutoff)
    count = 0
    for i in range(len(eval_set_positive)):
        # find the recommended semester so that we can know which courses to recommend for each students
        num = list(range(len(eval_set_positive[i, :-1])))
        num.reverse()
        k = 0
        for j in num:
            # find the recent one semester with enrollment histories
            if eval_set_positive[i, j] != {}:
                k = j
                real_courses = np.array(eval_set_positive[i, k]['course_grade'])[:, 0]
                break
        hidden, candidates = cal_hiddenstates(eval_set_positive[i], k, model)
        count += recommend_evaluation(model, hidden, candidates, real_courses)
    acc = count/len(eval_set_positive)
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['positive hit rate', acc])
    print('positive hit rate:', acc)

    count = 0
    for i in range(len(eval_set_negetive)):
        num = list(range(len(eval_set_negetive[i, :-1])))
        num.reverse()
        k = 0
        for j in num:
            if eval_set_negetive[i, j] != {}:
                k = j
                real_courses = np.array(eval_set_negetive[i, k]['course_grade'])[:, 0]
                break
        hidden, candidates = cal_hiddenstates(eval_set_negetive[i], k, model)
        count +=  recommend_evaluation(model, hidden, candidates, real_courses)
    acc = count/len(eval_set_negetive)
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['negative hit rate', acc])
    print('negative hit rate:', acc)

