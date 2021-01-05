__author__ = 'jwj'
import pickle
import numpy as np


"""
# training data: 2008 Fall to 2016 Fall
# test data: 2017 Spring
# subtraining data: 2008 Fall to 2015 Fall
# validation data: 2016 Spring
"""

def empty(check):
    if check[0] == {}:
        return 0
    else:
        return 1


def get_data_from_condense_seq(t):  # t==22, subtraining and validation; t==25, training and testing
    f = open('../../RNN2/data_preprocess/stu_sem_major_grade_condense.pkl', 'rb')
    data = pickle.load(f)['stu_sem_major_grade_condense']
    # data is a nested list, each row is a list: [{'major': major_id, 'grade': [course_id, grade_id]},{...},...{...}], each {} is for a semester.
    data = np.array(data)
    train = data[:, :t]
    vali = data[:, :t+1]
    f.close()
    num_stu = train.shape[0]
    num_sem = np.zeros(num_stu)
    num_sem1 = np.zeros(num_stu)

    for i in range(num_stu):
        nonempty_sem = np.apply_along_axis(empty, 0, train[i][np.newaxis, :])
        num_sem[i] = np.sum(nonempty_sem)
        nonempty_sem = np.apply_along_axis(empty, 0, vali[i][np.newaxis, :])
        num_sem1[i] = np.sum(nonempty_sem)

    # filter training data, periods >=2
    index = np.where(num_sem >= 2)[0]
    training_data = train[index]

    # filter validation data, periods >=2
    index = np.where(num_sem1 >= 2)[0]
    validation_data = vali[index]
    # filter validation data again, the last semester shouldn't be null
    vali = validation_data[:, t]
    nonempty_vali = np.apply_along_axis(empty, 0, vali[np.newaxis, :])
    index = np.where(nonempty_vali > 0)[0]
    validation_data = validation_data[index][:, :t+1]
    print("Number of sub-training samples: ", len(training_data))
    print("Number of validation samples: ", len(validation_data))
    return training_data, validation_data


# padding
# input: each row: [{'major': major_id, 'grade': [course_id, grade_id]},{...},...{...}], each {} is for a semester. output: dim(batchsize, max_semester_number, dim_grade+dim_course+dim_major)
def process_data(index, data, batchsize, dim_input_course, dim_input_grade, dim_input_major):
    batch = data[index]
    num_sem = np.zeros(batchsize, int)
    for i in range(batchsize):
        nonempty_sem = np.apply_along_axis(empty, 0, batch[i][np.newaxis, :])
        num_sem[i] = np.sum(nonempty_sem)
    sort_index = np.argsort(-num_sem)
    max_seq = num_sem.max()
    pro_grade = np.zeros((batchsize, int(max_seq), dim_input_grade), int)  # padded input
    pro_course = np.zeros((batchsize, int(max_seq), dim_input_course), int)  # padded input
    pro_major = np.zeros((batchsize, int(max_seq), dim_input_major), int)  # padded input
    input_padded = np.zeros((batchsize, int(max_seq), dim_input_course + dim_input_grade + dim_input_major), int)  # padded input
    label_padded = np.zeros((batchsize, int(max_seq)-1, dim_input_grade), int)  # padded label
    stu_flag = 0
    for j in sort_index:
        sem_flag = 0
        for k in batch[j]:
            if k != {}:
                pro_major[stu_flag, sem_flag, k['major']] = 1
                for s in k['grade']:
                    if s[1] <= 2:
                        pro_grade[stu_flag, sem_flag, s[0] * 4] = 1
                    elif s[1] <= 5:
                        pro_grade[stu_flag, sem_flag, s[0] * 4 + 1] = 1
                    elif s[1] == 6:
                        pro_grade[stu_flag, sem_flag, s[0] * 4 + 2] = 1
                    elif s[1] == 7:
                        pro_grade[stu_flag, sem_flag, s[0] * 4 + 3] = 1
                    pro_course[stu_flag, sem_flag, s[0]] = 1
                sem_flag += 1
        label_padded[stu_flag] = pro_grade[stu_flag, 1:]
        input_padded[stu_flag, :, dim_input_course:dim_input_course+dim_input_grade] = pro_grade[stu_flag]
        input_padded[stu_flag, :-1, :dim_input_course] = pro_course[stu_flag, 1:]
        input_padded[stu_flag, :, -dim_input_major:] = pro_major[stu_flag]
        input_padded[stu_flag, sem_flag-1] = 0
        stu_flag += 1

    # cut 0 vectors for the second dimension of input_padded.
    input_padded = input_padded[:, :-1]

    label_len = np.sum(label_padded, axis=2)
    label_sem = np.zeros((label_len.shape[0], label_len.shape[1]), int)
    label_sem[np.nonzero(label_len)] = 1
    label_length = np.sum(label_sem, axis=1)

    input_len = np.sum(input_padded, axis=2)
    input_sem = np.zeros((input_len.shape[0], input_len.shape[1]), int)
    input_sem[np.nonzero(input_len)] = 1
    input_length = np.sum(input_sem, axis=1)

    if not (label_length == input_length).all():
        print("label length or input length wrong! wrong padding!")
        exit()
    return input_padded, input_length, label_padded


# pad batch data x(batch_size, sem_num, [course_id, grade_id]) to x(batch_size, max_seq, course_num*7)
def process_data_cat_continuous(index, data, batchsize, dim_input):

    # dict for mapping discrete grade to continuous grade
    grade_dic = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 2, 7: 1}  # discrete: continuous
    batch = data[index]
    num_sem = np.zeros(batchsize, int)
    for i in range(batchsize):
        nonempty_sem = np.apply_along_axis(empty, 0, batch[i][np.newaxis, :])
        num_sem[i] = np.sum(nonempty_sem)
    sort_index = np.argsort(-num_sem)
    max_seq = num_sem.max()
    pro_input_data = np.zeros((batchsize, int(max_seq), dim_input), int)  # padded input
    pro_label_data = np.zeros((batchsize, int(max_seq), int(dim_input / 7)), int)
    input_padded = np.zeros((batchsize, int(max_seq), dim_input), int)  # padded input
    label_padded = np.zeros((batchsize, int(max_seq)-1, int(dim_input / 7)), int)  # padded label
    stu_flag = 0
    for j in sort_index:
        sem_flag = 0
        for k in batch[j]:
            if k != []:
                for s in k:
                    pro_input_data[stu_flag, sem_flag, s[0] * 7 - 1 + s[1]] = 1
                    pro_label_data[stu_flag, sem_flag, s[0]] = grade_dic[s[1]]
                sem_flag += 1
        label_padded[stu_flag] = pro_label_data[stu_flag, 1:]
        input_padded[stu_flag] = pro_input_data[stu_flag]
        input_padded[stu_flag, sem_flag-1] = 0
        stu_flag += 1

    # cut 0 vectors for the second dimension of input_padded.
    input_padded = input_padded[:, :-1]

    label_len = np.sum(label_padded, axis=2)
    label_sem = np.zeros((label_len.shape[0], label_len.shape[1]), int)
    label_sem[np.nonzero(label_len)] = 1
    label_length = np.sum(label_sem, axis=1)

    input_len = np.sum(input_padded, axis=2)
    input_sem = np.zeros((input_len.shape[0], input_len.shape[1]), int)
    input_sem[np.nonzero(input_len)] = 1
    input_length = np.sum(input_sem, axis=1)

    if not (label_length == input_length).all():
        print("label length or input length wrong! wrong padding!")
        exit()
    #print(input_length, label_length)

    return input_padded, input_length, label_padded

