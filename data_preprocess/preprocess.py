__author__ = 'jwj'
import pandas as pd
import pickle
import numpy as np
from utils import *


# filter out courses enrolled less than x times
def preprocess(enroll_data, x):
    count = enroll_data.groupby('course').size()
    count = count.to_frame().reset_index()
    count.columns = ['course', 'num']
    count = count.loc[count['num'] >= x]
    data = enroll_data.loc[enroll_data['course'].isin(count['course'])]
    data.drop_duplicates(inplace=True)
    return data


# get student id, filter out students with enrollment records in less than 2 semesters
# goes after preprocess()
def get_stu(enroll_data):
    stu = []
    data = enroll_data.groupby(' Student Identifier(ppsk)')
    for ppsk in data.groups.keys():
        data1 = data.get_group(ppsk).groupby('Semester Year Name Concat')
        num = len(data1.groups.keys())
        if num >= 2:
            stu.append([ppsk, num])
    #print(stu)
    stu = sorted(stu, key=lambda x: x[1], reverse=True)
    dic_stu = {}
    reverse_dic_stu = {}
    for i in enumerate(stu):
        dic_stu[i[0]] = i[1]
        reverse_dic_stu[i[1][0]] = i[0]
    stu_id = {'id_ppsk_num': dic_stu, 'ppsk_id': reverse_dic_stu}
    f = open('stu_id.pkl', 'wb')
    pickle.dump(stu_id, f)
    return stu


# add student major to enrollment data
# goes after preprocess() and get_stu()
def add_major_to_data(enroll_data, major_data):
    f = open('stu_id.pkl', 'rb')
    ppsk_id = pickle.load(f)['ppsk_id']
    stu = ppsk_id.keys()
    data = enroll_data.loc[enroll_data[' Student Identifier(ppsk)'].isin(stu)]
    data = pd.merge(data, major_data, how='left', on=['Semester Year Name Concat', ' Student Identifier(ppsk)'])
    return data


# goes after add_major_to_data() because have to only keep majors that are corrospond to enrollent records.
def get_major(data):
    count = data.groupby('major').size()
    count = pd.core.frame.DataFrame({'count': count}).reset_index()
    count.sort_values(by=['count'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.loc[:, ['major', 'count']]
    dic1 = dict()
    li = list(count['major'])
    for i in li:
        dic1[i] = len(dic1)
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {'major_id': dic1, 'id_major': reversed_dic1}
    f = open('major_id.pkl', 'wb')
    pickle.dump(alldata, f)


# get course id
# goes after preprocess and get_stu()
def get_course(data):
    f = open('stu_id.pkl', 'rb')
    stu = pickle.load(f)['ppsk_id'].keys()
    data = data.loc[data[' Student Identifier(ppsk)'].isin(stu)]
    count = data.groupby('course').size()
    count = pd.core.frame.DataFrame({'count': count}).reset_index()
    count.sort_values(by=['count'], ascending=False, inplace=True)
    count.reset_index(inplace=True)
    count = count.iloc[:, [1, 2]]
    dic = count.to_dict('dict')
    dic1 = dic['course']
    reversed_dic1 = dict(zip(dic1.values(), dic1.keys()))
    alldata = {'course_id': reversed_dic1, 'id_course': dic1}
    f = open('course_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


def set_semester_dic(data):
    year_sem = data['Semester Year Name Concat'].drop_duplicates().str.split()
    year = year_sem.str[0]
    sem = year_sem.str[1]
    year_sem = pd.DataFrame(zip(year, sem))
    year_sem.columns = ['year', 'sem']
    year_sem_group = year_sem.groupby('year')
    year = year_sem['year'].drop_duplicates().sort_values().tolist()
    sem_dict = {}
    i = 0
    for j in year:
        get_sems = year_sem_group.get_group(j)['sem']
        get_sems = pd.Categorical(get_sems, ["Spring", "Summer", "Fall"])
        get_sems = get_sems.sort_values().tolist()
        for k in get_sems:
            sem_dict[i] = str(j) + ' ' + k
            i += 1
    reverse_sem_dict = dict(zip(sem_dict.values(), sem_dict.keys()))
    alldata = {'semester_id': reverse_sem_dict, 'id_semester': sem_dict}
    f = open('semester_id.pkl', 'wb')
    pickle.dump(alldata, f)
    f.close()


def grade_dic():
    grade_id = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5, 'Credit': 6, 'No Credit': 7}
    id_grade = dict(zip(grade_id.values(), grade_id.keys()))
    dict_id = {'grade_id': grade_id, 'id_grade': id_grade}
    f = open('grade_id.pkl', 'wb')
    pickle.dump(dict_id, f)
    f.close()


def get_condense_grade_major(data):
    f = open('stu_id.pkl', 'rb')
    ppsk_id = pickle.load(f)['ppsk_id']
    stu = ppsk_id.keys()
    f = open('semester_id.pkl', 'rb')
    sem_id = pickle.load(f)['semester_id']
    f = open('course_id.pkl', 'rb')
    course_id = pickle.load(f)['course_id']
    f = open('grade_id.pkl', 'rb')
    grade_id = pickle.load(f)['grade_id']
    f = open('major_id.pkl', 'rb')
    major_id = pickle.load(f)['major_id']
    f.close()
    num_stu = len(stu)
    num_sem = len(sem_id)
    data = data.loc[data[' Student Identifier(ppsk)'].isin(stu), [' Student Identifier(ppsk)', 'Semester Year Name Concat', 'course', 'Grade Subtype Desc', 'major']]
    data = data.drop_duplicates()

    pd_sem = pd.DataFrame(zip(sem_id.keys(), sem_id.values()))
    pd_sem.columns = ['sem', 'id']
    pd_sem = pd_sem.sort_values(by=['id'])
    ranked_sem = pd_sem['sem'].tolist()

    data['Semester Year Name Concat1'] = pd.Categorical(data['Semester Year Name Concat'], ranked_sem)
    data.sort_values(by=[' Student Identifier(ppsk)', 'Semester Year Name Concat1'], ascending=True, inplace=True)
    data.fillna(method='ffill', inplace=True)  # in case major is null
    data = data.groupby([' Student Identifier(ppsk)'])
    print(len(data.groups.keys()))
    mat_data = [[{} for i in range(num_sem)] for j in range(num_stu)]
    flag = 0
    for key in data.groups.keys():
        print(flag)
        flag += 1
        stu_data = data.get_group(key)
        #print(stu_data)
        stu_sem = stu_data.groupby(['Semester Year Name Concat'])
        for sem in stu_sem.groups.keys():
            dic = {'major': [], 'course_grade': []}
            stu_sem_data = stu_sem.get_group(sem)
            majors = stu_sem_data['major'].drop_duplicates().apply(lambda x: major_id[x]).tolist()
            dic['major'].extend(majors)
            sem_grade = stu_sem_data.groupby('course')
            for c in sem_grade.groups.keys():
                grade = min(sem_grade.get_group(c)['Grade Subtype Desc'].apply(lambda x: grade_id[x]).tolist())
                dic['course_grade'].append((course_id[c], grade))
            mat_data[ppsk_id[key]][sem_id[sem]] = dic
        #print(mat_data[ppsk_id[key]])
    mat_file = {'stu_sem_major_grade_condense': mat_data}
    f = open('stu_sem_major_grade_condense.pkl', 'wb')
    pickle.dump(mat_file, f, protocol=4)
    f.close()
    print(mat_data)


if __name__ == '__main__':
    args = parse_args()
    # loading enrollment data
    print('loading enrollment data, preprocessing...')
    enrollment_data = pd.read_csv(args.input_enrollment, header=0)
    major_data = pd.read_csv(args.input_major, header=0)
    enroll_num_threshold = args.enrollment_threshold
    enrollment_data = preprocess(enrollment_data, enroll_num_threshold)

    # filter students
    print('preprocessing students...')
    get_stu(enrollment_data)

    # generate courses
    print('preprocessing courses...')
    get_course(enrollment_data)

    # add major
    print('preprocessing majors...')
    data = add_major_to_data(enrollment_data, major_data)
    get_major(data)

    # generate semester id and grade id
    set_semester_dic(enrollment_data)
    grade_dic()

    print('generating preprocessed data for training...')
    get_condense_grade_major(data)