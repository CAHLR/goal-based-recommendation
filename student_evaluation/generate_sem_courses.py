__author__ = 'jwj'
import numpy as np
import pickle
from utils import *


if __name__ == '__main__':
    args = parse_arguments()
    f = open(args.course_id_path, 'rb')
    course = pickle.load(f)
    course_id = course['course_id']
    id_course = course['id_course']
    f = open(args.input_path, 'rb')
    data = pickle.load(f)['stu_sem_major_grade_condense']
    data = np.array(data)
    num_sem = data.shape[1]
    dic = dict()
    for i in range(num_sem):
        courses_grades = data[:, i]
        courses = []
        for j in courses_grades:
            if j != {}:
                courses.extend(np.array(j['course_grade'])[:, 0])
        courses = set(courses)
        dic[i] = courses
    f = open('sem_courses.pkl', 'wb')
    pickle.dump(dic, f)
    f.close()
