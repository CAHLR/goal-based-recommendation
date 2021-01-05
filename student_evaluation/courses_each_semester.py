__author__ = 'jwj'
import numpy as np
import pickle


if __name__ == '__main__':

    f = open('/research/jenny/RNN/data_preprocess/course_id.pkl', 'rb')
    course = pickle.load(f)
    course_id = course['course_id']
    id_course = course['id_course']
    f = open('/research/jenny/RNN2/data_preprocess/stu_sem_major_grade_condense.pkl', 'rb')
    data = pickle.load(f)['stu_sem_major_grade_condense']
    data = np.array(data)
    num_sem = data.shape[1]
    dic = dict()
    for i in range(num_sem):
        courses_grades = data[:, i]
        courses = []
        for j in courses_grades:
            if j != {}:
                courses.extend(np.array(j['grade'])[:, 0])
        courses = set(courses)
        dic[i] = courses
    f = open('sem_courses.pkl', 'wb')
    pickle.dump(dic, f)
    f.close()
