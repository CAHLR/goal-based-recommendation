__author__ = 'jwj'
import pickle
import pandas as pd
from utils import *


args = parse_arguments()


class prereqs_pairs():

    def __init__(self):
        super(prereqs_pairs, self).__init__()
        self.data_prereqs = pd.read_csv(args.prereqs_path, header=0)
        self.course = pickle.load(open(args.course_id_path, 'rb'))
        self.course_id = self.course['course_id']
        self.id_course = self.course['id_course']
        self.subject_id = {}
        self.id_subject = {}

    def gene_target_dic(self):
        self.data_prereqs['prereqs_id'] = self.data_prereqs['prereqs'].map(self.course_id)
        self.data_prereqs['target_id'] = self.data_prereqs['target'].map(self.course_id)
        self.data_prereqs.dropna(inplace=True)

        target_id = dict()
        target = self.data_prereqs['target_id'].drop_duplicates()
        for i in target:
            target_id[i] = len(target_id)
        id_target = dict(zip(target_id.values(), target_id.keys()))
        dic = {'target_id': target_id, 'id_target': id_target}
        #print(len(target_id))
        f = open('target_id.pkl', 'wb')
        pickle.dump(dic, f)
    """
    # for student goal-based evaluation
    def gene_target_sub_rele_sub(self):
        f = open('subject_id.pkl','rb')
        subject_id = pickle.load(f)
        self.subject_id = subject_id['subject_id']
        self.data_prereqs['target_sub'] = self.data_prereqs['target'].str.split(' ').str[1:].apply(' '.join).map(self.subject_id)
        self.data_prereqs['prereqs_sub'] = self.data_prereqs['prereqs'].str.split(' ').str[1:].apply(' '.join).map(self.subject_id)
        self.data_prereqs.dropna(inplace=True)
        #print(self.data_prereqs)
        target_sub_prereqs_sub = self.data_prereqs.loc[:,['target_sub', 'prereqs_sub']]
        target_sub_group = target_sub_prereqs_sub.groupby('target_sub')
        dic = {}
        for i in target_sub_group.groups.keys():
            group = target_sub_group.get_group(i)
            dic[i] = set(group['prereqs_sub'].tolist())
        f = open('target_sub_pre_sub_filter.pkl', 'wb')
        pickle.dump(dic, f)
    """

    def gene_subject_id_course_subject_dic(self):
        # generate course_id_subject_id_dict
        course_sub = pd.DataFrame({'course_id': list(self.id_course.keys()), 'course': list(self.id_course.values())})
        #print(course_sub)
        course_sub['subject'] = course_sub['course'].str.split(' ').str[:-1].apply(' '.join)
        sub = course_sub['subject'].drop_duplicates().reset_index(drop=True)
        self.id_subject = sub.to_dict()

        self.subject_id = dict(zip(self.id_subject.values(), self.id_subject.keys()))
        #print(self.subject_id)
        dic = {'subject_id': self.subject_id, 'id_subject': self.id_subject}
        f = open('subject_id.pkl', 'wb')
        pickle.dump(dic, f)

        course_sub['subject_id'] = course_sub['subject'].apply(lambda x: self.subject_id[x])
        course_id_sub_id = dict(zip(course_sub['course_id'].tolist(), course_sub['subject_id'].tolist()))
        f = open('course_id_sub_id.pkl','wb')
        #print(course_id_sub_id)
        pickle.dump(course_id_sub_id, f)

    def gene_target_relevant_sub(self):
        self.data_prereqs['target_sub'] = self.data_prereqs['target'].str.split(' ').str[:-1].apply(' '.join).map(self.subject_id)
        self.data_prereqs['prereqs_sub'] = self.data_prereqs['prereqs'].str.split(' ').str[:-1].apply(' '.join).map(self.subject_id)
        self.data_prereqs['prereqs'] = self.data_prereqs['prereqs'].map(self.course_id)
        self.data_prereqs['target'] = self.data_prereqs['target'].map(self.course_id)
        #print(self.data_prereqs)
        self.data_prereqs.dropna(inplace=True)
        #print(len(self.data_prereqs))

        target_sub_prereqs_sub = self.data_prereqs.loc[:, ['target_sub', 'prereqs_sub']].drop_duplicates()
        target_sub_group = target_sub_prereqs_sub.groupby('target_sub')
        dic = {}
        for i in target_sub_group.groups.keys():
            group = target_sub_group.get_group(i)
            dic[i] = set(group['prereqs_sub'].tolist())
        f = open('target_sub_pre_sub_filter.pkl','wb')
        pickle.dump(dic, f)

        target_prereqs_sub = self.data_prereqs.loc[:, ['target', 'prereqs_sub']].drop_duplicates()
        target_group = target_prereqs_sub.groupby('target')
        dic1 = {}
        for i in target_group.groups.keys():
            group = target_group.get_group(i)
            dic1[i] = set(group['prereqs_sub'].tolist())
        
        del self.data_prereqs['prereqs'], self.data_prereqs['prereqs_sub']
        self.data_prereqs.drop_duplicates(inplace=True)
        self.data_prereqs['target_sub_prereqs_sub'] = self.data_prereqs['target_sub'].map(dic)
        self.data_prereqs['target_prereqs_sub'] = self.data_prereqs['target'].map(dic1)
        temp = self.data_prereqs['target_sub_prereqs_sub'] - self.data_prereqs['target_prereqs_sub']
        temp = temp.apply(lambda x: list(x)) + self.data_prereqs['target_sub'].apply(lambda x: [x])
        self.data_prereqs['target_prereqs_sub_consider'] = temp
        print(temp)
        data = self.data_prereqs.loc[:, ['target', 'target_prereqs_sub_consider']]
        data_dic = dict(zip(data['target'], data['target_prereqs_sub_consider']))
        f = open('target_prereqs_filter.pkl', 'wb')
        pickle.dump(data_dic, f)


if __name__ == '__main__':

    prereqs_pair = prereqs_pairs()
    prereqs_pair.gene_target_dic()
    prereqs_pair.gene_subject_id_course_subject_dic()
    prereqs_pair.gene_target_relevant_sub()




