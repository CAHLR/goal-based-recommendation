import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--target_course', type=str, default='Subject_33 101')

    parser.add_argument('--input_path', type=str, default='../data_preprocess/stu_sem_major_grade_condense.pkl',
                        help='path to the preprocessed data for training')

    parser.add_argument('--course_id_path', type=str, default='../data_preprocess/course_id.pkl',
                        help='path to the generated course dictionary')

    parser.add_argument('--subject_id_path', type=str, default='../prerequisite_evaluation/subject_id.pkl',
                        help='path to the generated subject dictionary')

    parser.add_argument('--major_id_path', type=str, default='../data_preprocess/major_id.pkl',
                        help='path to the generated major dictionary')

    parser.add_argument('--grade_id_path', type=str, default='../data_preprocess/grade_id.pkl',
                        help='path to the generated grade dictionary')

    parser.add_argument('--grade_cutoff', type=str, default='A',
                        help='grade cutoff to separate grades into two categories - should be consistent with the value in grade_prediction')

    parser.add_argument('--evaluated_model_path', type=str, default='../grade_prediction/models/LSTM1_50lr0.001drp0wd0clp0.pkl',
                        help='the path to the model saved from training for evaluation')

    parser.add_argument('--topN', type=int, default=10,
                        help='Top N results for evaluation')

    parser.add_argument('--target_sub_pre_sub_filter', type=str, default='../prerequisite_evaluation/target_sub_pre_sub_filter.pkl',
                        help='a dictionary mapping the subject of a target course to subjects of its prerequisite courses')

    parser.add_argument('--course_id_sub_id', type=str,
                        default='../prerequisite_evaluation/course_id_sub_id.pkl',
                        help='a dictionary mapping the id of a course to the id of its subject')

    parser.add_argument('--evaluated_semester', type=int, default=15,
                        help='the order position of the semester for validation/evaluation, which depends on the input data.'
                             'In this data, 15 refers to 2019 Fall as the semester for validation, 16 refers to 2020 Spring for evaluation')

    parser.add_argument('--results_path', type=str, default='results/',
                        help='path to the generated results folder to save the correctly predicted course pairs')


    return parser.parse_args()