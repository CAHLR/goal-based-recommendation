# Goal-base Course Recommendation
## Introduction

This repo includes code for running the three connected tasks (student grade prediction--section 5, prerequisite course inference--section 6, and personalized prerequsite course recommendation--section 7) in this paper:

* [Jiang, W.](https://www.jennywjjiang.com), [Pardos, Z.A.](https://gse.berkeley.edu/zachary-pardos), Wei, Q. (2019) [Goal-based Course Recommendation.](https://dl.acm.org/doi/10.1145/3303772.3303814) In C. Brooks, R. Ferguson & U. Hoppe (Eds.) *Proceedings of the 9th International Conference on Learning Analytics and Knowledge* (LAK). ACM. Tempe, Arizona. Pages 36-45.


### Dataset Descriptions:

Due to FERPA privacy protection, we cannot publish the original student enrollment dataset. However, here we provid a sythetic dataset that consists of simulated student enrollment data and student major data to serve as an example of the formatting used to work with the code. The dataset is located in folder [synthetic\_data\_samples](https://github.com/CAHLR/goal-based-recommendation/tree/master/synthetic_data_samples). 

*  _synthetic\_enrollment\_data.csv_

|  Sememster  | Anonymized Student ID | Course               | Grade  |
|:-----------:|:---------------------:|:--------------------:|:------:|
|  2014 Fall  |         103674        | Education C200       |    B   |
| 2015 Spring |         104251        | Computer Science 61B |    A   |
| 2015 Summer |         102673        | Sociology 1          | Credit |
|...|...|...|...|

Grade types: Letter grades -- A, B, C, D, F; Non-letter grades -- Credit and No Credit.

* _synthetic\_major\_data.csv_

|  Sememster  | Anonymized Student ID | Major            |
|:-----------:|:---------------------:|:----------------:|
|  2014 Fall  |         103674        | Education        |
| 2015 Spring |         104251        |       Math       |
| 2015 Spring |         104251        | Computer Science |
| 2015 Summer |         102673        | Sociology        |
|...|...|...|

Note that a student may have multiple majors in a semester, which are listed in multiple rows.

## Steps for Runing the Code
### Environment Prerequisites:
* python3
* pytorch
* install other dependencies by: *pip3 install -r requirements.txt*

	
### Data Preprocessing

**-- command:**

* Set up global parameters in _data\_preprocess/utils.py_
* `python data_preprocess/preprocess.py`
	
This command hard codes the locations of the expected data files to be in the synthetic data folder. This path can be changed in utils.py.

Then the following intermediate files will be generated for model training:
	
* **course dictionaries (_course\_id.pkl_)**: a pair of python dictionaries mapping courses to their preprocessed ID and vice versa.
* **grade dictionary (_grade\_id.pkl_)**: a pair of python dictionaries mapping all types of grades to their preprocessed ID and vice versa. 
* **major dictionary (_major\_id.pkl_)**: a pair of python dictionaries mapping majors to their preprocessed ID and vice versa
* **semester dictionary (_semester\_id.pkl_)**: a pair of python dictionaries mapping semesters to their preprocessed ID vice versa. For example, the earliest semester in the dataset, 2014 Fall, will be set 0 as its ID. 
* **condensed student enrollments and grades (_stu\_sem\_major\_grade\_condense.pkl_)**: a 2D python list with dimention `n×m`, where `n` is the number of students and `m` is the number of semesters covered in the dataset: <img src="https://latex.codecogs.com/gif.latex?[s_1,&space;s_2,&space;s_3,&space;...,&space;s_n]" title="[s_1, s_2, s_3, ..., s_n]" />
, where <img src="https://latex.codecogs.com/gif.latex?s_i=[t_{i1},&space;t_{i2},&space;...,&space;t_{im}]" title="s_i=[t_{i1}, t_{i2}, ..., t_{im}]" />, and <img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /> denotes the preprocessed enrollment histories of the i-th student in your data (multiple semesters) and <img src="https://latex.codecogs.com/gif.latex?t_{ik}" title="t_{ik}" /> represents the specific enrollment histories of the i-th student in the k-th semester. Note that the k-th semester of all the students refers to the same semester, for example, m=3, which means there are 3 semesters covered in your data: Fall 2019, Spring 2020, Summer 2020, then <img src="https://latex.codecogs.com/gif.latex?t_{i2}&space;(i=1,2,...,n)" title="t_{i2} (i=1,2,...,n)" /> will contain enrollment histories of Spring 2020 for all students in your data. <img src="https://latex.codecogs.com/gif.latex?t_{ik}=\{\}" title="t_{ik}=\{\}" /> (empty) if the i-th student did not enroll in any course in semester k.
The format of <img src="https://latex.codecogs.com/gif.latex?t_{ik}" title="t_{ik}" /> is a python dictionary: <img src="https://latex.codecogs.com/gif.latex?\{'major':&space;m_{ik},&space;'course\_grade':&space;[(c_{ik}^1,&space;g_{ik}^1),(c_{ik}^2,&space;g_{ik}^2),...,(c_{ik}^p,&space;g_{ik}^p)]\}" title="\{'major': m_{ik}, 'course\_grade': [(c_{ik}^1, g_{ik}^1),(c_{ik}^2, g_{ik}^2),...,(c_{ik}^p, g_{ik}^p)]\}" />, where <img src="https://latex.codecogs.com/gif.latex?m_{ik}" title="m_{ik}" /> refers to the major ID of the i-th student's major in the k-th semester, and <img src="https://latex.codecogs.com/gif.latex?(c_{ik}^p,&space;g_{ik}^p)" title="(c_{ik}^p, g_{ik}^p)" /> refers to the course ID of the p-th course the i-th student enrolled and the grade ID received for that course in the k-th semester. 
 
### 1. Student Grade Prediction:
 
**-- command**

* `cd grade_prediction`
*  Set up arguments and hyperparameters for training in _grade\_prediction/utils.py_ (optional)
*  training: `python train.py`
	*  The best model(.pkl) and the log file that records the training loss and validation loss will be saved in [_grade\_prediction/models_](https://github.com/CAHLR/goal-based-recommendation/tree/master/grade_prediction/models). 
*  Set up _evaluated\_model\_path_ and _evaluated\_semester_ in _grade\_prediction/utils.py_, which corresponds to the model and semester you aim to evaluate (optional).
*  evaluation: `python evaluate.py`. 
	* Evaluation results will be printed out based on these metrics: 

		* grade prediction accuracy on enrollments with letter grades
		* grade prediction accuracy on enrollments with non-letter grades
		* overall grade prediction accuracy
		* true positive rate, true negative rate, false negative rate, and false positive rate on letter grade prediction and non-letter grade prediction
		* F-score on letter grade prediction and non-letter grade prediction, overall F-score

### 2. Prerequisite Course Inference:

Use the trained grade prediction model to infer prerequisite courses for a given course, and evaluate the model on the official (synthetic) prerequisite course list (*synthetic_data\_samples/synthetic\_prereqs\_pairs.csv*):

|  prerequisite course  | target course | 
|:-----------:|:---------------------:|
|   Computer Science 70 |     Computer Science 188        | 
| Computer Science 61B |   Computer Science 188              |    
| Education 1 |         Education 200     | 
|...        |...| 

**-- command**:

* `cd prerequisite_evaluation`
* set up arguments in _prerequisite\_evaluation/utils.py_ (optional)
* Generate filters: `python generate_filters.py`
	* Filter files (.pkl) will be saved in the current directory.
	* A python dictionary (*target_id.pkl*) that maps all the target courses to their IDs and the vice versa will be saved in the current directory. The IDs of target courses are from 0 to the number of target course. 
* Evaluate on a single target course (optional): `python prereqs_evaluation.py --target_course_id xxx
`, where `xxx` is the ID of a target course in *target_id.pkl*.
	* 	This will save the correctly predicted target-prereq course pairs into a file named *xxx.tsv* in  [_prerequisite\_evaluation/results_](https://github.com/CAHLR/goal-based-recommendation/tree/master/prerequisite_evaluation/results)
* Evaluate on all target courses: It takes time to evaluate on a target course, so we use the command *qsub* to evaluate on multiple target course parallelly.
	* `for i in seq 0 n; do echo /usr/bin/python /xxx/.../prereqs_evaluate.py --target_course_id $i|qsub; done `
	* `n` is the total number of target courses
	* `/xxx/.../prereqs_evaluate.py` refers to the absolute path to the file.
* Merge evaluation results of all target courses: `cat results/*.tsv > all.tsv`


### 3. Personalized Prerequisite Course Recommendation

**-- command:**

* `cd student_evaluation`
* Set up arguments in _student\_evaluation/utils.py_
* Generate a filter: `python generate_sem_courses.py`
	* A filter file (.pkl) will be saved in the current folder.
* Evaluate on a goal course: `python personalized_prereqs_evaluation.py --target_course xxx`, where `xxx `is the name of a course (e.g., Subject_33 101) that you intend to set as a goal(target) course.
	* This will print out (1) the number of well-performing students and under-performing students in this course in the evaluated semester, (2) the recommendation accuracy for the two groups of students.
	* This will also save the enrollment histories and the recommended courses in the evaluated semester of these students to a csv file in [_student\_evaluation/results_](https://github.com/CAHLR/goal-based-recommendation/tree/master/student_evaluation/results)


## Contact and Citation
Please do not hesitate to contact us (jiangwj[at]berkeley[dot]edu, pardos[at]berkeley[dot]edu) if you have any questions. We appreciate your support and citation if you find this work useful.

```
@inproceedings{jiang2019goal,
  title={Goal-based course recommendation},
  author={Jiang, Weijie and Pardos, Zachary A and Wei, Qiang},
  booktitle={Proceedings of the 9th International Conference on Learning Analytics \& Knowledge},
  pages={36--45},
  year={2019}
}
```
 
	

