## Goal-base Course Recommendation
### [Publication](https://dl.acm.org/doi/10.1145/3303772.3303814):
[Jiang, W.](https://www.jennywjjiang.com), [Pardos, Z.A.](https://gse.berkeley.edu/zachary-pardos), Wei, Q. (2019) Goal-based Course Recommendation. In C. Brooks, R. Ferguson & U. Hoppe (Eds.) Proceedings of the 9th International Conference on Learning Analytics and Knowledge (LAK). ACM. Tempe, Arizona. Pages 36-45.

This repo includes code for the three connected tasks in the paper, i.e., student grade prediction (section 5), prerequisite course inference (section 6), and personalized prerequsite course recommendation (section 7). 


### Prerequisites:
* environment:
	* python3
	* pytorch
	* install other dependencies by: pip install -r requirements.txt
* preprocessed data:
	* student enrollment data: a 2D python list
	* course dictionary: a python dictionary mapping courses to their preprocessed ID, and a reversed one
	* grade dictionary: a python dictionary mapping all types of grades to their preprocessed ID, and a reversed one
	* major dictionary: a python dictionary mapping majors to their preprocessed ID, and a reversed one


### 1. Student Grade Prediction:
In order to train the proposed LSTM grade prediction model, you should provide a student enrollment dataset in a preprocessed 2D python list with dimention `n×m`, where `n` is the number of students and `m` is the number of semesters covered in your dataset: <img src="https://latex.codecogs.com/gif.latex?[s_1,&space;s_2,&space;s_3,&space;...,&space;s_n]" title="[s_1, s_2, s_3, ..., s_n]" />
, where <img src="https://latex.codecogs.com/gif.latex?s_i=[t_{i1},&space;t_{i2},&space;...,&space;t_{im}]" title="s_i=[t_{i1}, t_{i2}, ..., t_{im}]" />, and <img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /> denotes the preprocessed enrollment histories of the i-th student in your data (multiple semesters) and <img src="https://latex.codecogs.com/gif.latex?t_{ik}" title="t_{ik}" /> represents the specific enrollment histories of the i-th student in the k-th semester. Note that the k-th semester of all the students refers to the same semester, for example, m=3, which means there are 3 semesters covered in your data: Fall 2019, Spring 2020, Summer 2020, then <img src="https://latex.codecogs.com/gif.latex?t_{i2}&space;(i=1,2,...,n)" title="t_{i2} (i=1,2,...,n)" /> should contain enrollment histories of Spring 2020 for all students in your data. <img src="https://latex.codecogs.com/gif.latex?t_{ik}=\{\}" title="t_{ik}=\{\}" /> (empty) if the i-th student did not enroll in any course in semester k.

The format of <img src="https://latex.codecogs.com/gif.latex?t_{ik}" title="t_{ik}" /> is a python dictionary:

{'major': <img src="https://latex.codecogs.com/gif.latex?m_{ik}" title="m_{ik}" />, 'grade': <img src="https://latex.codecogs.com/gif.latex?[[c_{ik}^1,&space;g_{ik}^1],[c_{ik}^2,&space;g_{ik}^2],...,[c_{ik}^p,&space;g_{ik}^p]]" title="[[c_{ik}^1, g_{ik}^1],[c_{ik}^2, g_{ik}^2],...,[c_{ik}^p, g_{ik}^p]]" />},

 where <img src="https://latex.codecogs.com/gif.latex?m_{ik}" title="m_{ik}" /> refers to the major ID of the i-th student's major in the k-th semester, <img src="https://latex.codecogs.com/gif.latex?[c_{ik}^p,&space;g_{ik}^p]" title="[c_{ik}^p, g_{ik}^p]" /> refers to the course ID and grade ID the i-th student received for the p-th course they enrolled in the k-th semester. 
 
* **command**
	*  training: python train.py
	*  evaluation: python evaluate.py
	*  



