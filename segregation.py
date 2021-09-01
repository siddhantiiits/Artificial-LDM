import numpy as np
import tensorflow as tf
import re

import re
import time


corpus = open('Pract2/output.txt',encoding='utf-8',errors='ignore').read().split('\n')
# for c in corpus:
# 	print(c)
# 	print('\n\n+++++\n\n')
questions = []
answers = []
# print(corpus)
for case in range(len(corpus)-1):
	if not case:
		continue
	res = re.search(' \+\+\+\$\+\+\+ ', corpus[case])
	ques = (corpus[case][:res.start()])
	ans = (corpus[case][res.end():])

    # _case = case.split(' +++$+++ ')
    # questions.append(_case[0])
    # answers.append(_case[1])

print(ques,'++++++++\n\n++++++',ans)
