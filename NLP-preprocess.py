import re

import time
import pickle

questions = pickle.load( open( "Pract2/ques.pkl", "rb" ) )
answers = pickle.load( open( "Pract2/ans.pkl", "rb" ) )

# print(questions,answers)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-()\"#/@:;<>+=~|.?,]", "", text)
    return text

#cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# print(questions)

# print(clean_questions)

#cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
# print(clean_answers)


word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# print(word2count)

threshold= 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

# print(questionswords2int,answerswords2int)

tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

answersint2word = {w_i: w for w, w_i in answerswords2int.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Converting words to numbers

questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)


answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)


print(questions_into_int)
time.sleep(0.5)
print(answers_into_int)
time.sleep(0.5)
print(questions_into_int)
time.sleep(0.5)
print(answers_into_int)
print(questions_into_int)
time.sleep(0.5)
print(answers_into_int)
print(questions_into_int)
time.sleep(0.5)
print(answers_into_int)
print(questions_into_int)
time.sleep(0.5)
print(answers_into_int)


# sorted_clean_questions = []
# sorted_clean_answers = []
# for length in range(1, 25 + 1):
#     for i in enumerate(questions_into_int):
#         if len(i[1]) == length:
#             sorted_clean_questions.append(i[1])
#             sorted_clean_answers.append(answers_into_int[i[0]])

# print(sorted_clean_questions,sorted_clean_answers)
