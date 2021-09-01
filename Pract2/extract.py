# importing required modules
import re
import PyPDF2
import csv
import pickle

# Create a variable


# Open a file and use dump()


# with open('output.tsv', 'w', newline='') as f_output:
#     tsv_output = csv.writer(f_output, delimiter='\t')
#     tsv_output.writerow(['Start','Intro','Facts','Decission'])
filing  = open("output.txt", "w+")
filing.write('This is the output file\n')
filing.close

ls = [['Start','Intro','Facts','Decission']]
ques = []
ans = []
for i in range(2,7):
# creating a pdf file object
	pdfFileObj = open(' '+str(i)+'.pdf', 'rb')

	# creating a pdf reader object
	pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

	# printing number of pages in pdf file
	total_pages=pdfReader.numPages
	print(pdfReader.numPages)
	script=''
	# creating a page object

	for page in range(1,total_pages):
		pageObj = pdfReader.getPage(page)

		script+=pageObj.extractText()

	# extracting text from page
	# print(script)

	# closing the pdf file object
	pdfFileObj.close()

	res = re.search(r'INTRODUCTION', script)
	start = (script[:res.start()])

	res = re.search(r'INTRODUCTION', script)
	res1 = re.search(r'THE FACTS', script)
	intro = (script[res.end()+1:res1.start()])

	res = re.search(r'THE FACTS', script)
	res1 = re.search(r'FOR THESE REASONS, THE COURT', script)
	facts = (script[res.end()+1:res1.start()])

	res = re.search(r'FOR THESE REASONS, THE COURT', script)
	forTheseReasons = (script[res.end()+1:])
#

	q = intro +facts
	ques.append(q)
	a = forTheseReasons
	ans.append(a)


	filing  = open("output.txt", "a+")

	filing.write(intro+facts+' +++$+++ '+'decision' + forTheseReasons+'\n')
	filing.close

	# data = [start, intro , facts, forTheseReasons]
	# with open('output.tsv', 'w', newline='') as f_output:
	#     tsv_output = csv.writer(f_output, delimiter='\t')
	#     tsv_output.writerow(data)




# target_string = "Abraham Lincoln was born on February 12, 1809,"
# res = re.search(r'born', target_string)
# res1 = re.search(r'12', target_string)
# print(res1.group())
# # Output 1809
#
# # save start and end positions
#
# start = res1.start()
# end = res1.end()
# print(start,end)
# print(target_string[res.end()+1:res1.start()])
# print(target_string[start:end])
# # Output 1809
with open('ques.pkl', 'wb') as file1:
    pickle.dump(ques, file1)
with open('ans.pkl', 'wb') as file2:
    pickle.dump(ans, file2)
print(ques)
print('\n\n\n---------------\n\n\n')
print(ans)

