import re
import csv
import os
import pandas as pd
 


folder_path = 'C:/Users/awang/Downloads/Earnings-Calls-NLP-main/Earnings-Calls-NLP-main/transcripts/sandp500'

transcripts_results = []
transcripts_results = []
questions = []
answers = []
transcript = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as file:
            text = file.readlines()

            operator_flag = False
            question_flag = False
            question_asker = ""
            question = ""
            
            for i in range(1, len(text)):
                line = text[i].strip()

                if line.strip() == "Operator" and "question" in text[i+1].strip():
                    operator_flag = True

                if "?" in line and operator_flag:
                    question_asker = text[i-1].strip().split()
                    print(question_asker)
                    if len(question_asker) > 1:
                        if len(question_asker[1]) > 1:
                            if question_asker[1][0].isupper or question_asker[1][1] == '.':
                                question_flag = True
                                operator_flag = False
                                #print(question_asker)
                                question = line.strip()
                
                words = line.split()
                if question_flag and len(words) > 1 and len(question_asker) > 1:
                    if len(words[1]) > 1 and len(question_asker[1]) > 1:
                        if len(words) < 10 and (words[1][0].isupper() or (len(words[1]) == 2 and words[1][1] == ".")) and (words[0] != question_asker[0] and words[1] != question_asker[1]):
                            question_flag = False
                            questions.append(question)
                            answers.append(text[i+1].strip())
                            transcript.append(filename)

print(len(questions))
print(len(answers))

for i in range(1, len(answers)):
    print("question: ")
    print(questions[i])
    print("\n")
    print("answer: ")
    print(answers[i])
    print("\n")

print(answers)
print(len(answers))
df = pd.DataFrame({'Questions': questions, 'Answers': answers, 'Transcript File': transcript})

df.to_csv('C:/Users/awang/theil_fnp-20/question-answer.csv')

