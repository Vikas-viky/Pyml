import csv 
with open("data.csv") as f: 
 csv_file=csv.reader(f) 
 data=list(csv_file) 
 s=data[1][:-1] #1st record is taken as specific hypothesis 
 g=[['?' for i in range(len(s))] for j in range(len(s))] #? is put for all the records 
 
 for i in data: 
 if i[-1]=="Yes": # if target attribute is yes, then it should match specific hypothesis 
 for j in range(len(s)): 
 if i[j]!=s[j]: # if record doesn't match specific hypothesis, put ? in specific columns 
 s[j]='?' 
 g[j][j]='?' 
 elif i[-1]=="No": # if target attribute is no, then specific hypothesis should not match 
 for j in range(len(s)): 
 if i[j]!=s[j]: 
 g[j][j]=s[j] 
 else: 
 g[j][j]="?" 
 print("\nSteps of Candidate Elimination Algorithm",data.index(i)+1) 
 print(s) 
 print(g) 
 gh=[] 
 for i in g: 
 for j in i: 
 if j!='?': 
 gh.append(i) 
 break 
 print("\nFinal specific hypothesis:\n",s) 
 print("\nFinal general hypothesis:\n",gh)