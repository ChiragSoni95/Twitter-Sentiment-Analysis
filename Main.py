import pandas
import re
import math
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
count_vect = CountVectorizer()
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support
from nltk.stem import PorterStemmer
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier as XGBoostClassifier
import string
from string import punctuation
 
ps = PorterStemmer()
df = pandas.read_excel('/home/sreeharsha/Downloads/training-Obama-Romney-tweets.xlsx',sheet_name ="Obama",keep_default_na = False)

#dt = pandas.read_excel('/home/sreeharsha/Downloads/testing-Obama-Romney-tweets-spring-2013 (1).xlsx',sheet_name ="Romney",keep_default_na = False)





FORMAT = ['Anootated tweet','Class']

tweets = df['Anootated tweet'] 
Classes = df['Class']

#t = dt['Anootated tweet'] 
#c = dt['Class']
abbr_dict = {}


def readAbbrFile(abbrFile):
    global abbr_dict

    f = open( abbrFile )
    lines = f.readlines()
    f.close()
    for i in lines:
        tmp = i.split( '|' )
        abbr_dict[tmp[0]] = tmp[1]

    return abbr_dict



def convertCamelCase(word):
    return re.sub("([a-z])([A-Z])","\g<1> \g<2>",word)
#end



def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def removeHash(s):
    return re.sub(r'#([^\s]+)', r'\1', s)


def cleanhtml(raw_html):
    cleanr = re.compile('<a>|</a>|<e>|</e>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def isNan(num):
    return num != num


def replaceAbbr(s):
    for word in s:
        if word.lower() in abbr_dict.keys():
            s = [abbr_dict[word.lower()] if word.lower() in abbr_dict.keys() else word for word in s]
    return s

stopwords = {}


with open("/home/sreeharsha/Downloads/stop_words.txt", "r") as f:
    lines = f.read().splitlines()
for line in lines:
    stopwords[line] = 1

tokenized_words_list=[]
i=0
model = []
mylist= []
classes = []
abbrfile = "/home/sreeharsha/Downloads/abbr"
abbr_dict = readAbbrFile(abbrfile)

def pre_process(l):
	l = cleanhtml ( l )
	l = replaceAbbr ( l )
	l = removeHash ( l )
	l = re.sub ( '((www\.[\s]+)|(https?://[^\s]+))', '', l )
	l = re.sub ( r'\\[xa-z0-9.*]+', '', l )
	l = convertCamelCase ( l )
	l = replaceTwoOrMore ( l )

	l = re.sub ( r'^RT[\s]+', '', l, flags=re.MULTILINE )  # removes RT
	#removing white spaces

	# Removing usernames
	l = re.sub ( '@[^\s]+', '', l )

	# Removing words that start with a number or a special character
	l = re.sub ( r"^[^a-zA-Z]+", ' ', l )

	l = re.sub ( '[\s]+', ' ', l )



	# Removing words that end with digits
	l = re.sub ( r'\d+', '', l )

	# Replace the hex code "\xe2\x80\x99" with single quote
	l = re.sub ( r'\\xe2\\x80\\x99', "'", l )

	# Removing punctuation
	exclude=set(string.punctuation)
	l = ''.join ( ch for ch in l if ch not in exclude )

	# Remove trailing spaces and full stops
	l = l.strip ( ' .' )

	# Convert everything to lower characters
	l = l.lower ()
	
	tokens = re.split(' ',l)
	
	temp = []
	temp_tweet = ""
	for t in tokens:
		if(t!='' and t.lower() not in stopwords):
			ps.stem(t)
			temp.append(t)
			temp_tweet=temp_tweet+t+" "
	return temp_tweet
	

for l,m in zip(tweets,Classes):
	if(isNan(l)):
		continue
	if(type(m)!=int or not(m==0 or  m==1 or m==-1) ):
		continue
	temp_tweet = pre_process(l)
	classes.append(m)
	mylist.append(temp_tweet) 
left = np.array(mylist)
right = np.array(classes)

test = []
classes1=[]
'''
for l,m in zip(t,c):
	if(isNan(l)):
		continue
	if(type(m)!=int or not(m==0 or  m==1 or m==-1) ):
		continue
	temp_tweet = pre_process(l)
	classes1.append(m)
	test.append(temp_tweet)
test_left = np.array(test)
test_right = np.array(classes1)
#print (test_left)
#print len(classes1)
'''
linearSVM= LinearSVC( random_state=666, class_weight="balanced", max_iter=5000,  C=2.0,tol=0.001, dual=True )
linearSVM_SVC= SVC( C=1, kernel="rbf", tol=1, random_state=0,gamma=1 )
logistic = LogisticRegression( fit_intercept=True,class_weight="balanced", n_jobs=-1, C=1.0,
                                   max_iter=200 )
rand_forest = RandomForestClassifier( n_estimators=403, random_state=666, max_depth=73, n_jobs=-1 )
bc=BaggingClassifier( base_estimator=logistic, n_estimators=403, n_jobs=-1, random_state=666,
                                                max_features=410)

ensemble_voting=VotingClassifier([("svm",linearSVM_SVC),("logistic",logistic),("rand_forest",rand_forest),("sgdc",SGDClassifier())],weights=[1,2,1,1])
boost = AdaBoostClassifier(base_estimator=logistic)
xgboost= XGBoostClassifier( n_estimators=103, seed=666, max_depth=4, objective="multi:softmax" )
X_train,X_test,Y_train,Y_test=train_test_split(left,right,test_size=0.1)
		
text_clf = Pipeline([('vect', CountVectorizer()),
			         ('tfidf', TfidfTransformer()),
			         ('clf-svm',ensemble_voting)])
text_clf = text_clf.fit(X_train,Y_train)
predicted = text_clf.predict(X_test)
print classification_report(Y_test,predicted)


    


    
    
    


