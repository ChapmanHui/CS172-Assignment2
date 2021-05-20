import sys
import re
import os
import math
import zipfile

#============================================================
#Document class to store all document information
class Document:
    def __init__(self, doc_name, total_terms, unique_terms):
        self.doc_name = str(doc_name)
        self.total_terms = int(total_terms)
        self.unique_terms = int(unique_terms)
        self.words = set()
    def __repr__(self):
        return "Doc: " + self.doc_name + " Total Terms: " + str(self.total_terms) + " Unique Terms: " + str(self.unique_terms)
    def __str__(self):
        return "Doc: " + self.doc_name + " Total Terms: " + str(self.total_terms) + " Unique Terms: " + str(self.unique_terms)

#============================================================
#Functions
#Dot product
def dot_product(list1, list2):
    result = 0
    for i in range(len(list1)):
        result = result + (list1[i] * list2[i])
    return result

def mag_of_vector(l):
    result = 0
    for i in l:
        result += i * i
    return math.sqrt(result)

def cosine_sim(l1, l2):
    dot_prod = dot_product(l1, l2)
    mag = mag_of_vector(l1) * mag_of_vector(l2)
    return dot_prod/mag

#==============================================================
#Read in collection
with open("stopwords.txt",'r') as f:
    stopwords = [line.rstrip() for line in f]
punctuation = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'",'`', '_']

#Contains the mapping of all term and document ids to their tokens
termInfo = {}
docIds = {}

# Regular expressions to extract data from the corpus
doc_regex = re.compile("<DOC>.*?</DOC>", re.DOTALL)
docno_regex = re.compile("<DOCNO>.*?</DOCNO>")
text_regex = re.compile("<TEXT>.*?</TEXT>", re.DOTALL)

with zipfile.ZipFile("data/ap89_collection_small.zip", 'r') as zip_ref:
    zip_ref.extractall()
   
# Retrieve the names of all files to be indexed in folder ./ap89_collection_small of the current directory
for dir_path, dir_names, file_names in os.walk("ap89_collection_small"):
    allfiles = [os.path.join(dir_path, filename).replace("\\", "/") for filename in file_names if (filename != "readme" and filename != ".DS_Store")]
#==============================================================

for file in allfiles:
    with open(file, 'r', encoding='ISO-8859-1') as f:
        filedata = f.read()
        result = re.findall(doc_regex, filedata)  # Match the <DOC> tags and fetch documents

        for document in result[0:]:
            # Retrieve contents of DOCNO tag
            docno = re.findall(docno_regex, document)[0].replace("<DOCNO>", "").replace("</DOCNO>", "").strip()
            # Retrieve contents of TEXT tag
            text = "".join(re.findall(text_regex, document))\
                      .replace("<TEXT>", "").replace("</TEXT>", "")\
                      .replace("\n", " ")
            # step 1 - lower-case words, remove punctuation, etc.
            for punc in punctuation:
                text = text.replace(punc, "")
            text = text.lower().split()
            docIds[docno] = Document(docno, len(text), 0)
            current_position = 1
            #step 2 - create tokens ignoring stop-words
            #step 3 - build index
            for word in text:
                if word in stopwords:
                    current_position +=1
                else:
                    if (termInfo.get(word)):
                        if(termInfo[word].get(docno)):
                            termInfo[word][docno].append(current_position)
                            docIds[docno].words.add(word)
                        else:
                            termInfo[word][docno] = []
                            termInfo[word][docno] = [current_position]
                            docIds[docno].unique_terms += 1
                            docIds[docno].words.add(word)
                        current_position += 1
                    else:
                        termInfo[word] = {}
                        termInfo[word][docno] = []
                        termInfo[word][docno] = [current_position]
                        current_position += 1
                        docIds[docno].unique_terms += 1
                        docIds[docno].words.add(word)

#======================================================================================
arguments = len(sys.argv)

try:
    query_file = sys.argv[1]
    output_file = sys.argv[2]
except:
    raise Exception("python ./VSM query_file output_file")

#Read in all queries
try:
    with open(query_file, 'r') as q:
        queries = [line.strip() for line in q]
except:
    raise Exception("Failed to open query file")

try:
    output = open(output_file, "w")
except:
    raise Exception("Failed to open output file")

vector_space = set()
relevant_docs = set()
query_set = set()
doc_set = set()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for q in queries:
    for punc in punctuation:
        q = q.replace(punc, "")
    q = q.lower().split()

    query_num = q[0]
    q.pop(0)

    #Gather all relevant documents
    query_set = set()
    relevant_docs = set()
    for term in q:
        if term in stopwords:
            continue
        else:
            query_set.add(term)
            try:
                for doc in termInfo[term]: relevant_docs.add(doc)
            except:
                continue

    #Check relevance for each document
    ranking = []
    for doc in relevant_docs:
    #Union set between query and doc
        for word in docIds[doc].words:
            vector_space.add(word)
        
        query_tf = []
        doc_tf = []
        vector_space = vector_space.union(query_set)
        #Adding binary weights to query and document sets
        for i in vector_space:
            if i in query_set:
                query_tf.append(1)
            else:
                query_tf.append(0)

            if i in docIds[doc].words:
                doc_tf.append(1)
            else:
                doc_tf.append(0)
        #Calculate Cosine Sim using query and doc weights        
        doc_space = [cosine_sim(doc_tf,query_tf), str(doc)]
        ranking.append(doc_space)
        vector_space.clear()
    query_set.clear()
    relevant_docs.clear()

    #Sort ranking in decending order and print output
    ranking.sort(reverse = True)
    for i in range(10):
        output.write(str(query_num) + " Q0 " + str(ranking[i][1]) + " " + str(i + 1) + " " + str(ranking[i][0]) + " Exp\n")
        #print(str(query_num) + " Q0 " + str(ranking[i][1]) + " " + str(i) + " " + str(ranking[i][0]) + " Exp")
