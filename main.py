import multiprocessing
import os
import random
import re
import tabulate
from stop_words import get_stop_words
from selenium import webdriver


def list_dir(b_path, reg):
    dir_path = b_path
    res = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    if reg == "n":
        return res
    else:
        p = re.compile(reg)
        return sorted(res, key=lambda s: int(p.search(s).group()))


def title_list(util):
    driver = webdriver.Chrome()
    driver.get(util[1])
    get_title = driver.title
    return util[0], util[1], get_title


def get_titles(url_s):
    util = []
    titles = []
    for idx, url in enumerate(url_s):
        util.append((idx, url))
    with multiprocessing.Pool() as pool:
        for result in pool.imap(title_list, util):
            titles.append(result)
    titles.sort(key=lambda a: a[0])
    return titles


def get_url_samples(sample_list, review_files):
    links = []
    with open("P3- Text mining/reviews_url.txt", "r") as file:
        for line in file:
            stripped_line = line.replace('/usercomments\n', "")
            links.append(stripped_line)
    index_reviews = []
    for sample_file in sample_list:
        for index, item in enumerate(review_files):
            if item == sample_file:
                index_reviews.append(index)

    url_list = []
    for selected_index in index_reviews:
        for index, item in enumerate(links):
            if index == selected_index:
                url_list.append(item)

    return url_list


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./,:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        data = data.replace(i, '')
    return data


def remove_stop_words(data):
    stop_words = list(get_stop_words('en'))
    new_text = ""
    for word in data.split(" "):
        if word not in stop_words:
            new_text = new_text + " " + word
    return new_text


def remove_apostrophe(data):
    return data.replace("'", " ")


def remove_single_characters(data):
    new_text = ""
    for w in data.split(" "):
        if len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def get_sample(list_reviews, reg):
    sample_reviews = random.choices(list_reviews, k=10)
    p = re.compile(reg)
    return sorted(sample_reviews, key=lambda s: int(p.search(s).group()))


def prepare_text(text):
    text = remove_punctuation(text)
    text = remove_stop_words(text)
    text = remove_apostrophe(text)
    text = remove_single_characters(text)
    return text


def prepare_sample_dict(sample_files, stop_words, excluded_words):
    result_list = []
    bag_of_words = []
    num_of_words = []

    for file in sample_files:
        with open("P3- Text mining/reviews/" + file) as f:
            text_in_file = str(f.readlines()).lower()

        text_in_file = prepare_text(text_in_file)
        bag_of_words.append(text_in_file.split(" "))
    unique_words = set("")
    for bag in bag_of_words:
        unique_words = unique_words.union(set(bag))
    unique_words.remove("")
    # print(unique_words)

    for i in range(10):
        i = dict.fromkeys(unique_words, 0)
        num_of_words.append(i)
    # print(num_of_words)

    for idx, bag in enumerate(bag_of_words):
        # item = dict.fromkeys(unique_words, 0)
        for word in bag:
            if word not in stop_words and word != "" and word not in excluded_words and "'" not in word:
                num_of_words[idx][word] += 1
        tuples_list = (sorted(num_of_words[idx].items(), key=lambda x: x[1], reverse=True))[0:10]
        # print(tuples_list)
        result = dict(tuples_list)
        result_list.append(result)
        # num_of_words.append(item)
        # item = {}

    return num_of_words, bag_of_words, result_list, unique_words


def compute_tf(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def compute_tf_un_normalized(wordDict):
    tfDict = {}
    for word, count in wordDict.items():
        tfDict[word] = count
    return tfDict


def compute_idf(documents):
    import math
    N = len(documents)
    dfDict = dict.fromkeys(documents[0].keys(), 0)
    idfDict = dict.fromkeys(documents[0].keys(), 0)

    for document in documents:
        for word, val in document.items():
            if val > 0:
                dfDict[word] += 1
                idfDict[word] += 1
    for word, val in idfDict.items():
        if val != 0:
            idfDict[word] = math.log(N / float(val))
    return dfDict, idfDict


def compute_tf_idf(tfBagOfWords, id_fs):
    tf_idf_list = []
    for bag in tfBagOfWords:
        tf_idf = {}
        for word, val in bag.items():
            tf_idf[word] = val * id_fs[word]
        tf_idf_list.append(tf_idf)
    return tf_idf_list


def text_mining():
    stop_words = list(get_stop_words('en'))
    excluded_words = ["von", "lmn", "can", "even", "br", "isn", "else", "anyone", "semi", "sat",
                      "feud", "jr'", "ll", "ve", "re", "fi", "mr", "film", "movie", "production", "ever", "actually",
                      "whole", "movies", "films", "seen", "video", "like", "us", "tcm", "show", "get"]
    reg = "^[0-9]*"
    list_reviews = list_dir("P3- Text mining/reviews", reg)
    sample = get_sample(list_reviews, reg)
    print(sample)
    sample_url_list = get_url_samples(sample, list_reviews)
    print(sample_url_list)
    titles_list = get_titles(sample_url_list)
    for title in titles_list:
        print(title)
    numOfWords, bagOfWords, mostUsed, uniqueWords = prepare_sample_dict(sample, stop_words, excluded_words)

    # Un - Normalized
    tf_results_o = []
    for idx, numWords in enumerate(numOfWords):
        tf_item = compute_tf_un_normalized(numWords)
        tf_results_o.append(tf_item)
    # list of dict tf computed
    print("\nTOP 10 - Most Used Words per Document")
    for most_list in mostUsed:
        print(most_list)
    header = list(tf_results_o[0].keys())
    rows = [x.values() for x in tf_results_o]
    print("\nTF List")
    print(tabulate.tabulate(rows, header, showindex="always"))
    # print(numOfWords)
    df_results_o, idf_results_o = compute_idf(numOfWords)
    preProcess = [df_results_o, idf_results_o]
    header = list(preProcess[0].keys())
    rows = [x.values() for x in preProcess]
    print("\nDF(index row 0) and IDF Lists (index row 1)")
    print(tabulate.tabulate(rows, header, showindex="always"))

    tf_idf_results_o = compute_tf_idf(tf_results_o, idf_results_o)
    header = list(tf_idf_results_o[0].keys())
    rows = [x.values() for x in tf_idf_results_o]
    print("\nTF-IDF Results")
    print(tabulate.tabulate(rows, header, showindex="always"))

    print("\n\n-------------------------------------------------------------------------------------------------\n\n")
    # Normalized
    tf_results = []
    for idx, numWords in enumerate(numOfWords):
        tf_item = compute_tf(numWords, bagOfWords[idx])
        tf_results.append(tf_item)
    # list of dict tf computed
    # print("\n10 Most Used Words per Document")
    # for most_list in mostUsed:
    #     print(most_list)
    header = list(tf_results[0].keys())
    rows = [x.values() for x in tf_results]
    print("\nTF List Normalized")
    print(tabulate.tabulate(rows, header, showindex="always"))
    df_results, idf_results = compute_idf(numOfWords)
    preProcess = [df_results, idf_results]
    header = list(preProcess[0].keys())
    rows = [x.values() for x in preProcess]
    print("\nDF(index row 0) and IDF Lists (index row 1)")
    print(tabulate.tabulate(rows, header, showindex="always"))

    tf_idf_results = compute_tf_idf(tf_results, idf_results)
    header = list(tf_idf_results[0].keys())
    rows = [x.values() for x in tf_idf_results]
    print("\nTF-IDF Normalized Results")
    print(tabulate.tabulate(rows, header, showindex="always"))
    query = ""

    x = input("Use default query? y/n: ")
    if x == "y":
        query = "good way lead long interesting good good way way way, complete, bit explain great laugh laugh laugh " \
                "appropriate interested expect"
    else:
        x = input("Type a query\n")
        query = x

    query = prepare_text(query)
    queryDict = dict.fromkeys(uniqueWords, 0)
    queryBag = query.split(" ")
    for word in queryBag:
        if word in queryDict and word not in stop_words and word != "" and word not in \
                excluded_words and "'" not in word:
            queryDict[word] += 1
    tf_query_result = compute_tf(queryDict, queryBag)
    tf_idf_query_result = compute_tf_idf([tf_query_result], idf_results)

    header = list(tf_idf_query_result[0].keys())
    rows = [x.values() for x in tf_idf_query_result]
    print("\nTF-IDF QUERY Result")
    print(tabulate.tabulate(rows, header, showindex="always"))

    productList = []
    for idx, docDict in enumerate(tf_idf_results):
        product = 0
        for word in docDict:
            product += (docDict[word] * tf_idf_query_result[0][word])
        productList.append((product, sample[idx], sample_url_list[idx], titles_list[idx][2]))

    productList.sort(key=lambda a: a[0], reverse=True)

    header = ["similarity", "review", "url_reference", "product_title"]
    rows = [x for x in productList]
    print("\nOrdered Query Results")
    print(tabulate.tabulate(rows, header, showindex="always"))

    productOfListProducts = []
    bestProductOfListProducts = []
    for idx, docDict in enumerate(tf_idf_results):
        bestCoincidence = []
        for idy, innerDoc in enumerate(tf_idf_results):
            product = 0
            if idx != idy:
                for word in docDict:
                    product += (docDict[word] * innerDoc[word])
                bestCoincidence.append((product, sample[idx],
                                        sample_url_list[idx], sample[idy], sample_url_list[idy],
                                        titles_list[idx][2], titles_list[idy][2]
                                        ))
        bestCoincidence.sort(key=lambda a: a[0], reverse=True)
        productOfListProducts.append(bestCoincidence)
        bestProductOfListProducts.append(bestCoincidence[0])

    header = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    rows = [x for x in productOfListProducts]
    print("\nOperations similarity between sample files")
    print(tabulate.tabulate(rows, header, showindex="always"))

    header = ["similarity", "review Origin",
              "url_reference Origin", "review Final", "url_reference Final", "product_title Origin",
              "product_title Final"
              ]
    rows = [x for x in bestProductOfListProducts]
    print("\nBest similarity between sample files")
    print(tabulate.tabulate(rows, header, showindex="always"))


if __name__ == '__main__':
    text_mining()
