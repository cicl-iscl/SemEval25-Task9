"""
    Description:
    This is the Data_preprocessing class for dealing with 2 parts: feature extraction and
    imbalanced data;
    I will use feature extraction methods to deal with the features such as day,
    month, country, text and so on. Since they belong to different types, there will
    be different methods to deal with it.
    For imbalanced data, I use the SMOTE method or ramdom sampling or weights manually calculated.

    Author: Weiting Wang
"""
import pandas as pd
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
import re


from hashlib import md5
import csv

label_encoder = LabelEncoder()
scaler = StandardScaler()
vectorizer = TfidfVectorizer()
encoder = OneHotEncoder()
pca = PCA(n_components=0.95)

class Datapreprocess:
    def __init__(self):
        """
        what attributes many need
        """
        self.features = None
        self.hazard_label = None
        self.product_label = None
        self.exact_hazard_label = None
        self.exact_product_label = None
        self.weights = {}

    def read_csv_file(self, file):
        """
        a method to read the file:
         first 6 columns are features
         last 4 columns are labels
        """
        data = pd.read_csv(file, index_col=0)
        self.features = data.iloc[:, : 6]
        self.hazard_label = data["hazard-category"]
        self.product_label = data["product-category"]
        self.exact_hazard_label = data["hazard"]
        self.exact_product_label = data["product"]

    def feature_extraction(self):
        """
        a helper method to extract the features
        """

        date_features = scaler.fit_transform(self.features[['year', 'month', 'day']])
        country_features = encoder.fit_transform(self.features[['country']])
        title_features = vectorizer.fit_transform(self.features['title'])
        text_features = vectorizer.fit_transform(self.features['text'])

        combined_features = hstack([
            csr_matrix(date_features),
            csr_matrix(country_features),
            title_features,
            text_features
        ])

        dense_features = combined_features.toarray()
        self.features = pca.fit_transform(dense_features)

    def label_encoder(self, label):
        """
        a method to deal with labels
        """
        label_encoder = encoder.fit_transform(label)
        return pca.fit_transform(label_encoder.toarray())


    def calculate_label_weights(self):
        """
        a method to deal with imbalanced class/label weights, it may be used in train
        """
        total_samples = len(self.features.index)
        labels = {
            "hazard_label": self.hazard_label,
            "product_label": self.product_label,
            "exact_hazard_label": self.exact_hazard_label,
            "exact_product_label": self.exact_product_label
        }

        for name, data in labels.items():
            label_counts = Counter(data)
            num_classes = len(label_counts)
            weights = {
                label: total_samples / (num_classes * count )
                for label, count in label_counts.items()
            }
            self.weights[name] = weights

    def remove_noise (self,text):
        """
        a method to remove unnecessary information in text (eg.34)
        such as dates or reduplicated information
        """
        #remove website link
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # remove the noise first
        seen = set()
        result = []
        for str in text.split("\n"):
            hash_val = md5(str.strip().encode()).hexdigest()
            if hash_val not in seen:
                seen.add(hash_val)
                result.append(str)
        return '\n'.join(result)


        # find the closet information for the target
    # text_clean: remove 1. duplicate  2.website, unnecessary noise
    # text_process: remove noises, choose the semantic closet info/keywords information

    # label and features ??



if __name__ == "__main__":
    # with open("Data/incidents_train.csv",'r') as f:
    #     csv_reader = csv.DictReader(f)
    #     line_count = 0
    #     for row in csv_reader:
    #         print(row)
    data = pd.read_csv("Data/train/incidents_train.csv", index_col=0, encoding='utf-8')
    #print(data.columns)
    # for str in data['text'].loc[1]:
    #     print(str)
        # print("----------------------")
        # print (str.split("\n"))
    # 获取第 100 行的 'text' 列的内容
    # 34 is duplicated
    print(data['text'].iloc[3])
    print("---------____________")
    test = data['text'].iloc[3]
    seen = set()
    result = []
    for line in test.split("\n"):
        hash_val = md5(line.strip().encode()).hexdigest()
        if hash_val not in seen:
            seen.add(hash_val)
            result.append(line)
    s = '\n'.join(result)
    print(s)
    print("---------____________")
    print(len(test))
    print(len(s))

    # the data with lots of \n and website and reduplicate things

