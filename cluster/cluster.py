import json
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering,AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

def loadData():
    Dict = []
    Data = []
    Label = []
    file = open("D:\\Datamining\\Tweets.txt",'r')
    for line in file.readlines():
        lines = json.loads(line)
        Dict.append(lines)
    for line in Dict:
        Data.append(line['text'])
        Label.append(line['cluster'])
    tfidf = TfidfTransformer().fit_transform(CountVectorizer(decode_error='ignore',stop_words='english').fit_transform(Data))
    Data = tfidf.toarray()
    return Data,Label

def Kmeans(Data,Label):
    n_cluster_number = len(set(Label))
    DataCluster = KMeans(n_clusters=n_cluster_number, random_state=10 ).fit(Data)
    printResult(DataCluster.labels_,Label)

def affinityPropagation(Data,Label):
    clustering = AffinityPropagation().fit(Data)
    printResult(clustering.labels_,Label)

def meanShift(Data,Label):
    clustering = MeanShift().fit(Data)
    printResult(clustering.labels_,Label)

def WardHierarchicalClustering(Data,Label):
    n_cluster_number = len(set(Label))
    clustering = AgglomerativeClustering(n_clusters=n_cluster_number,linkage='ward').fit(Data)
    printResult(clustering.labels_,Label)

def spectralClustering(Data,Label):
    n_cluster_number = len(set(DataLabels))
    DataCluster = SpectralClustering(n_clusters=n_cluster_number).fit(Data)
    printResult(DataCluster.labels_,Label)

def agglomerativeClustering(Data,Label):
    n_cluster_number = len(set(DataLabels))
    DataCluster = AgglomerativeClustering(n_clusters=n_cluster_number).fit(Data)
    printResult(DataCluster.labels_,Label)

def dBSCAN(Data,Label):
    clustering = DBSCAN(eps=1.13).fit(Data)
    printResult(clustering.labels_,Label)

def gaussianMixture(Data,Label):
    n_cluster_number = len(set(Label))
    GM = GaussianMixture(n_components=n_cluster_number,covariance_type='diag').fit(Data)
    clustering = GM.predict(Data)
    printResult(clustering,Label)

def printResult(DataCluster,DataLabels):
    print("NMI: %s" % (metrics.normalized_mutual_info_score(DataLabels, DataCluster)))
    
if __name__ == '__main__':
    Data, Label = loadData()
    
    Kmeans(Data,Label)
    affinityPropagation(Data,Label)
    meanShift(Data,Label)
    WardHierarchicalClustering(Data,Label)
    spectralClustering(Data,Label)
    agglomerativeClustering(Data,Label)
    dBSCAN(Data,Label)
    gaussianMixture(Data,Label)