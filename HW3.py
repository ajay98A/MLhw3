import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import scipy 
import sklearn
from collections import Counter
from sklearn.metrics import multilabel_confusion_matrix
from scipy import spatial
from sklearn.model_selection import train_test_split
read_data = pd.read_csv('kmeans_data_2/data.csv')
read_labels = pd.read_csv('kmeans_data_2/label.csv',names=['label'],header=None)
read_data.info()
train_data, test_data = train_test_split( read_data, test_size=0.08, random_state=42)
train_labels, test_labels = train_test_split( read_labels, test_size=0.08, random_state=42)
class Kmeans_algo:
    
    def sseCalculation(self, centr_value_dict, centr_dict,read_data):
        sse_data = 0
        for i in centr_dict:
            sse_cluster = 0
            # np.sum()
            for j in centr_dict[i]:
                d = list(read_data.iloc[int(j)])
                for a,b in zip(centr_value_dict[i],d):
                    sse_cluster += (a-b)**2
            sse_data+=sse_cluster
        return sse_data    
    
    def initializingCent(self,read_data,K):
        m = read_data.shape[0]
        centr_value_dict={}
        for i in range(K):
            r = np.random.randint(0, m-1)
            centr_value_dict[i] = read_data.iloc[r]
        return centr_value_dict
    
    def jaccardSimilarity(self,centr, d):
        intersection = len(list(set(centr).intersection(d)))
        union = (len(set(centr)) + len(set(d))) - intersection
        return float(intersection) / union

    def trainKmeans(self,read_data,K,max_iter=20,mode=1,tol=10):
        centr_value_dict = self.initializingCent(read_data,K)
        new_centr_value_dict = {}
        count = 0
        centr_dict = {}
        convergence = False
        while((count<max_iter) and not convergence):
            
            for i in list(centr_value_dict.keys()):
                centr_dict[i]=[]
            for i in range(read_data.shape[0]):
                x = read_data.iloc[i]
                if mode==1 :
                    distance_measure = [np.linalg.norm(x-centr_value_dict[j])  for j in centr_value_dict]
                    idx = np.argmin(distance_measure)
                    centr_dict[idx].append(i)
                elif mode==2 :
                    distance_measure = [self.jaccardSimilarity(list(x),centr_value_dict[j]) for j in centr_value_dict]
                    idx = np.argmax(distance_measure)
                    centr_dict[idx].append(i)
                elif mode==3 :
                    distance_measure = [1-scipy.spatial.distance.cosine(x,list(centr_value_dict[j]))  for j in centr_value_dict]
                    idx = np.argmax(distance_measure)
                    centr_dict[idx].append(i)
                
                prev_centr=dict(centr_value_dict)
                
            
            for i in centr_dict:
                if len(centr_dict[i]):
                    dps_centr = centr_dict[i]
                    centr_value_dict[i] = np.average(read_data.iloc[dps_centr],axis=0)
            
            
            current_tol=-1
            for i in centr_value_dict:
                prev_centr_point = prev_centr[i]
                new_centr_point = centr_value_dict[i]
                change = np.sum(np.absolute(new_centr_point-prev_centr_point))
                current_tol = max(change, current_tol)
            print("SSE value for iteration ",count,": ", self.sseCalculation(centr_value_dict, centr_dict, read_data)) 
            print("Tolerance for the Iteration ",count,": ",current_tol)
            
            count+=1
            if (current_tol<10):
                convergence = True
                break
           # print("KMeans Iteration",count)
        return centr_value_dict,centr_dict
def predict_cluster_labels(C, S, labels):
    cluster_labels = np.zeros(10,dtype=int)
    for c in C:
        labels_of_points = []
        for point in S[c]:
            labels_of_points.extend(labels.iloc[point])
        counter = Counter(labels_of_points)
        try:
            cluster_labels[c] = max(counter, key=counter.get)
        except:
            cluster_labels[c] = np.random.randint(0,9)
    return cluster_labels
def accuracy(centroids, centroid_Labels, test_data, true_labels, mode=1):
    y_true = list(true_labels['label']);
    y_pred = []
    for index in range(test_data.shape[0]):
        featureset = test_data.iloc[index]
        if mode==1:
            distances = [np.linalg.norm(featureset - centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            y_pred.append(centroid_Labels[classification])
        elif mode==2:
            similarity = [jaccardSimilarity(featureset, centroids[centroid]) for centroid in centroids]
            classification = similarity.index(max(similarity))
            y_pred.append(centroid_Labels[classification]) 
        elif mode==3:
            similarity = [1 - spatial.distance.cosine(featureset, centroids[centroid]) for centroid in centroids]
            classification = similarity.index(max(similarity))
            y_pred.append(centroid_Labels[classification])
    denominator = test_data.shape[0]
    correctly_classified = 0
    for i in range(0,len(y_pred)):
        if y_true[i] == y_pred[i]:
            correctly_classified += 1
    accuracy = correctly_classified/denominator
    return accuracy
kmeans_model = Kmeans_algo()
centroids_euclidian,clusters_euclidian = kmeans_model.trainKmeans(read_data,10, max_iter=100,mode=1)
SSE_Euclidean = kmeans_model.sseCalculation(centroids_euclidian,clusters_euclidian,read_data)
print("SSE value for Euclidean = ",SSE_Euclidean)
cluster_labels_euclidian = predict_cluster_labels(centroids_euclidian,clusters_euclidian,read_labels)
cluster_labels_euclidian
Euclidean_accuracy = accuracy(centroids_euclidian, cluster_labels_euclidian,test_data,test_labels)
Euclidean_accuracy
centroids_jaccard,clusters_jaccard = kmeans_model.trainKmeans(read_data,10, max_iter=100,mode=2)
SSE_jaccard = kmeans_model.sseCalculation(centroids_jaccard,clusters_jaccard,read_data)
print("SSE value for Jaccard  = ",SSE_jaccard)
cluster_labels_jaccard = predict_cluster_labels(centroids_jaccard,clusters_jaccard,read_labels)
cluster_labels_jaccard
Accuracy_Jaccard = accuracy(centroids_jaccard, cluster_labels_jaccard,test_data,test_labels)
Accuracy_Jaccard
centroids_cossine,clusters_cossine = kmeans_model.trainKmeans(read_data,10, max_iter=100,mode=3)
SSE_cossine = kmeans_model.sseCalculation(centroids_cossine,clusters_cossine,read_data)
print("SSE value for cossine  = ",SSE_cossine)
cluster_labels_cossine = predict_cluster_labels(centroids_cossine,clusters_cossine,read_labels)
cluster_labels_cossine
Accuracy_cossine = accuracy(centroids_cossine, cluster_labels_cossine,test_data,test_labels)
Accuracy_cossine
print("Euclidean accuracy = ",Euclidean_accuracy)
print("Jacard accuracy = ",Accuracy_Jaccard)
print("Cosine accuracy = ",Accuracy_cossine)
print("SSE value for Euclidean = ",SSE_Euclidean)
print("SSE value for Jaccard = ",SSE_jaccard)
print("SSE value for cossine  = ",SSE_cossine)