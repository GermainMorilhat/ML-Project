# ML-Project
This repository presents our Machine Learning project. The group of work is composed by:

<ul>
  <li> Alaeddine Ben Fradj</li>
    <li> Germain Morilhat</li>
    <li> Jules Saïdane</li>


  </ul>
  
This project aims to apply Binary Classifcation on two different datasets: 
<ul>
  <li> <b> Banknote Authentication Dataset </b> : https://archive.ics.uci.edu/ml/datasets/banknote+authentication <br>
    This dataset is composed by 1372 instances, with 5 attributes each, and it gives as an output if the Banknote is falsified or not. The purpose will be to train our classification model, to be able to determine if a given banknote is valid or not.
  </li>
  <li> <b> Chronic Kidney Disease </b> : https://www.kaggle.com/mansoordaku/ckdisease <br>
    This dataset is composed by 400 instances, with 24 attributes each, such as patient's age or red blood cell count and gives as an output if a given patient is suffering from chronic kidney disease or not. We will train our classification model to predict if a new patient is likely to suffer from chronic kidney disease.
  </li>
</ul>

<h2>I - Good programming practices </h2>

<p>
Before we start present our work, we think that it is important to deal with the programming practices that we will use through the project. One the main principle that we will follow will be the use of Git to share and synchronize our work. This will be used to a lesser extent on the code part, since we worked a lot together, and it was therefore easier to work on a single computer.<br> Yet Git was very useful, especially for bug fixes after most of the work was done together, or when writing this report, which is an exercise that allows to separate the work into several parts, which makes the use of a synchronization tool like Git ideal.
</p>
<br>
<p>
Another thing we would like to focus on is the automation of our script, with the objective that the only action of a user trying to use it would be to change the database. For this, we will try to automate the different steps of the project, whether it is during the preprocessing with the cleaning of the data, or afterwards during the training and execution of the model.<br> To prevent bugs from appearing and to facilitate the maintenance of our code, we will also try to separate our code into multiple simple functions, allowing us to avoid a large number of incompatibilities between functions and to reduce the entropy of our script.
 
  We have also tried to comment as much as possible on our functions so that we can work on different functions and provide effective help. Regular testing allowed us to make rapid progress and to understand errors more quickly. We solved the problems with this method for pre-processing functions.
  In the end, the group work allowed to take a step back on the work of each one and to bring a solution. The fact of working together allowed us to be faster and more efficient. 

</p>

<h2> II - Development steps </h2>

<h3> a) Preprocessing</h3>

The first thing that we have to care about is the errors that could come from missing or corrupted values in the dataset. In order to deal with the issue, we have implemented a <b> Preprocessing function </b> which takes as an input the csv file and other parameters which describe it, like character that separate the column. This preprocessing function deals with missing values by replacing them with the mean of other instance on this attribute. <br>
<br>
In order to reduce the complexity and the computation time of the problem, we also use technics to try to reduce the size of the dataset. By computing the correlation matrix of the data columns, we are able to determine which columns are closely linked. If two columns are correlated at more than 75%, we keep only one of the both, ensuring us to reduce the size of the problem.
<br>
![image](https://github.com/GermainMorilhat/ML-Project/blob/main/correlation_matrix_banknote.png) ![image](https://github.com/GermainMorilhat/ML-Project/blob/main/correlation_matrix_kidney.png)



<br> A possible upgrade of the preprocessing function would be to implement a <b>Principal Component Analysis (PCA)</b> to identify more efficiently the truly relevant elements of the dataset.

<h3> b) Evaluation of the results </h3>


  In order to evaluate the accuracy of our result, and compare the different methods we implement a N-cross-validation function. Thus, we will be able to have, for a dataset, N different repartition of the training and test sets, which will allow us to have a representative information of the performance of a method.

<h3> c) Methods used </h3>
  We chose to use the scikit learn library to implement our classification models. We will implement several models such as :

  <ul>
  <li> SVM and Kernel methods : We will use several kernel such as linear kernel, polynomail kernel, Radial basis function kernel, sigmoid </li>
    <li> Stochastic Gradient Descent</li>
    <li> Decision Trees</li>
    <li> Bayesien classifcation</li>
    <li> Random Forest</li>
    <li> Neural network :  We have a neural network composed of 2 layers. One with 5 neurons and the other with 2 neurons. </li>
    <li> Logistic regression :    Logisitic regression allows to make a classification with an optimization of the "delta" parameter. </li>
  </ul>
   

<h2> III - Results </h2>

<h3> a) Banknote Authentication Dataset </h3>

We notice that every model is quite accurate on that dataset, even the least accurate model which is the SVM with a degre 2 polynome. The best model on that dataset, is the SVM with a Radial basis function kernel. We also notice that the Decision trees, Random Forest and Neural Network are efficient model to compute this type of problem. These performance could even be enhance by the study of the optimal depth of the Trees, or the structure of the Neural Network.

![image](https://github.com/GermainMorilhat/ML-Project/blob/main/cross_validation_banknote.png)
<br>
<h3> b) Chronic Kidney Disease </h3>

For this dataset, the results are more balanced. As in the case of the Banknote dataset, the Bayesian Clasifier is one of the least performing model. Another time, the SVM with a RBF kernel is the best performer. This type of kernel is really adapted to this kind problem.
![image](https://github.com/GermainMorilhat/ML-Project/blob/main/cross_validation_kidney.png)

## :grey_question: How does it work ?

```bash
# Clone this project
$ git clone https://github.com/GermainMorilhat/ML-Project

```

# Appendix : 

## Preprocessing
```bash
  Libraries
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import random as rnd
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  
  def preprocessing(csv_name, class_name, idx, sep): # made by Jules Saïdane and Germain Morilhat
    csv_name is data name
    class_name represents the data's class_name for classification
    sep is the data separator 
    idx is a arbitrary row number
    
    # First, we need to load the data
    data = pd.read_csv(csv_name,sep=sep)
    N = len(data[class_name])
    
    # Separating labels from data
    # If labels are str type, we change them into 1 or 0
    if isinstance(data[class_name][idx], str):
        labels = np.multiply([data[class_name]==data[class_name][0]],1)[0] #data[class_name][0] represents a 1
    else:
        labels = data[class_name]
    data.drop(class_name, 1, inplace=True) # We drop labels from data in order to get vectors values
    
    # Getting vectors values
    for c in data:
        if isinstance(data[c][idx], str):
            new_col = np.multiply([data[c] == data[c][idx]],1)
            data.drop(c, 1, inplace=True)
            data[c] = new_col[0]                
        data[c] = np.nan_to_num(data[c], copy=True, nan=data[c].median())
        
    # Using correlation matrix to get rid of some columns
    correlation = data.corr().abs()

    f = plt.figure(figsize=(19, 15))
    plt.matshow(correlation, fignum=f.number)
    plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    
    correlated_values=np.where(correlation>0.75)
    correlated_values=np.array([(correlation.columns[x],correlation.columns[y]) for x,y in zip(*correlated_values) if x!=y and x<y])
    
    for i in range(len(correlated_values)):
        c = correlated_values[i][0]
        data.drop(c, 1, inplace=True)
    
    # Finally
    X = data
    X = StandardScaler().fit_transform(X) #normalize our data
    # Creation of a dataset to train and another to test
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=42)
    return  x_train, x_test, y_train, y_test, X, labels
  

```

## Cross-validation 

```bash
# Models import list
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import SGDClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# made by Jules Saïdane and Alaeddine Ben Fradj
def  cross_validation_comparaison(X, Y, N): #N is the number of validation that we will compute

      # Models dictionnary

    models = {svm.SVC(kernel='linear'):'SVM with linear kernel',
              svm.SVC(kernel='poly', degree=2, gamma='auto'):'SVM with polynomial (degree 2) kernel',
              svm.SVC(kernel='rbf', gamma='auto'):'SVM with RBF kernel',
              svm.SVC(kernel='sigmoid', gamma=1./150):'SVM with sigmoid kernel',
              SGDClassifier():'Stochastic Gradient Descent',
              DecisionTreeClassifier():'Decision Trees',
              GaussianNB():'Bayesian classifier',
              RandomForestClassifier(n_estimators = 100, random_state = 42):'Random Forest',
              MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=42,max_iter=1000):'Neural Network',
              LogisticRegression(random_state=0):'Logistic Regression'}
    
    # Models score list with N_cross-validation.
    scores=[]
    scores = [np.mean(cross_val_score(clf, X, Y, cv=N)) for clf in list(models.keys())]
    print(list(models.values())[scores.index(max(scores))], max(scores))

    return "Method with the best accuracy is:",list(models.values())[scores.index(max(scores))],"Here are the different scores:",[(list(models.values())[i],scores[i]) for i in range(len(scores))]
```

## Plot function
```bash
def plot_result(result):
  plt.figure(figsize=(30,10))
  plt.bar([result[i][0] for i in range(len(result))],[result[i][1] for i in range(len(result))])
  plt.ylabel('Cross validation score')
  plt.title('Cross Validation score of each model')
  return plt.show()
```
  
