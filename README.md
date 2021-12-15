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


