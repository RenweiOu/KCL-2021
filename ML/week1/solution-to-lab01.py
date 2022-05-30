#--
# solution-to-lab01.py
# Edited by Munkhtulga Battogtokh 11-Dec-2021
# Solutions for "Practical 1: Machine Learning Metrics"
# -----------------------------------------------------
# Solve_Practical2.py
# Enrique M Miranda/17-feb-2017
# Solutions for "Practical 2: Machine Learning Metrics".
# Exercise number 6 is not included.
#--

from __future__ import division
import matplotlib.pyplot as plt
import sys
import os
import random as rnd
import math
#import statistics as stat
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# -------------------------------------------
# CREATE CLASSES TO AVOID EXTRA CODE
# -------------------------------------------
# Class for Confusion Matrix:
class ConfusionMatrix( object ):
      def getMatrix( self, test_data, target_data, tree, prediction):
          # initialise counters
          tp = 0 # true positives
          tn = 0 # true negatives
          fp = 0 # false positive
          fn = 0 # false negatives

          # Now iterate for each of the values of the X_test sample data to compare the
          # prediction against the real value:

          for i in range(len(test_data)):
              pred = []
              if len(prediction) >= 1:
                  pred = prediction
                  j = i
              else:
                  pred = tree.predict(test_data[i].reshape(1,-1))
                  j = 0

              if pred[j] == 1 and target_data[i] == 1: # True positive
                  tp += 1
              if pred[j] == 1 and target_data[i] == 0: # False positive
                  fp += 1
              if pred[j] == 0 and target_data[i] == 1: # False negative
                  fn += 1
              if pred[j] == 0 and target_data[i] == 0: # True negative
                  tn += 1

          s = 2
          cMatrix = [[0 for x in range(s)] for y in range(s)]
          cMatrix[0][0] = tp
          cMatrix[0][1] = fp
          cMatrix[1][0] = fn
          cMatrix[1][1] = tn
          return (cMatrix)

# Class for cross validation data sets:
class CrossVal ( object ):
      def getTrainTest ( self, X, y , testRatio):
          # In theory,both X and y should have the same length
          totalRecords = len(X)
          # Estimate the records corresponding to testRatio
          totalTestRecords = math.ceil( totalRecords / (100*testRatio))
          # Now generate an array of random numbers
          testLines = rnd.sample(range(0, int(totalRecords)), int(totalTestRecords))
          # Now find all those lines not in totalRecords
          allLines = range(0, totalRecords)
          trainLines = set(allLines) - set(testLines)
          # Finally, sepparate the data into Train and Test samples:
          X_train = X[list(trainLines),:]
          X_test  = X[testLines,:]
          y_train = y[list(trainLines)]
          y_test  = y[testLines]
          return (X_train, X_test, y_train, y_test)

# Create function for the statistics:
def getStats ( data, mode):
    # MEAN, STD:
    meanS = (1/len(data))*sum(data)
    stdDS = math.sqrt( (1/(len(data)-1))*sum([pow(i-meanS,2) for i in data]) )
    if mode == 1:
       statS = meanS
    elif mode == 2:
       statS = stdDS
    elif mode == 0:
       statS = [meanS, stdDS]
    return ( statS )

# -------------------------------------------
# CREATE FOLDER/FILE TO SAVE RESULTS:
# -------------------------------------------
dir2save = "RESULTS"
command = "mkdir " + dir2save
os.system(command)
command = "./" + dir2save + "/Results.txt"
file2write = open(command,"w")
file2write.write("These are the results:\n")

# -------------------------------------------
# LOAD THE DATA:
# -------------------------------------------
bc = load_breast_cancer()
X = bc.data[:,:] # all features included
y = bc.target


# -------------------------------------------
#            3. CROSS-VALIDATION
# -------------------------------------------

# 3.1: Split the breat cancer data so that you use 90% for training and 10% for testing.
#      What is the accuracy of the tree that you construct?
#      Make sure that you use all the features in the dataset.
# NOTE: 'random_state' variable used to reproduce same results in several runs - data
#        split is the same all the times random_state specifies

overall_test_size = 0.1 # percentage of data for the test sample (used in most exercises)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=overall_test_size,
                                                     random_state=0)
# Now create a tree:
bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)

# Now get the score on the test data:
my_Score = bc_tree.score(X_test, y_test)
file2write.write("\n3.1: Accuracy (score) on X_test and y_test data: %0.4f\n" % (my_Score))

# 3.2: Now find the average score when you build and test 10 such decision trees. And
#      compute the standard deviation of the scores so that you can report the standard
#      error in the average.
scores = [] # initialise variable for all scores
# These tree variables will be used in 4.5
all_trees = []
all_X_train = []
all_y_train = []
all_X_test = []
all_y_test = []
for i in range(10):
    X_train2, X_test2, y_train2, y_test2 = train_test_split( X, y,
                                                        test_size=overall_test_size,
                                                        random_state=11)
    tmp_bc_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train2, y_train2)
    all_trees.append(tmp_bc_tree)
    all_X_train.append(X_train2)
    all_X_test.append(X_test2)
    all_y_train.append(y_train2)
    all_y_test.append(y_test2)
    scores.append(tmp_bc_tree.score(X_test2, y_test2))

# Save the average score (accuracy) and its standard deviation:
file2write.write("\n3.2: After 10 runs of cross-validation, the average score and\n     standard deviation of the trees on the test data are:\n")
file2write.write("     Avg. score: %.4f\n" % (getStats(scores,1)))
file2write.write("     Std. Dev. : %.4f\n" % (getStats(scores,2)))
file2write.write("     Accuracy: %.4f (+/- %0.4f)\n" % (getStats(scores,1), getStats(scores,2)) )

#file2write.write("     Avg. Precision: %.4f (+/- %.4f)\n" % (getStats(precisions,1), getStats(precisions,2)))


# 3.3: Now try out the cross-validation function in scikit-learn. Run a 10-fold cross-
#      validation on the breast cancer data. How do the average accuracy and standard
#      deviation compare with the results you obtained earlier with the 10 trees from
#      randomly assigned data?

# Run the 10-fold cross-validation on bc data (generated in 3.1):
cv_scores = cross_val_score(bc_tree, X, y, cv=10, scoring = 'f1_macro')
file2write.write("\n3.3: After 10-fold cross-validation, the results on the test\n     data are:\n")
file2write.write("     Avg. score: %.4f\n" % (getStats(cv_scores,1)))
file2write.write("     Std. Dev. : %.4f\n" % (getStats(cv_scores,2)))
file2write.write("     Accuracy: %.4f (+/- %0.4f)\n" % (getStats(cv_scores,1), getStats(cv_scores,2)) )
# -> "tmp_bc_tree.score" is more consistent (lower std. dev.) than "cross_val_score"

# 3.4: See CLASS "CrossVal" and follow this example:
# Create object:
crossValidation = CrossVal()
cV = crossValidation.getTrainTest(X, y, 0.1) # 10% for test data
tmpXtrain = cV[0]
tmpXtest  = cV[1]
tmpYtrain = cV[2]
tmpYtest  = cV[3]
# You can iterate the object and get the different data sets.

# -------------------------------------------
#                4. METRICS
# -------------------------------------------

# 4.1: Compute the confusion matrix of a breast cancer decision tree. Start with last one
#      you learnt for the cross-validation exercise. You should have a tree, call it
#      yr_tree, a test set X_test and a set of true labels for the test data, y_test. The
#      simplest way to compute the precision is to run through X_test, example by
#      example, feeding the example into the classifier and comparing the result with the
#      corresponding element of y_test. Depending on the result, you will have a true
#      positive, a true negative, a false positive or a false negative. Running through
#      the whole test set will give you the numbers you need for a confusion matrix.

# The Confusion Matrix is:
#                  |              ACTUAL
#                  |      YES              NO
#            YES   | True Positive     False Positive
# PREDICTED        |
#            NO    | False Negative    True Negative

# Create yr_tree (use X_train, X_test, y_train_ y_test from 3.1):
yr_tree = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
# Create an object for the confusion matrix:
cM = ConfusionMatrix()
# Get the confusion matrix
prediction = [] # variable used in the class
M = cM.getMatrix( X_test, y_test, yr_tree, prediction )
tp = M[0][0]
fp = M[0][1]
fn = M[1][0]
tn = M[1][1]
# Now print the confusion matrix:
file2write.write("\n4.1: The confusion matrix is:\n")
file2write.write("                 |      ACTUAL\n")
file2write.write("                 |   YES      NO\n")
file2write.write("            YES  |   %d        %d\n" % (tp , fp) )
file2write.write(" PREDICTED       |\n")
file2write.write("            NO   |   %d        %d\n" % (fn , tn) )

# 4.2: Compute the precision of a decision tree. This is easy now that you have the true
#      positives and false positives.
precision = tp / (tp + fp)
file2write.write("\n4.2: The precision is: %.4f\n" % precision)

# 4.3: Compute the recall of a decision tree.
recall = tp / (tp + fn)
file2write.write("\n4.3: The recall is: %.4f\n" % recall)

# 4.4: Compute the F1 score of a decision tree.
F1_score = 2 * ((precision * recall)/(precision + recall))
file2write.write("\n4.4: The F1 score is: %.4f\n" % F1_score)

# 4.5: Now you can compute the mean and standard error of the precision and recall of all
#      the decision trees that you learn during 10-fold cross-validation.

cnt = 0 # extra counter
precisions = []
recalls = []
prediction = [] # variable used in the class
for z_tree in all_trees:
    cM = ConfusionMatrix() # create an object for the confusion matrix
    M = cM.getMatrix( all_X_test[cnt], all_y_test[cnt], z_tree, prediction ) # get the confusion matrix
    precision = M[0][0] / (M[0][0] + M[0][1]) # tp / (tp + fp)
    recall = M[0][0] / (M[0][0] + M[1][0]) # tp / (tp + fn)
    precisions.append(precision)
    recalls.append(recall)
    cnt += 1

file2write.write("\n4.5: After 10-fold cross-validation, the results on the test\n     data are:\n")
file2write.write("\n4.5: After 10-fold cross-validation, the results on the test\n     data are:\n")
file2write.write("     Precision: %.4f (+/- %.4f)\n" % (getStats(precisions,1), getStats(precisions,2)))
file2write.write("     Recall: %.4f (+/- %.4f)\n" % (getStats(recalls,1), getStats(recalls,2)))



# 4.6: Compute the ROC curve of a breast cancer classifier. The Breast cancer decision
#      trees don't give much of an ROC curve because they only make definite class
#      predictions (so adjusting the threshold does nothing). So, we will use a k-nearest
#      neighbour classifier.
# REMARK: most classification problems can be cast as a probabilistic prediction
#         returning
#                     p(y_i | X_i D )
#         and we turn this into "yes" and "no" using:
#                    *  YES,   if p(y_i | X_i D ) > threshold
#  classification =
#                    *  NO,    otherwise
#
#      Moving the threshold changes the true positives (tp) and false positives (fp).
#      Therefore, the ROC curve plots pairs of tp and fp as we vary the threshold. The
#      area under the curve is related to the probability that the classifier will
#      correctly classify a randomly chosen example.

# First, we get the classifier from k-nearest neighbour (use the data from 3.1):
bc_neigh = KNeighborsClassifier(n_neighbors=5)
bc_neigh.fit(X_train, y_train)
p_prediction = bc_neigh.predict_proba(X_test)

# Now range across a set of thresholds to build data for an ROC curve:
# Initialise the list for tp and fp
tp_list = []
fp_list = []
precisions = []
recalls = []
tree_tmp = [] # variable used in the class
for j in range(500): # test 500 different thresholds
    threshold = 0.01 * j
    # Initialise list of predictions
    t_predictions = []

    # p_prediction is a set of class probabilities. For a given threshold, we can turn
    # this into a list of class predictions. REMEMBER: the area under the curve is
    # related to the probability that the classifier will correctly classify a randomly
    # chosen example
    for i in range(len(X_test)):
        if p_prediction[i,0] >= threshold: # above the threshold
           t_predictions.append(0)
        else:                              # below the threshold
           t_predictions.append(1)


    # Get the confusion matrix elements:
    cM = ConfusionMatrix() # create object
    M = cM.getMatrix( X_test, y_test, tree_tmp, t_predictions ) # get matrix

    # Now get the tp and fp:
    tp_list.append(M[0][0])
    fp_list.append(M[0][1])

    # For the statistics:
    if M[0][0] != 0 or M[0][1] != 0:
       precisions.append(M[0][0] / (M[0][0] + M[0][1]))
    else:
       precisions.append(0)

    if M[0][0] != 0 or M[1][0] != 0:
       recalls.append(M[0][0] / (M[0][0] + M[1][0]))
    else:
       recalls.append(0)

# The stats:
avgF1_score = 2 * ((getStats(precisions,1) * getStats(recalls,1))/(getStats(precisions,1) + getStats(recalls,1)))

file2write.write("\n4.6: The averaged precision, recall and F1 score are:\n")
file2write.write("     Avg. Precision: %.4f (+/- %.4f)\n" % (getStats(precisions,1), getStats(precisions,2)))
file2write.write("     Avg. Recall: %.4f (+/- %.4f)\n" % (getStats(recalls,1), getStats(recalls,2)))
file2write.write("     Avg. F1 score: %.4f \n" % (avgF1_score))

# Normalise the tp and fp lists in percentual terms:
max_tp = max(tp_list)
max_fp = max(fp_list)
tp_list = [(100*(i/max_tp)) for i in tp_list]
fp_list = [(100*(i/max_fp)) for i in fp_list]
# The plot:
plt.plot(fp_list, tp_list, '-o')
plt.xlabel('False positives rate')
plt.ylabel('True positives rate')
plt.axis([-10, 110, -10, 110])
command = "./" + dir2save + "/ROC.pdf"
plt.savefig(command)
#plt.show()


# -------------------------------------------
#            5. NEAREST NEIGHBOUR
# -------------------------------------------

# 5.1: Run a 10-fold cross-validation on the k-nearest neighbour classifier applied to
#      the breast cancer dataset.

precisions = []
recalls = []
F1_scores = []
tree_tmp = []
threshold = 0.8 # static threshold - depending on this value, the results will vary.
file2write.write("\n5.1: The results for the 10-fold cross validation on the\nk-nearest neighbour are:\n")
file2write.write("            Precision   Recall   F1 score\n")
for i in range(10):
    # Cross-validation data sets:
    crossValidation = CrossVal()
    cV = crossValidation.getTrainTest(X, y, 0.1) # 10% for test data
    Xtrain = cV[0]
    Xtest  = cV[1]
    Ytrain = cV[2]
    Ytest  = cV[3]

    # Use train and test samples produced in 3.2.
    # Get classification:
    bcf_neigh = KNeighborsClassifier(n_neighbors=5)
    bcf_neigh.fit(Xtrain, Ytrain)
    pf_prediction = bcf_neigh.predict_proba(Xtest)

    t_prediction = []
    # New prediction list:
    for j in range(len(Xtest)):
        if pf_prediction[j,0] >= threshold: # above the threshold
           t_prediction.append(0)
        else:                              # below the threshold
           t_prediction.append(1)

    # Get the confusion matrix elements:
    cM = []
    M = []
    cM = ConfusionMatrix() # create object
    M = cM.getMatrix( Xtest, Ytest, tree_tmp, t_prediction ) # get matrix
    # For the statistics:
    precision = (M[0][0] / (M[0][0] + M[0][1]))
    recall = (M[0][0] / (M[0][0] + M[1][0]))
    F1s = 2 * ((precision * recall)/(precision + recall))

    precisions.append(precision)
    recalls.append(recall)
    F1_scores.append(F1s)

    file2write.write("     Run %d:   %.4f    %.4f    %.4f\n" % (i, precision, recall, F1s))

# 5.2: Compute the average precision, recall and F1 score across the 10-fold cross-
#      validation. That is, compute precision, recall and F1 score for each fold, and
#      compute the average and standard deviation over the 10 values

file2write.write("\n5.2: The averaged results for the 10-fold cross validation on the\nk-nearest neighbour are:\n")
file2write.write("     Avg. Precision: %.4f (+/- %.4f)\n" % (getStats(precisions,1), getStats(precisions,2)))
file2write.write("     Avg. Recall: %.4f (+/- %.4f)\n" % (getStats(recalls,1), getStats(recalls,2)))
file2write.write("     Avg. F1 score: %.4f (+/- %.4f) \n" % (getStats(F1_scores,1), getStats(F1_scores,2)))

file2write.close()
sys.exit()



