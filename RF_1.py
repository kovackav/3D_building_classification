from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
from matplotlib import *
from matplotlib import pyplot as plt

training = gpd.read_file(r'C:\Users\vitak\Desktop\DP_V2\RF\Data\learning_non_agg_build.shp')
testing = gpd.read_file(r'C:\Users\vitak\Desktop\DP_V2\RF\Data\testing_non_agg_build.shp')
evaluate = gpd.read_file(r'C:\Users\vitak\Desktop\DP_V2\RF\Data\testing_non_agg_build.shp')
evaluate_y = evaluate['typ']
output = r'C:\Users\vitak\Desktop\DP_V2\RF\Output\RF_out_cela_Praha.shp'

# drop of target variable for training data
yt = training['typ']
Xt = training.drop(['typ', 'geometry'], axis = 1)

# drop of target variable for testing data
testing['typ'] = ''
ya = testing['typ']
Xa = testing.drop(['typ', 'geometry'], axis = 1)

X_train = Xt
y_train = yt

print("Shape of training dataset: " + str(X_train.shape))
print("Shape of target: " + str(y_train.shape))

# here tunning of RF Classifier can be done
rf_Model = RandomForestClassifier()

rf_Model.fit(X_train,y_train)
print("Features used for training: " + str(rf_Model.score(X_train,y_train)))
output = (rf_Model.predict(Xa))
testing['typ'] = output


# testing accuracy
print("Estimated accuracy: " + str(accuracy_score(evaluate_y, output)))

# save output
testing.to_file(output)

# confusion matrix 
matrix = confusion_matrix(evaluate_y, output)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# calculation of precision
precision = ((matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3] + matrix[4][4]) /
            (np.sum(matrix)))
print(precision)
# plot
plt.figure(figsize=(12,5))
sns.set(font_scale=1)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Oranges, linewidths=0.1)

# add labels to the plot
#class_names = ['Residental', 'Non-residental'] 
class_names = ['bd', 'rd', 'o', 'p', 'k']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted values')
plt.ylabel('True values')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()