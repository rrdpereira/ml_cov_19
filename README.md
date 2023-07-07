# ml_cov_19
Machine Learning project for COVID-19 dataset provided by University of Montreal [https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset]

# Machine Learning (AM) Project for CCO - 724 - Machine Learning of PPGCC UFSCAr [https://www.ppgcc.ufscar.br/pt-br/programa/estrutura-curricular/disciplinas-do-programa/cco-724-aprendizado-de-maquina]
- Students: Gustavo das Neves Ubeda and Robson Rogério Dutra Pereira
- Professor: Prof. Dr. Diego Furtado Silva
- Dataset: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
- Video Presentation: https://loom.com/share/5b2fd25db9974fd59cf1967a5a2b6f6c
- Video Presentation: https://youtu.be/3Eo-i8RRLeI

### About the dataset
The dataset used throughout the experiment was provided by the University of Montreal for the COVID-19 detection using only chest X-rays. The dataset was made available already separated into training and test subsets, containing the following classes and their respective amounts of images:
- Training partition:
   - Patients with COVID-19: 111 images [Covid]
   - Patients with Pneumonia: 70 images [Viral Pneumonia]
   - Healthy patients: 70 images [Normal]

- Test partition:
   - Patients with COVID-19: 26 images [Covid]
   - Patients with Pneumonia: 20 images [Viral Pneumonia]
   - Healthy patients: 20 images [Normal]

### Windows OS filepath
- In our case, we are using:
   - C:\GitProjW\ml_cov_19\Covid19-dataset\train
   - C:\GitProjW\ml_cov_19\Covid19-dataset\train\output
   - C:\GitProjW\ml_cov_19\Covid19-dataset\test

- Wich are conveted on the Python code for:
   - import sys, time, os, datetime, glob
   - dir_train = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','train')
   - dir_aug = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','train','output')
   - dir_test = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','test')

### Linux Ubuntu OS filepath
- The Python code '~'(home):
   - dir_train = os.path.join('~','GitProjW','ml_cov_19','Covid19-dataset','train')
   - dir_aug = os.path.join('~','GitProjW','ml_cov_19','Covid19-dataset','train','output')
   - dir_test = os.path.join('~','GitProjW','ml_cov_19','Covid19-dataset','test')

### Data balancing
The analyzed dataset does not have balance for all classes, only patients with pneumonia and normal are balanced, and those with COVID-19 have more examples in the training and test samples. As this disparity is positive for the most important class (the detection of COVID-19), then techniques for balancing classes was not implemented.

After the training and the tests, we got the following results:

                precision    recall  f1-score   support

          Covid       0.76      0.96      0.85        26
Viral Pneumonia       0.57      0.60      0.59        20
         Normal       0.83      0.50      0.62        20

       accuracy                           0.71        66
      macro avg       0.72      0.69      0.69        66
   weighted avg       0.72      0.71      0.70        66

And the cross validation was performed too:

Cross Validation accuracy scores: [0.80769231 0.8        0.96       0.96       0.8        0.88
 0.88       0.96       0.84       0.8       ]
Cross Validation accuracy MEAN +/- STANDARD DEVIATION: 0.869 +/- 0.066

### Augmentation
Since it is possible to obtain better predictive results for Machine Learning (ML) models trained from larger databases, as seen in the course, the augmentation process was done to the training set to validate data augmentation hypothesis.

After the augmentation, the training and the tests, we got the following results:
                 precision    recall  f1-score   support

          Covid       0.81      0.96      0.88        26
Viral Pneumonia       0.52      0.55      0.54        20
         Normal       0.71      0.50      0.59        20

       accuracy                           0.70        66
      macro avg       0.68      0.67      0.67        66
   weighted avg       0.69      0.70      0.69        66

And also the cross validation was performed too:

Cross Validation accuracy scores: [0.92105263 0.88157895 0.89473684 0.81333333 0.88       0.89333333
 0.85333333 0.88       0.89333333 0.86666667]

Cross Validation accuracy MEAN +/- STANDARD DEVIATION: 0.878 +/- 0.027

### References
- https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
- https://stackoverflow.com/questions/58270129/convert-categorical-data-into-numerical-data-in-python
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
- https://augmentor.readthedocs.io/en/master/userguide/install.html