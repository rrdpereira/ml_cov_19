# -*- coding: utf-8 -*-
"""ml_cov_19_code.ipynb

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
In our case, we are using:
C:\GitProjW\ml_cov_19\Covid19-dataset\train
C:\GitProjW\ml_cov_19\Covid19-dataset\train\output
C:\GitProjW\ml_cov_19\Covid19-dataset\test

Wich are conveted on the Python code for:
import sys, time, os, datetime, glob
dir_train = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','train')
dir_aug = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','train','output')
dir_test = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','test')

### Linux Ubuntu OS filepath
The Python code '~'(home):
dir_train = os.path.join('~','GitProjW','ml_cov_19','Covid19-dataset','train')
dir_aug = os.path.join('~','GitProjW','ml_cov_19','Covid19-dataset','train','output')
dir_test = os.path.join('~','GitProjW','ml_cov_19','Covid19-dataset','test')

### Data balancing
The analyzed dataset does not have balance for all classes, only patients with pneumonia and normal are balanced, and those with COVID-19 have more examples in the training and test samples. As this disparity is positive for the most important class (the detection of COVID-19), then techniques for balancing classes was not implemented.
"""

import sys, time, os, datetime, glob
from skimage.io import imread
from skimage.color import rgb2gray,rgba2rgb
from skimage.transform import resize
import numpy as np

# Image features lib
from skimage.feature import hog

# augmentation lib
import Augmentor

# Machine Learning modeling and avaliation
from sklearn.ensemble import RandomForestClassifier # Trunk method Ensemble of Machine Learning
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from platform import python_version

print(f"(Sys version) :|: {sys.version} :|:")
os.system("which python")
print(f"(Python version) :#: {python_version()} :#:")

# Datasets 
dir_train = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','train')
print(dir_train)
dir_aug = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','train','output')
print(dir_aug)
dir_test = os.path.join('c:\\','GitProjW','ml_cov_19','Covid19-dataset','test')
print(dir_test)

# Classes
class_names = ['Covid', 'Viral Pneumonia', 'Normal']

def read_process_img(dir_datasets, class_names):
  # KDD definitions
  class_list   = []
  n_class_list = []
  feature_list = []

  # Datasets images reading
  class_number = 0
  for class_name in class_names:
    # Linux OS filepath
    # for img_name in os.listdir(f'{dir_datasets}/{class_name}'):
    #   # print("ForImread:\n{0}".format(f'{dir_datasets}/{class_name}/{img_name}'))
    #   img = imread(f'{dir_datasets}/{class_name}/{img_name}')

    # Windows OS filepath
    for img_name in os.listdir(f'{dir_datasets}\{class_name}'):
      # print("ForImread:\n{0}".format(f'{dir_datasets}\{class_name}\{img_name}'))
      img = imread(f'{dir_datasets}\{class_name}\{img_name}')
      # print("imgShape:\n{0}".format(img.shape))

      # Black and white conversion
      if len(img.shape) == 2:
          img_gray = img
      elif len(img.shape) == 3:
          if img.shape[-1] == 4:
              img_rgb = rgba2rgb(img)
              img_gray = rgb2gray(img_rgb)
          else:
              img_gray = rgb2gray(img)

      # Images resizing
      r_img = resize(img_gray, (128, 128))

      # Feature extration by HOG method
      hog_feature, hog_img = hog(r_img, orientations=9,
                                pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                visualize=True)

      # KDD data storage
      class_list.append(class_name)
      n_class_list.append(class_number)
      feature_list.append(hog_feature)

    class_number += 1

  return feature_list, n_class_list, class_list

X_train,y_train,y_train_class=read_process_img(dir_train,class_names)
X_test,y_test,y_test_class=read_process_img(dir_test,class_names)

classificator = RandomForestClassifier(n_estimators=100, random_state=13)
classificator.fit(X_train, y_train)

# Class predition of test set
y_predicted = classificator.predict(X_test)

# Report model performance
print(classification_report(y_test, y_predicted, target_names = class_names))

# Runs cross validation
scores = cross_val_score(classificator, X=X_train, y=y_train, cv=10, n_jobs=1)

print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

# Number of augmented images
count_img = 0
for cl in class_names:
  try:
    count_img += len(os.listdir(f'{dir_aug}/{cl}'))
  except:
    pass

if count_img < 502:
  # Increase train set
  p = Augmentor.Pipeline(dir_train)

  p.rotate(probability=0.9, max_left_rotation=10, max_right_rotation=10) # Images rotation
  p.zoom(probability=0.1, min_factor=1.1, max_factor=1.3) # Imagese zooming 
  p.sample(502) # Creating N new examples

X_aug, y_aug, y_aug_class = read_process_img(dir_aug, class_names)

classificator_aug = RandomForestClassifier(n_estimators=100, random_state=13)
classificator_aug.fit(X_train + X_aug, y_train + y_aug)

# Class predition of test set
y_predicted_aug = classificator_aug.predict(X_test)

# Report model performance
print(classification_report(y_test, y_predicted_aug, target_names = class_names))

# Runs cross validation
scores_aug = cross_val_score(classificator_aug, X=(X_train + X_aug), y=(y_train + y_aug), cv=10, n_jobs=1)

print('CV accuracy scores: %s' % scores_aug)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores_aug),np.std(scores_aug)))