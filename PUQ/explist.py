
# This file provides a list of experiments that can be tested with varying methods.
# Each entry of an experiment is detailed below:
# input: the filename of the dataset.
# output: the filename structure of the output. The first %s refers to the type of experiment
#   (which is automatically inserted by Experiment.py, ExperimentEachNeg, etc.). The second %s
#   refers to the method applied.
# class_feature: which column of the dataset indicates the class value.
# positive_label: which value of the class_feature column corresponds to the positive class.
# negative_labels: a list of negative classes, from the dataset, that should be considered in the experiment.
#   If it is None, then all classes other than the positive one are considered.
#   It can also be a filter function that accepts a string and returns True if the corresponding class should be
#   considered, and False otherwise.
# features: a list of features, from the dataset, that should be considered in the experiment.
#   If it is None, then all features other than the class_feature are considered.
#   It can also be a filter function that accepts a string and returns True if the corresponding feature should be
#   considered, and False otherwise.

Experiments = {
  "wine": { # Wine Quality
    "input": "../data/wine.csv",
    "output": "../results%s/wine_%s.csv",
    "class_feature": "type",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "pen": { # Pen-Based Recognition of Handwritten Digits
    "input": "../data/pendigits.csv",
    "output": "../results%s/pendigits_%s.csv",
    "class_feature": "digit",
    "positive_label": 5,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "bgn": { # BNG (Japanese Vowels)
    "input": "../data/BNG_JapaneseVowels.csv",
    "output": "../results%s/BNG_JapaneseVowels_%s.csv",
    "class_feature": "speaker",
    "positive_label": 1,
    "negative_labels": None,
    "features": ["coefficient1","coefficient2","coefficient3","coefficient4","coefficient5","coefficient6","coefficient7","coefficient8","coefficient9","coefficient10","coefficient11","coefficient12"],
  },
  "letter": { # Letter
    "input": "../data/letter.csv",
    "output": "../results%s/letter_%s.csv",
    "class_feature": "class",
    "positive_label": "W",
    "negative_labels": None, # all other labels
    "features": ["x-box","y-box","width","high","onpix","x-bar","y-bar","x2bar","y2bar","xybar","x2ybr","xy2br","x-ege","xegvy","y-ege","yegvx"],
  },
  "pulsar": { # HRU2
    "input": "../data/HTRU_2.csv",
    "output": "../results%s/HTRU_2_%s.csv",
    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "frogs": { # Anuran Calls
    "input": "../data/Frogs_MFCCs.csv",
    "output": "../results%s/Frogs_MFCCs_%s.csv",
    "class_feature": "Family",
    "positive_label": "Hylidae",
    "negative_labels": ["Leptodactylidae"],
    "features": ['MFCCs_1','MFCCs_2','MFCCs_3','MFCCs_4','MFCCs_5','MFCCs_6','MFCCs_7','MFCCs_8','MFCCs_9','MFCCs_10','MFCCs_11','MFCCs_12','MFCCs_13','MFCCs_14','MFCCs_15','MFCCs_16','MFCCs_17','MFCCs_18','MFCCs_19','MFCCs_20','MFCCs_21','MFCCs_22'],
  },
  "handwritten": { # Handwritten
    "input": "../data/Handwritten.csv",
    "output": "../results%s/Handwritten_%s.csv",
    "class_feature": "letter",
    "positive_label": "q",
    "negative_labels": None, # all other labels
    "features": lambda x: x != 'letter' and x != 'author',
  },
  "arabic": { # Arabic Digit
    "input": "../data/ArabicDigit.csv",
    "output": "../results%s/ArabicDigit_%s.csv",
    "class_feature": "digit",
    "positive_label": 0,
    "negative_labels": None, # all other labels
    "features": lambda x: x != 'sex' and x != 'digit',
  },
  "insects": { # Insects
    "input": "../data/AllSpecies.csv",
    "output": "../results%s/AllSpecies_%s.csv",
    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": [1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 24],
    "features": ["temperature", "wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"],
  },
  "newinsects": { # Insects v2
    "input": "../data/InsectsPUQ.csv",
    "output": "../results%s/InsectsPUQ_%s.csv",
    "class_feature": "label",
    "positive_label": 2,
    "negative_labels": None, # all other
    "features": None,
  },
}