import TIcE
import ExTIcE
import OCScorers
import KM12
from EN import EN
from PAT import PAT
from PAT import PAT_ActualTime
from ODIn import ODIn
from PE import PE

# This file lists all methods available to be used by one of the Experiment*.py scripts.
# Each method receives a list of obervation points, a list of features that must be used, and the name of the feature
# that marks whether one observation is labeled or not.
# It is possible to pass additional configuration arguments required each method. Follow the examples below.

methods = {
  "extice": { # ExTIcE
    "func": ExTIcE.TIcE,
    "kargs": {
      "evaluation": TIcE.EvaluatePaper
    },
  },
  "tice": { # TIcE
    "func": TIcE.TIcE,
    "kargs": {
      "evaluation": TIcE.EvaluatePaper
    },
  },
  "en": { # Elkan (SVM's gamma = auto)
    "func": EN,
    "kargs": {},
  },
  "en_scl": { # Elkan (SVM's gamma = scale)
    "func": EN,
    "kargs": {"gamma":"scale"},
  },
  "patm": { # PAT with Mahalanobis
    "func": PAT,
    "kargs": {
      "scorer": OCScorers.Mahalanobis
    },
  },
  "patm_nowup": { # PAT with Mahalanobis (exclusive for ExperimentTimePAT.py)
    "func": PAT_ActualTime,
    "kargs": {
      "scorer": OCScorers.Mahalanobis
    },
  },
  "patsvm": { # PAT with One-class SVM (gamma = auto)
    "func": PAT,
    "kargs": {
      "scorer": OCScorers.OneClassSVM()
    },
  },
  "patsvm_scl": { # PAT with One-class SVM (gamma = scale)
    "func": PAT,
    "kargs": {
      "scorer": OCScorers.OneClassSVM(gamma="scale")
    },
  },
  "odinm": { # ODIn with Mahalanobis
    "func": ODIn,
    "kargs": {
      "scorer": OCScorers.Mahalanobis
    },
  },
  "odinsvm": { # ODIn with One-class SVM (gamma = auto)
    "func": ODIn,
    "kargs": {
      "scorer": OCScorers.OneClassSVM()
    },
  },
  "odinsvm_scl": { # ODIn with One-class SVM (gamma = scale)
    "func": ODIn,
    "kargs": {
      "scorer": OCScorers.OneClassSVM(gamma="scale")
    },
  },
  "km1": {
    "func": KM12.KM1,
    "kargs": {},
  },
  "km2": {
    "func": KM12.KM2,
    "kargs": {},
  },
  "pe": {
    "func": PE,
    "kargs": {},
  },
  "tpm": { # Minimum p between PAT with Mahalanobis and ExTIcE.
    "func": lambda *args, **kargs: min(TIcE.TIcE(*args, **kargs), PAT(*args, **kargs, scorer=OCScorers.Mahalanobis), key=lambda x: (x[1], -x[0])),
    "kargs": {},
  }
}