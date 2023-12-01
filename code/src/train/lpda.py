import logging
from exception import DataEmptyException
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import DataFrame



class LPDA:
    lpdaR = None

    def init():
        if (lpdaR is None):
            # Install LPDA
            utils = importr('utils')
            utils.chooseCRANmirror(ind=1)
            lpdaR = importr('lpda')  

    def lpda(data: pd.DataFrame, group: pd.DataFrame, f1 = None, f2 = None):
        logging.debug(">> " + __class__.__name__ + ".fit()")
        model = None
        if (lpdaR is None):
            raise Exception("FATAL: LPDA library reference is uninitialized!")
        if (data is not None and group is not None and len(data) > 0 and len(group) > 0):
            data =  DataFrame(data)
            group = DataFrame(group)
            model = lpdaR.lpda(data, group)
        else:
            raise DataEmptyException("Data empty")
        logging.debug("<< " + __class__.__name__ + ".fit()")
        return model
    
    def predict(coef):
        logging.debug(">> " + __class__.__name__ + ".predict(coef)")
        if (lpdaR is None):
            raise Exception("FATAL: LPDA library reference is uninitialized!")
        prediction = None
        if (coef is not None and len(coef) > 0):
            prediction = lpdaR.predict(coef)
        else:
            raise DataEmptyException("Data empty")
        logging.debug("<< " + __class__.__name__ + ".predict(coef)")
        return prediction