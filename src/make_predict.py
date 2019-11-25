from .data import make_dataset
from .features import build_features
from .models import predict_model
from .visualization import visualize


def make_predict(pFolder,pName):
    """Run the whole 2D analysis

    This function collects all the necessary modules and runs them
    in order, to replicate the results of the manuscript for a single patient
    using previously trained models.
    The functions which are being used have been imported from jupyter notebooks
    using the makeSrc.sh script in the helper/ folder.

    Parameters
    ----------
    pFolder : string
        directory where the data are stored
    pName : string
        in the pFolder, there must be a file tMap_<pName>.nii
        accordingly, this variable defines what the <pName> string
        of that file is

    Returns
    -------
    
    nothing, instead writes data to pFolder
    """
    make_dataset.makeP(pFolder,pName) # 04-mw-get-lat-data.ipynb
    build_features.makeP(pFolder,pName) # 05-mw-compute-2d-and-li.ipynb
    predict_model.makeP(pFolder,pName) # 09-mw-apply-classifier-to-data.ipynb
    fig = visualize.makeP(pFolder,pName) # 12-mw-single-patient-results.ipynb
    prob_fig = visualize.makePProb(pFolder,pName) # 12-mw-single-patient-results.ipynb
    return
