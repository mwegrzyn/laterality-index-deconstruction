
# coding: utf-8

# # Make predictions  
# 
# This script uses the classifiers with the highest accuracy to get the LIs and predictions for all cases

# ### import modules
#
## In[1]:
#
#get_ipython().magic(u'matplotlib inline')
#
#

# In[2]:

import os
import fnmatch

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pylab as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn import preprocessing, model_selection, metrics
from nilearn import plotting

import pickle


# ### get absolute directory of project

# In[3]:

# after converstion to .py, we can use __file__ to get the module folder
try:
    thisDir = os.path.realpath(__file__)
# in notebook form, we take the current working directory (we need to be in 'notebooks/' for this!)
except:
    thisDir = '.'
# convert relative path into absolute path, so this will work with notebooks and py modules
supDir = os.path.abspath(os.path.join(os.path.dirname(thisDir), '..'))

supDir


# In[4]:

sns.set_style('white')
sns.set_context('poster')


# ### load labeler

# In[5]:

myLabeler = pickle.load(open('%s/models/myLabeler.p' % supDir, 'rb'))

#
## In[6]:
#
#myLabeler.classes_
#
#

# ### collect the parameters that allow for above-chance prediction

# In[7]:

#aboveDf = pd.read_csv('%s/models/aboveDf_clf_1d_drop.csv' % supDir,index_col=[0,1])

#
## In[8]:
#
#aboveDf.T
#
#

# In[9]:

#clfDict = pickle.load(open('%s/models/clf_1d_drop.p' % supDir))

#
## In[14]:
#
#clfDict[95][3-0]
#
#

# ### predictions for one value and one parameter set

# In[15]:

def makePred(x, roiPc, tThresh, clfDict, myLabeler=myLabeler):

    thisClf = clfDict[roiPc][tThresh]['clf']
    thisScaler = clfDict[roiPc][tThresh]['scaler']

    xArr = np.array(x)
    xScaled = thisScaler.transform(xArr.reshape(1, -1 * xArr.shape[-1]))

    y_pred = thisClf.predict_proba(xScaled)

    df = pd.DataFrame(y_pred).T
    idx = [myLabeler.inverse_transform([x])[-1] for x in thisClf.classes_]
    df.index = idx

    return df


# Example:
#
## In[16]:
#
#thesePreds = makePred([0.0], 0, 3, clfDict)
#
#
#
## In[17]:
#
#thesePreds
#
#

# ### predictions for one patient, for all above-chance parameters

# In[18]:

def changeDf(df):

    idx1 = df.columns.get_level_values(0).astype(float)
    idx2 = df.columns.get_level_values(1)

    mIdx = pd.MultiIndex.from_arrays([idx2, idx1])

    df.columns = mIdx
    df.sort_index(axis=1,inplace=True)

    return df


# Example Patient:
#
## In[19]:
#
#pCsv = '%s/data/interim/csv/roiLaterality_patX.csv' % supDir
#
#
#
## In[20]:
#
#pName = 'pat_%s' % (pCsv.split('_pat')[-1].split('.')[0])
#pName
#
#
#
## In[21]:
#
#pDf = pd.read_csv(pCsv, index_col=[0], header=[0, 1])
#pDf = changeDf(pDf)
#
#
#
## In[22]:
#
#pDf.tail()
#
#

# In[23]:

def getP(pDf, pName, roiSize, thresh, dims, myLabeler=myLabeler):

    if dims == 1:
        liValue = pDf.loc[roiSize, 'LI'].loc[thresh]
        thisDf = pd.DataFrame([liValue], index=[pName], columns=['LI'])

    elif dims == 2:
        diffValue = pDf.loc[roiSize, 'L-R'].loc[thresh]
        diffDf = pd.DataFrame([diffValue], index=[pName], columns=['L-R'])

        addValue = pDf.loc[roiSize, 'L+R'].loc[thresh]
        addDf = pd.DataFrame([addValue], index=[pName], columns=['L+R'])

        thisDf = pd.concat([diffDf, addDf], axis=1)

    return thisDf

#
## In[24]:
#
#getP(pDf, pName, roiSize=50, thresh=5.8, dims=1)
#
#

# In[25]:

def makeBestPreds(pCsv, aboveDf, clfDict, dims):

    pName = 'pat_%s' % (pCsv.split('_pat')[-1].split('.')[0])
    pDf = pd.read_csv(pCsv, index_col=[0], header=[0, 1])
    pDf = changeDf(pDf)
    
    valueDict = {}
    predDf = pd.DataFrame()

    # here we loop through the aboveDf, which has in its index
    # all parameters that we want

    # get the table with the roi size
    for pc in aboveDf.index.levels[0]:
        # get the data for the threshold
        for t in aboveDf.loc[pc].index:

            thisParam = getP(pDf, pName, pc, t, dims)

            # store the value
            thisVals = list(thisParam.loc[pName])
            valueDict[str(pc) + '_' + str(t)] = thisVals

            # make predictions, these are like df's
            try:
                thisPred = makePred(thisVals, pc, t, clfDict)
            except:
                thisPred = pd.DataFrame({
                    'bilateral': 0,
                    'left': 0,
                    'right': 0,
                    'inconclusive': 1
                },
                                        index=[0]).T
            #store predictions
            thisPred = thisPred.T
            thisPred.index = [str(pc) + '_' + str(t)]
            predDf = pd.concat([predDf, thisPred])

    if dims == 1:
        valueDf = pd.DataFrame(valueDict, index=['LI']).T
    elif dims == 2:
        valueDf = pd.DataFrame(valueDict, index=['L-R', 'L+R']).T

    # average
    
    meanValueDf = pd.DataFrame(valueDf.mean())
    meanPredDf = pd.DataFrame(predDf.mean())
    
    meanDf = pd.concat([meanValueDf,meanPredDf]).T
    meanDf.index = [pName]

    return valueDf, predDf, meanDf


# Example:
#
## In[26]:
#
#valueDf, predDf, meanDf = makeBestPreds(pCsv, aboveDf, clfDict, dims=1)
#
#
#
## In[27]:
#
#meanDf
#
#
#
## In[28]:
#
#fuDf = predDf.copy()
#fuDf.index = pd.MultiIndex.from_tuples(list([x.split('_') for x in fuDf.index]))
#
#

# In[29]:

def changeIdx(df):

    idx1 = df.index.get_level_values(0).astype(int)
    idx2 = df.index.get_level_values(1).astype(float)

    mIdx = pd.MultiIndex.from_arrays([idx2, idx1])

    df.index = mIdx
    df.sort_index(axis=0,inplace=True)

    return df

#
## In[30]:
#
#fuDf = changeIdx(fuDf)
#
#
#
## In[31]:
#
#fig = plt.figure(figsize=(16,6))
#for i,c in enumerate(fuDf.columns):
#    ax = plt.subplot(1,fuDf.columns.shape[-1],i+1)
#    thisDf = fuDf.loc[:,[c]].unstack()[c].T
#    sns.heatmap(thisDf,cmap='rainbow',vmin=0,vmax=1,axes=ax)
#    ax.set_title(c)
#plt.tight_layout()
#plt.show()
#
#

# In[32]:

def makeAllComputations(pCsv, dims, drop, sigLevel=0.001):

    dropStr = ['full', 'drop'][drop]
    dimStr = ['1d', '2d'][dims - 1]

    # load the classifier
    clfDict = pickle.load(
        open('%s/models/clf_%s_%s.p' % (supDir, dimStr, dropStr), 'rb'))

    accDict = pickle.load(
        open('%s/models/acc_%s_%s.p' % (supDir, dimStr, dropStr), 'rb'))

    aboveDf = pd.read_csv(
        '%s/models/aboveDf_clf_%s_%s.csv' % (supDir, dimStr, dropStr),
        index_col=[0, 1])

    # compute
    valueDf, predDf, meanDf = makeBestPreds(pCsv, aboveDf, clfDict, dims=dims)

    # if we compute the 1-dimensional LI and do not want to model inconclusive cases,
    # we still need to handle cases where division by zero occurs
    # therefore, we compute the proportion of cases where neither of the 3 main classes was predicted
    if dims == 1 and drop == True:
        meanDf.loc[:,'inconclusive'] = 1 - meanDf.loc[:,['left','bilateral','right']].sum(axis=1)
    
    return valueDf, predDf, meanDf

#
## In[33]:
#
#valueDf, predDf, meanDf = makeAllComputations(pCsv, dims=2, drop=True)
#
#
#
## In[36]:
#
#valueDf.tail()
#
#
#
## In[37]:
#
#predDf.tail()
#
#
#
## In[38]:
#
#meanDf
#
#

# ### do all variations

# In[39]:

from datetime import datetime

#
## In[40]:
#
#def makeP(pCsv):
#
#    pName = 'pat%s' % (pCsv.split('_pat')[-1].split('.')[0])
#
#    bigDf = pd.DataFrame()
#
#    for myDim in [1, 2]:
#        for myDrop in [True, False]:
#
#            dimStr = ['1d', '2d'][myDim - 1]
#            dropStr = ['full', 'drop'][myDrop]
#            #print myDim, myDrop, datetime.now()
#
#            valueDf, predDf, meanDf = makeAllComputations(
#                pCsv, dims=myDim, drop=myDrop)
#
#            valueDf.to_csv('%s/data/processed/csv/values_%s_%s_%s.csv' %
#                           (supDir, pName, dimStr, dropStr))
#            predDf.to_csv('%s/data/processed/csv/predictions_%s_%s_%s.csv' %
#                          (supDir, pName, dimStr, dropStr))
#
#            meanDf.index = pd.MultiIndex.from_arrays([[dimStr], [dropStr]])
#            bigDf = pd.concat([bigDf, meanDf])
#
#    bigDf.to_csv('%s/data/processed/csv/meanTable_%s.csv' % (supDir, pName))
#
#    return bigDf
#
#

def makeP(pFolder, pName):

    pCsv = '%s/roiLaterality_%s.csv' % (pFolder, pName)

    bigDf = pd.DataFrame()

    for myDim in [2]:
        for myDrop in [False]:

            dimStr = ['1d', '2d'][myDim - 1]
            dropStr = ['full', 'drop'][myDrop]

            valueDf, predDf, meanDf = makeAllComputations(
                pCsv, dims=myDim, drop=myDrop)

            valueDf.to_csv(
                '%s/values_%s_%s_%s.csv' % (pFolder, pName, dimStr, dropStr))
            predDf.to_csv('%s/predictions_%s_%s_%s.csv' % (pFolder, pName,
                                                           dimStr, dropStr))

            meanDf.index = pd.MultiIndex.from_arrays([[dimStr], [dropStr]])
            bigDf = pd.concat([bigDf, meanDf])

    bigDf.to_csv('%s/meanTable_%s.csv' % (pFolder, pName))

    return bigDf


#
## In[42]:
#
#meanDf = makeP(pCsv)
#
#
#
## In[43]:
#
#meanDf
#
#

# ## do this for all patients

# ### collect all patients
#
## In[44]:
#
#my_train = pickle.load(open('../models/my_nest.p', 'rb'))
#my_test = pickle.load(open('../models/my_test.p', 'rb'))
#my_all = my_train + my_test
#len(my_all)
#
#
#
## In[45]:
#
#csvList = [
#    '../data/interim/csv/%s' % x for x in os.listdir('../data/interim/csv/')
#    if x.startswith('roiLaterality_pat')
#]
#csvList.sort()
#
#
#
## In[48]:
#
#def makeDf(csvList,trainOrTest):
#    df = pd.DataFrame()
#    for pat in csvList:
#        for lab in trainOrTest:
#            if lab[-1] in pat:
#                thisDf = pd.DataFrame([pat], index=[[lab[0]], [lab[1]]])
#                df = pd.concat([df, thisDf])
#    df.columns = ['csv']
#
#    df.sort_index(inplace=True)
#    return df
#
#
#
## In[49]:
#
#dfAll = makeDf(csvList,trainOrTest=my_all)
#
#
#
## In[50]:
#
#dfAll.shape
#
#
#
## In[43]:
#
#dfAll.tail()
#
#

# ### run for all patients
#
## In[58]:
#
#doneList = [
#    x.split('meanTable_')[-1].split('.')[0]
#    for x in os.listdir('%s/data/processed/csv/' % supDir)
#    if x.startswith('meanTable_')
#]
#len(doneList)
#
#
#
## In[61]:
#
#for p in dfAll.index:
#    if p[1] not in doneList:
#        pCsv = dfAll.loc[p,'csv']
#        print datetime.now(),pCsv
#        meanDf = makeP(pCsv)
#
#
