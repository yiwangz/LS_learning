import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
import STowl_multi as st
import multiprocessing as mp
import os
import math

############################################################################################################
m = 5
n = 1000
nindpt = 5000
d = 10
num_imp = 5

cost_vec = np.array([1/128,1/64,1/32,1/16,1/8,1/4,1/2,1.0])
tuned_paras = [{'C': cost_vec}]

itermax = 50
itertol = 1e-4

studySetting = 'longitudinal'
tuneSetting = 'exponential'
obsSetting = 'observational'

############################################################################################################
def simulation(seed_base):

    np.random.seed(1234+seed_base)

    ########################################################################################################
    ## read in generated simulation datasets by R ##########################################################
    data_indpt = pd.read_csv("".join(["data_indpt",str(seed_base+1),".txt"]))
    data = pd.read_csv("".join(["data",str(seed_base+1),".txt"]))
    data_indpt = data_indpt.to_numpy()
    data = data.to_numpy()

    Xindpt = data_indpt[:,:d]
    Tindpt = data_indpt[:,d]

    Xall = data[:,:d]
    Aall = data[:,d]
    Tall = data[:,d+1]
    B_mat = data[:,(d+2):(d+2+m)]
    miss_mat = data[:,(d+2+m):(d+2+m*2)]
    dataLabel = data[:,d+2+m*2]

    valid_index = [item for sublist in np.where(np.isnan(dataLabel) == False) for item in sublist]
    Xall = Xall[valid_index,:]
    Aall = Aall[valid_index]
    Tall = Tall[valid_index]
    B_mat = B_mat[valid_index,:]
    miss_mat = miss_mat[valid_index,:]
    dataLabel = dataLabel[valid_index]
    n_valid = len(dataLabel)
    dataLabel = np.asarray([int(dataLabel[i]) for i in range(n_valid)])

    ########################################################################################################
    ## check the unique values of dataLabel, the labels are ordered
    uniqueIndex = list(set(dataLabel))

    if uniqueIndex[-1] != m-1:
        dresult = dict()
        dresult['cost_tune'] = np.nan
        dresult['par_tune'] = np.nan
        dresult['acc_all_tune'] = np.nan
        dresult['acc_all_1CV'] = np.nan
        dresult['acc_all_allCV'] = np.nan
        dresult['acc_all_impCV'] = np.nan
        dresult['acc_indpt_tune'] = np.nan
        dresult['acc_indpt_1CV'] = np.nan
        dresult['acc_indpt_allCV'] = np.nan
        dresult['acc_indpt_impCV'] = np.nan
        dresult['evf_B_tune'] = np.nan
        dresult['evf_B_1CV'] = np.nan
        dresult['evf_B_allCV'] = np.nan
        dresult['evf_B_impCV'] = np.nan
        dresult['ts_tune'] = np.nan
        dresult['ts_1CV'] = np.nan
        dresult['ts_allCV'] = np.nan
        dresult['ts_impCV'] = np.nan
        dresult['time_1CV'] = np.nan
        dresult['time_allCV'] = np.nan
        dresult['time_impCV'] = np.nan
        dresult['time_tune'] = np.nan
        dresult['n_valid'] = n_valid
    else:
        Ball = np.full(n_valid, np.nan)
        for i in range(n_valid):
            Ball[i] = B_mat[i, dataLabel[i]]

        #########################################################################################################
        ## determine propensity score ###########################################################################
        propenScore = st.propensityScore(Xall, Aall, uniqueIndex, dataLabel)
        pall = propenScore.p(obsSetting=obsSetting)

        #########################################################################################################
        ## OWL on S_m only ######################################################################################
        start_time_1CV = time.time()

        index = [item for sublist in np.where(dataLabel == m-1) for item in sublist]
        Xm = Xall[index,:]
        Bm = Ball[index]
        pm = pall[index]
        Am = Aall[index]

        model1 = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy', fit_params={'sample_weight': Bm/pm})
        model1.fit(Xm, Am)

        time_1CV = time.time()-start_time_1CV

        predAll_model1 = model1.best_estimator_.predict(Xall)
        acc_all_1CV = st.evalPred(predAll_model1, Tall).acc()
        predIndpt_model1 = model1.best_estimator_.predict(Xindpt)
        acc_indpt_1CV = st.evalPred(predIndpt_model1, Tindpt).acc()

        ts1 = st.tuneStat(Xall, Aall, Ball, m, uniqueIndex, dataLabel, model1)
        ts_1CV = ts1.tsSSE(model='linear')

        evf1 = st.EVF()
        evf1.evfCal(Xall, Aall, Ball, uniqueIndex, dataLabel, model1)
        evf_B_1CV = evf1.evfSeq

        ############################################################################################################
        ## OWL on S ################################################################################################
        start_time_allCV = time.time()

        model = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy',fit_params={'sample_weight': Ball/pall})
        model.fit(Xall, Aall)

        time_allCV = time.time()-start_time_allCV

        predAll_model = model.best_estimator_.predict(Xall)
        acc_all_allCV = st.evalPred(predAll_model, Tall).acc()
        predIndpt_model = model.best_estimator_.predict(Xindpt)
        acc_indpt_allCV = st.evalPred(predIndpt_model, Tindpt).acc()

        ts = st.tuneStat(Xall, Aall, Ball, m, uniqueIndex, dataLabel, model)
        ts_allCV = ts.tsSSE(model='linear')

        evf = st.EVF()
        evf.evfCal(Xall, Aall, Ball, uniqueIndex, dataLabel, model)
        evf_B_allCV = evf.evfSeq

        ############################################################################################################
        ## MICE imputation #########################################################################################
        start_time_impCV = time.time()

        acc_all_vec = np.zeros(num_imp)
        acc_indpt_vec = np.zeros(num_imp)
        ts_vec = np.zeros(num_imp)
        evf_B_mat = np.zeros((num_imp, len(uniqueIndex)))

        for i in range(num_imp):
            B_imp = data[:,(d+2+m*(i+2)):(d+2+m*(i+3))]
            B_imp = B_imp[valid_index, :]

            B_shift_imp = np.nanmin(B_imp)
            if B_shift_imp < 0:
                B_imp = B_imp+abs(B_shift_imp)+0.001

            modelimp = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy', fit_params={'sample_weight': B_imp[:,-1]/pall})
            modelimp.fit(Xall, Aall)
            predAll_modelimp = modelimp.best_estimator_.predict(Xall)
            acc_all_vec[i] = st.evalPred(predAll_modelimp, Tall).acc()
            predIndpt_modelimp = modelimp.best_estimator_.predict(Xindpt)
            acc_indpt_vec[i] = st.evalPred(predIndpt_modelimp, Tindpt).acc()

            tsimp = st.tuneStat(Xall, Aall, Ball, m, uniqueIndex, dataLabel, modelimp)
            ts_vec[i] = tsimp.tsSSE(model='linear')

            evfimp = st.EVF()
            evfimp.evfCal(Xall, Aall, Ball, uniqueIndex, dataLabel, modelimp)
            evf_B_mat[i,:] = evfimp.evfSeq

        time_impCV = time.time() - start_time_impCV

        acc_all_impCV = sum(acc_all_vec)/num_imp
        acc_indpt_impCV = sum(acc_indpt_vec)/num_imp
        ts_impCV = sum(ts_vec)/num_imp
        evf_B_impCV = np.mean(evf_B_mat, axis=0)

        ############################################################################################################
        ## SS-learning (super oracle) ##############################################################################
        start_time_super = time.time()

        modelSuper = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy', fit_params={'sample_weight': B_mat[:,-1]/pall})
        modelSuper.fit(Xall, Aall)

        time_super = time.time() - start_time_super

        predAll_modelSuper = modelSuper.best_estimator_.predict(Xall)
        acc_all_super = st.evalPred(predAll_modelSuper, Tall).acc()
        predIndpt_modelSuper = modelSuper.best_estimator_.predict(Xindpt)
        acc_indpt_super = st.evalPred(predIndpt_modelSuper, Tindpt).acc()

        tsSuper = st.tuneStat(Xall, Aall, Ball, m, uniqueIndex, dataLabel, modelSuper)
        ts_super = tsSuper.tsSSE(model='linear')

        evfSuper = st.EVF()
        evfSuper.evfCal(Xall, Aall, Ball, uniqueIndex, dataLabel, modelSuper)
        evf_B_super = evfSuper.evfSeq

        ############################################################################################################
        ## proposed method #########################################################################################

        ## let lambda be an exponential function a*exp(b*t/T), aexp(b)=1, a in (0,1]
        if tuneSetting == 'linear':
            par_vec = np.arange(0, 1.1, 0.1)
            lam_mat = np.full([len(par_vec), len(uniqueIndex)], np.nan)
            for ll in range(len(par_vec)):
                for kk in range(len(uniqueIndex)):
                    lam_mat[ll, kk] = 1-par_vec[ll]+par_vec[ll]*(uniqueIndex[kk]+1)/m
        elif tuneSetting == 'exponential':
            par_vec = np.arange(0.1, 1.1, 0.1)
            lam_mat = np.full([len(par_vec), len(uniqueIndex)], np.nan)
            for ll in range(len(par_vec)):
                for kk in range(len(uniqueIndex)):
                    lam_mat[ll,kk] = par_vec[ll]*np.exp((math.log(1/par_vec[ll]))*(uniqueIndex[kk]+1)/m)

        conv = np.full([cost_vec.shape[0], lam_mat.shape[0]], np.nan)  ##matrix
        obj_conv = np.full([cost_vec.shape[0], lam_mat.shape[0]], np.nan)  ##matrix
        ts_conv = np.full([cost_vec.shape[0], lam_mat.shape[0]], np.nan)  ##matrix
        evf_B_conv = np.full([cost_vec.shape[0], lam_mat.shape[0], len(uniqueIndex)], np.nan)  ##matrix
        acc_all_conv = np.full([cost_vec.shape[0], lam_mat.shape[0]], np.nan) ##matrix
        acc_indpt_conv = np.full([cost_vec.shape[0], lam_mat.shape[0]], np.nan) ##matrix

        start_time_tune = time.time()

        out = st.STowlLinear(Xall, Aall, Ball, B_mat, miss_mat, n_valid, m, uniqueIndex, dataLabel, pall)

        for ii in np.arange(cost_vec.shape[0]):
            # initial fitting
            out.iniFit(cost_vec[ii], study=studySetting)

            for jj in np.arange(lam_mat.shape[0]):

                lam = lam_mat[jj,:]
                out.fit(lam, itermax, itertol, track=True)

                conv[ii,jj] = out.conv

                if(out.conv != 99):
                    obj_conv[ii,jj] = out.objConv
                    ts_conv[ii,jj] = out.tsConv
                    acc_all_conv[ii,jj] = st.evalPred(out.predConv, Tall).acc()
                    predIndpt = out.predict(Xindpt, track=False)
                    acc_indpt_conv[ii,jj] = st.evalPred(predIndpt, Tindpt).acc()
                    evfConv = st.EVF()
                    evfConv.evfCal(Xall, Aall, Ball, uniqueIndex, dataLabel, out.modelConv)
                    evf_B_conv[ii,jj,:] = evfConv.evfSeq
                else:
                    obj_conv[ii,jj] = 1e5
                    ts_conv[ii,jj] = 0
                    acc_all_conv[ii,jj] = 0
                    acc_indpt_conv[ii,jj] = 0
                    evf_B_conv[ii,jj,:] = 0

        time_tune = time.time()-start_time_tune

        opt_index = st.evalTune()
        opt_index.maxTune(ts_conv, acc_all_conv, method='min')
        cost_tune_idx = opt_index.tune_idx[0]
        par_tune_idx = opt_index.tune_idx[1]

        cost_tune = cost_vec[cost_tune_idx]
        par_tune = par_vec[par_tune_idx]
        acc_all_tune = acc_all_conv[cost_tune_idx, par_tune_idx]
        acc_indpt_tune = acc_indpt_conv[cost_tune_idx, par_tune_idx]
        ts_tune = ts_conv[cost_tune_idx, par_tune_idx]
        evf_B_tune = evf_B_conv[cost_tune_idx, par_tune_idx,:]


        ################################################################################################
        dresult = dict()
        dresult['cost_tune'] = cost_tune
        dresult['par_tune'] = par_tune
        dresult['acc_all_tune'] = acc_all_tune
        dresult['acc_all_1CV'] = acc_all_1CV
        dresult['acc_all_allCV'] = acc_all_allCV
        dresult['acc_all_impCV'] = acc_all_impCV
        dresult['acc_all_super'] = acc_all_super
        dresult['acc_indpt_tune'] = acc_indpt_tune
        dresult['acc_indpt_1CV'] = acc_indpt_1CV
        dresult['acc_indpt_allCV'] = acc_indpt_allCV
        dresult['acc_indpt_impCV'] = acc_indpt_impCV
        dresult['acc_indpt_super'] = acc_indpt_super
        dresult['evf_B_tune'] = evf_B_tune
        dresult['evf_B_1CV'] = evf_B_1CV
        dresult['evf_B_allCV'] = evf_B_allCV
        dresult['evf_B_impCV'] = evf_B_impCV
        dresult['evf_B_super'] = evf_B_super
        dresult['ts_tune'] = ts_tune
        dresult['ts_1CV'] = ts_1CV
        dresult['ts_allCV'] = ts_allCV
        dresult['ts_impCV'] = ts_impCV
        dresult['ts_super'] = ts_super
        dresult['time_1CV'] = time_1CV
        dresult['time_allCV'] = time_allCV
        dresult['time_impCV'] = time_impCV
        dresult['time_super'] = time_super
        dresult['time_tune'] = time_tune
        dresult['n_valid'] = n_valid

    return(dresult)

############################################################################################################
if __name__ == '__main__':
    ncpus = 1
    pool = mp.Pool(processes=ncpus)
    len_replicate = ncpus
    slurm_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    slurm_index_str = str(slurm_index)
    results = pool.map(simulation, range(ncpus*(slurm_index - 1), ncpus*slurm_index))

    '''
    pool = mp.Pool(processes=4)
    len_replicate = 1
    results = pool.map(simulation, range(len_replicate))
    '''

    cost_tune = np.row_stack(results[i]['cost_tune'] for i in range(len_replicate))
    par_tune = np.row_stack(results[i]['par_tune'] for i in range(len_replicate))
    acc_all_tune = np.row_stack(results[i]['acc_all_tune'] for i in range(len_replicate))
    acc_all_1CV = np.row_stack(results[i]['acc_all_1CV'] for i in range(len_replicate))
    acc_all_allCV = np.row_stack(results[i]['acc_all_allCV'] for i in range(len_replicate))
    acc_all_impCV = np.row_stack(results[i]['acc_all_impCV'] for i in range(len_replicate))
    acc_all_super = np.row_stack(results[i]['acc_all_super'] for i in range(len_replicate))
    acc_indpt_tune = np.row_stack(results[i]['acc_indpt_tune'] for i in range(len_replicate))
    acc_indpt_1CV = np.row_stack(results[i]['acc_indpt_1CV'] for i in range(len_replicate))
    acc_indpt_allCV = np.row_stack(results[i]['acc_indpt_allCV'] for i in range(len_replicate))
    acc_indpt_impCV = np.row_stack(results[i]['acc_indpt_impCV'] for i in range(len_replicate))
    acc_indpt_super = np.row_stack(results[i]['acc_indpt_super'] for i in range(len_replicate))
    evf_B_tune = np.row_stack(results[i]['evf_B_tune'] for i in range(len_replicate))
    evf_B_1CV = np.row_stack(results[i]['evf_B_1CV'] for i in range(len_replicate))
    evf_B_allCV = np.row_stack(results[i]['evf_B_allCV'] for i in range(len_replicate))
    evf_B_impCV = np.row_stack(results[i]['evf_B_impCV'] for i in range(len_replicate))
    evf_B_super = np.row_stack(results[i]['evf_B_super'] for i in range(len_replicate))
    ts_tune = np.row_stack(results[i]['ts_tune'] for i in range(len_replicate))
    ts_1CV = np.row_stack(results[i]['ts_1CV'] for i in range(len_replicate))
    ts_allCV = np.row_stack(results[i]['ts_allCV'] for i in range(len_replicate))
    ts_impCV = np.row_stack(results[i]['ts_impCV'] for i in range(len_replicate))
    ts_super = np.row_stack(results[i]['ts_super'] for i in range(len_replicate))
    time_1CV = np.row_stack(results[i]['time_1CV'] for i in range(len_replicate))
    time_allCV = np.row_stack(results[i]['time_allCV'] for i in range(len_replicate))
    time_impCV = np.row_stack(results[i]['time_impCV'] for i in range(len_replicate))
    time_super = np.row_stack(results[i]['time_super'] for i in range(len_replicate))
    time_tune = np.row_stack(results[i]['time_tune'] for i in range(len_replicate))
    n_valid = np.row_stack(results[i]['n_valid'] for i in range(len_replicate))


    ### save files ###############################################################
    np.savetxt("cost_tune_"+slurm_index_str+".txt", cost_tune, delimiter=",")
    np.savetxt("par_tune_"+slurm_index_str+".txt", par_tune, delimiter=",")
    np.savetxt("acc_all_tune_"+slurm_index_str+".txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_all_1CV_"+slurm_index_str+".txt", acc_all_1CV, delimiter=",")
    np.savetxt("acc_all_allCV_"+slurm_index_str+".txt", acc_all_allCV, delimiter=",")
    np.savetxt("acc_all_impCV_"+slurm_index_str+".txt", acc_all_impCV, delimiter=",")
    np.savetxt("acc_all_super_"+slurm_index_str+".txt", acc_all_super, delimiter=",")
    np.savetxt("acc_indpt_tune_"+slurm_index_str+".txt", acc_indpt_tune, delimiter=",")
    np.savetxt("acc_indpt_1CV_"+slurm_index_str+".txt", acc_indpt_1CV, delimiter=",")
    np.savetxt("acc_indpt_allCV_"+slurm_index_str+".txt", acc_indpt_allCV, delimiter=",")
    np.savetxt("acc_indpt_impCV_"+slurm_index_str+".txt", acc_indpt_impCV, delimiter=",")
    np.savetxt("acc_indpt_super_"+slurm_index_str+".txt", acc_indpt_super, delimiter=",")
    np.savetxt("evf_B_tune_" + slurm_index_str + ".txt", evf_B_tune, delimiter=",")
    np.savetxt("evf_B_1CV_" + slurm_index_str + ".txt", evf_B_1CV, delimiter=",")
    np.savetxt("evf_B_allCV_" + slurm_index_str + ".txt", evf_B_allCV, delimiter=",")
    np.savetxt("evf_B_impCV_" + slurm_index_str + ".txt", evf_B_impCV, delimiter=",")
    np.savetxt("evf_B_super_" + slurm_index_str + ".txt", evf_B_super, delimiter=",")
    np.savetxt("ts_tune_"+slurm_index_str+".txt", ts_tune, delimiter=",")
    np.savetxt("ts_1CV_" + slurm_index_str + ".txt", ts_1CV, delimiter=",")
    np.savetxt("ts_allCV_" + slurm_index_str + ".txt", ts_allCV, delimiter=",")
    np.savetxt("ts_impCV_" + slurm_index_str + ".txt", ts_impCV, delimiter=",")
    np.savetxt("ts_super_" + slurm_index_str + ".txt", ts_super, delimiter=",")
    np.savetxt("time_1CV_"+slurm_index_str+".txt", time_1CV, delimiter=",")
    np.savetxt("time_allCV_"+slurm_index_str+".txt", time_allCV, delimiter=",")
    np.savetxt("time_impCV_"+slurm_index_str+".txt", time_impCV, delimiter=",")
    np.savetxt("time_super_"+slurm_index_str+".txt", time_super, delimiter=",")
    np.savetxt("time_tune_"+slurm_index_str+".txt", time_tune, delimiter=",")
    np.savetxt("n_valid_"+slurm_index_str+".txt", n_valid, delimiter=",")

    '''
    np.savetxt("cost_tune.txt", cost_tune, delimiter=",")
    np.savetxt("par_tune.txt", par_tune, delimiter=",")
    np.savetxt("acc_all_tune.txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_all_1CV.txt", acc_all_1CV, delimiter=",")
    np.savetxt("acc_all_allCV.txt", acc_all_allCV, delimiter=",")
    np.savetxt("acc_all_impCV.txt", acc_all_impCV, delimiter=",")
    np.savetxt("acc_all_super.txt", acc_all_super, delimiter=",")
    np.savetxt("acc_indpt_tune.txt", acc_indpt_tune, delimiter=",")
    np.savetxt("acc_indpt_1CV.txt", acc_indpt_1CV, delimiter=",")
    np.savetxt("acc_indpt_allCV.txt", acc_indpt_allCV, delimiter=",")
    np.savetxt("acc_indpt_impCV.txt", acc_indpt_impCV, delimiter=",")
    np.savetxt("acc_indpt_super.txt", acc_indpt_super, delimiter=",")
    np.savetxt("evf_B_tune.txt", evf_B_tune, delimiter=",")
    np.savetxt("evf_B_1CV.txt", evf_B_1CV, delimiter=",")
    np.savetxt("evf_B_allCV.txt", evf_B_allCV, delimiter=",")
    np.savetxt("evf_B_impCV.txt", evf_B_impCV, delimiter=",")
    np.savetxt("evf_B_super.txt", evf_B_super, delimiter=",")
    np.savetxt("ts_tune.txt", ts_tune, delimiter=",")
    np.savetxt("ts_1CV.txt", ts_1CV, delimiter=",")
    np.savetxt("ts_allCV.txt", ts_allCV, delimiter=",")
    np.savetxt("ts_impCV.txt", ts_impCV, delimiter=",")
    np.savetxt("ts_super.txt", ts_super, delimiter=",")
    np.savetxt("time_1CV.txt", time_1CV, delimiter=",")
    np.savetxt("time_allCV.txt", time_allCV, delimiter=",")
    np.savetxt("time_impCV.txt", time_impCV, delimiter=",")
    np.savetxt("time_super.txt", time_super, delimiter=",")
    np.savetxt("time_tune.txt", time_tune, delimiter=",")
    np.savetxt("n_valid.txt", n_valid, delimiter=",")
    '''

