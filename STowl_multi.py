import numpy as np
from sklearn import svm
#from sklearn import linear_model
#from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from pygam import GAM, LinearGAM, s, l
#from patsy import dmatrix
#import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

########################################################################################################################
########################################################################################################################
## calculate propensity score
class propensityScore:
    def __init__(self, Xall, Aall, uniqueIndex, dataLabel):
        self.Xall = Xall
        self.Aall = Aall
        self.uniqueIndex = uniqueIndex
        self.dataLabel = dataLabel

    def p(self, obsSetting = 'trial'):
        if obsSetting == 'trial':
            pall = np.full(self.Aall.shape[0], 0.5)
        elif obsSetting == 'observational':
            pall = []

            for i in self.uniqueIndex:
                index = [item for sublist in np.where(self.dataLabel == i) for item in sublist] #[l for l, x in enumerate(self.dataLabel) if x == i]
                pvec = np.zeros(len(index))
                Xtemp = self.Xall[index,:]
                Atemp = self.Aall[index]
                logReg = LogisticRegression()
                logReg.fit(Xtemp[:,:], Atemp)

                index1 = [item for sublist in np.where(Atemp == 1) for item in sublist]
                index0 = [item for sublist in np.where(Atemp == -1) for item in sublist]

                pvec[index1] = logReg.predict_proba(Xtemp)[index1, 1]
                pvec[index0] = logReg.predict_proba(Xtemp)[index0, 0]

                pall = np.concatenate((pall, pvec))

        return pall


########################################################################################################################
########################################################################################################################
## Model selection (parameter tuning).
class tuneStat:
    def __init__(self, Xall, Aall, Ball, m, uniqueIndex, dataLabel, model):
        self.Xall = Xall
        self.Aall = Aall
        self.Ball = Ball
        self.m = m
        self.uniqueIndex = uniqueIndex
        self.dataLabel = dataLabel
        self.model = model

    def tsSSE(self, model='linear'):

        sse = 0

        for i in self.uniqueIndex:

            index = [item for sublist in np.where(self.dataLabel == i) for item in sublist] #[l for l, x in enumerate(self.dataLabel) if x == i]
            Xfit = self.Xall[index,:]
            Afit = self.Aall[index]
            Bfit = self.Ball[index]

            Af = Afit * self.model.decision_function(Xfit)
            Xmat = np.column_stack((Xfit, Af))

            if model == 'linear':
                Xmat = sm.add_constant(Xmat)
                BModel = sm.OLS(Bfit, Xmat)
                res = BModel.fit()
                pred = res.predict()
            elif model == 'GAM':
                BModel = LinearGAM(fit_intercept=True)
                res = BModel.fit(Xmat, Bfit)
                pred = res.predict(Xmat)

            sse = sse + sum([(Bfit[elem]-pred[elem])**2 for elem in range(len(Bfit))])

        return sse

class tuneStat_longitudinal:
    def __init__(self, Xall, Aall, Ball, B_mat, miss_mat, n_valid, m, uniqueIndex, dataLable, model):
        self.Xall = Xall
        self.Aall = Aall
        self.Ball = Ball
        self.B_mat = B_mat
        self.miss_mat = miss_mat
        self.n_valid = n_valid
        self.m = m
        self.uniqueIndex = uniqueIndex
        self.dataLabel = dataLable
        self.model = model

    def tsSSE_multiple(self):

        d = self.Xall.shape[1]

        fit_mat = np.full([int(self.n_valid*self.m-np.sum(self.miss_mat)),int(5+d*2)], np.nan)

        B_obs = np.copy(self.B_mat)
        B_obs[self.miss_mat == 1] = np.nan

        fit_index = 0

        for i in range(self.n_valid):
            for j in range(self.m):
                if not np.isnan(B_obs[i,j]):
                    fit_mat[fit_index,0] = i
                    fit_mat[fit_index,1] = 1
                    fit_mat[fit_index,2] = (j+1)/self.m
                    fit_mat[fit_index,3:(3+d)] = self.Xall[i,:]
                    fit_mat[fit_index,(3+d):(3+d*2)] = self.Xall[i,:]*(j+1)/self.m
                    fit_mat[fit_index,(3+d*2)] = self.Aall[i]*self.model.decision_function(self.Xall[i,:])
                    fit_mat[fit_index,(4+d*2)] = B_obs[i,j]
                    fit_index = fit_index+1

        mod = sm.NominalGEE(endog=fit_mat[:,-1],exog=fit_mat[:,1:(4+d*2)],groups=fit_mat[:,0])
        res = mod.fit()
        pred = res.predict(fit_mat[:,1:(4+d*2)])

        sse = sum([(fit_mat[elem,-1] - pred[elem])**2 for elem in range(len(pred))])

        return sse

    def tsSSE_single(self):

        sse = 0

        for i in self.uniqueIndex:

            index = [item for sublist in np.where(self.dataLabel == i) for item in sublist]
            Xfit = self.Xall[index, :]
            Afit = self.Aall[index]
            Bfit = self.Ball[index]
            time = np.repeat((i+1)/self.m,len(index))
            Xt = Xfit*((i+1)/self.m)

            Af = Afit*self.model.decision_function(Xfit)
            Xmat = np.column_stack((Xfit,time,Xt,Af))

            BModel = LinearGAM(l(0)+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7)+l(8)+l(9)+
                               s(10)+s(11)+s(12)+s(13)+s(14)+s(15)+s(16)+s(17)+s(18)+s(19)+s(20)+l(21),fit_intercept=True)
            res = BModel.fit(Xmat, Bfit)  ##the GAM model can be specified differently
            pred = res.predict(Xmat)

            sse = sse + sum([(Bfit[elem]-pred[elem])**2 for elem in range(len(index))])

        return sse


########################################################################################################################
########################################################################################################################
## calculate the estimated value function separately
class EVF:
    def evfCal(self, Xall, Aall, Ball, uniqueIndex, dataLabel, model):

        labelall = model.predict(Xall)

        evfSeq = np.zeros(len(uniqueIndex))

        for i in uniqueIndex:
            index = [item for sublist in np.where(dataLabel == i) for item in sublist] #[l for l, x in enumerate(dataLabel) if x == i]
            Acal = Aall[index]
            Bcal = Ball[index]
            labelcal = labelall[index]

            evfSeq[i] = sum(Bcal*(Acal==labelcal))/sum((Acal==labelcal))

        self.evfSeq = evfSeq


########################################################################################################################
########################################################################################################################
## Self training OWL
class STowlLinear:
    def __init__(self, Xall, Aall, Ball, B_mat, miss_mat, n_valid, m, uniqueIndex, dataLabel, pall):
        self.Xall = Xall
        self.Aall = Aall
        self.Ball = Ball
        self.B_mat = B_mat
        self.miss_mat = miss_mat
        self.n_valid = n_valid
        self.m = m
        self.uniqueIndex = uniqueIndex
        self.dataLabel = dataLabel
        self.pall = pall

    def iniFit(self, cost, study='single'):

        if study == 'single':
            index = []
            for i in range(self.m):
                index = np.concatenate((index, [item for sublist in np.where(self.dataLabel == i) for item in sublist]))

            index1 = [item for sublist in np.where(self.dataLabel == 0) for item in sublist]
            indexRemain = index[len(index1):]
            indexRemain = [int(indexRemain[elem]) for elem in range(len(indexRemain))]

        elif study == 'longitudinal':
            index = []
            for i in self.uniqueIndex:
                index = np.concatenate((index, [item for sublist in np.where(self.dataLabel == i) for item in sublist]))

            index1 = [item for sublist in np.where(self.dataLabel == self.m-1) for item in sublist]
            indexRemain = index[:(len(index)-len(index1))]
            indexRemain = [int(indexRemain[elem]) for elem in range(len(indexRemain))]

        X1 = self.Xall[index1, :]
        A1 = self.Aall[index1]
        B1 = self.Ball[index1]
        p1 = self.pall[index1]
        dataLabel1 = self.dataLabel[index1]

        Xremain = self.Xall[indexRemain, :]  ##ordered
        Aremain = self.Aall[indexRemain]  ##ordered
        Bremain = self.Ball[indexRemain]  ##ordered
        premain = self.pall[indexRemain]  ##ordered
        dataLabelremain = self.dataLabel[indexRemain]

        model_ini = svm.SVC(kernel='linear', C=cost, decision_function_shape="ovo")
        model_ini.fit(X1, A1, sample_weight=B1/p1)
        pred_ini = model_ini.predict(Xremain)

        self.pred_ini = pred_ini
        self.cost = cost
        self.X1 = X1
        self.A1 = A1
        self.B1 = B1
        self.p1 = p1
        self.Xremain = Xremain
        self.Aremain = Aremain
        self.Bremain = Bremain
        self.premain = premain
        self.study = study
        self.dataLabel1 = dataLabel1
        self.dataLabelremain = dataLabelremain

    ## set itertol=0, track=True to perform convergence analysis
    def fit(self, lam, itermax=50, itertol=1e-4, track=True):
        if not hasattr(self, 'pred_ini'):
            print('Run iniFit() for initial predicted labels!')
        else:
            iter = 0
            rel_obj = 1
            obj_old = 1

            if self.m == 2:
                lam = [lam]

            predRemainK = np.copy(self.pred_ini)

            X_aux = np.concatenate((self.X1, self.Xremain, self.Xremain), axis=0)
            B_aux = np.concatenate((self.B1/self.p1, self.Bremain/self.premain))
            wts_aux = np.repeat(1, self.A1.shape[0])

            if self.study == 'single':
                index = []
                for i in range(1,self.m):
                    index.append([item for sublist in np.where(self.dataLabelremain == i) for item in sublist]) #[l for l, x in enumerate(self.dataLabel) if x == i]

                for j in range(self.m-1):
                    Btemp = self.Bremain[index[j]]
                    ptemp = self.premain[index[j]]
                    B_aux = np.concatenate((B_aux, np.repeat(np.average(Btemp/ptemp, axis=0), len(index[j]))))
                    wts_aux = np.concatenate((wts_aux, np.repeat(lam[j], len(index[j]))))

                for k in range(self.m-1):
                    wts_aux = np.concatenate((wts_aux, np.repeat(1-lam[j], len(index[j]))))

                wts_aux = wts_aux*B_aux

            elif self.study == 'longitudinal':
                uniqueIndex_new = self.uniqueIndex.copy()
                uniqueIndex_new.remove(self.m-1)

                index = []
                for i in uniqueIndex_new:
                    index.append([item for sublist in np.where(self.dataLabelremain == i) for item in sublist]) #[l for l, x in enumerate(self.dataLabel) if x == i]

                for j in range(len(uniqueIndex_new)):
                    Btemp = self.Bremain[index[j]]
                    ptemp = self.premain[index[j]]
                    B_aux = np.concatenate((B_aux, np.repeat(np.average(Btemp/ptemp, axis=0), len(index[j]))))
                    wts_aux = np.concatenate((wts_aux, np.repeat(lam[j], len(index[j]))))

                for k in range(len(uniqueIndex_new)):
                    wts_aux = np.concatenate((wts_aux, np.repeat(1-lam[k], len(index[k]))))

                wts_aux = wts_aux*B_aux


            if track:
                obj_path = np.full(itermax, np.nan)
                pred_path = np.full([itermax, (self.Aall.shape[0])], np.nan)
                model_path = []
                ts_path = np.full(itermax, np.nan)


            while (iter < itermax and rel_obj >= itertol):

                A_aux = np.concatenate((self.A1, self.Aremain, predRemainK))  ##this is wrong

                modelk = svm.SVC(kernel='linear', C=self.cost, decision_function_shape="ovo")
                modelk.fit(X_aux, A_aux, sample_weight=wts_aux)  ##this is wrong

                xi_vec = 1-A_aux*modelk.decision_function(X_aux)
                xi_vec[xi_vec < 0] = 0

                obj_new = np.sum(np.power(modelk.coef_,2))*(1/2)+np.sum(xi_vec*wts_aux)*self.cost
                rel_obj = abs(obj_new-obj_old)/obj_old

                obj_old = np.copy(obj_new)
                predRemainK = modelk.predict(self.Xremain)

                tuning_stat = tuneStat_longitudinal(self.Xall, self.Aall, self.Ball, self.B_mat, self.miss_mat,
                                                    self.n_valid, self.m, self.uniqueIndex, self.dataLabel, modelk)
                ts_stat = tuning_stat.tsSSE_single()

                predk = modelk.predict(self.Xall)

                if track:
                    obj_path[iter] = obj_new
                    pred_path[iter,:] = predk
                    model_path.append(modelk)
                    ts_path[iter] = ts_stat

                iter += 1

            if iter >= itermax:
                conv = 99
            else:
                conv = iter

            self.conv = conv
            self.objConv = obj_new
            self.predConv = predk
            self.modelConv = modelk
            self.tsConv = ts_stat

            if track:
                self.objPath = obj_path
                self.predPath = pred_path
                self.modelPath = model_path
                self.tsPath = ts_path

    ## predictions on new datasets
    def predict(self, Xindpt, track):
        ## results based on final model
        if not track:
            if not hasattr(self, 'modelConv'):
                print('Please run fit() first.')
            else:
                return self.modelConv.predict(Xindpt)  ##nindpt*1

        ## results based on the whole path of models (multi-D predicted labels)
        else:
            if not hasattr(self, 'modelPath'):
                print('Please run fit() with track = True.')
            else:
                predOut = []
                for model in self.modelPath:
                    predOut.append(model.predict(Xindpt))
                return np.asarray(predOut)  ##nindpt*conv


########################################################################################################################
########################################################################################################################
## calculate the norm of RKHS for nonlinear kernel
class RKHSnorm:
    def __init__(self, sv, svIndex, lagMulti, gamma):
        self.sv = sv  ##support vectors
        self.svIndex = svIndex  ##index of support vectors
        self.lagMulti = lagMulti  ##Lagrange Multiplier
        self.gamma = gamma

    def omegaNorm(self):

        omega_penalty = np.zeros(len(self.svIndex)**2)

        for i in range(len(self.svIndex)):
            for j in range(len(self.svIndex)):
                omega_penalty[i*len(self.svIndex)+j] = self.lagMulti[0,i]*self.lagMulti[0,j]*np.exp(-self.gamma*np.sum(np.power(self.sv[i]-self.sv[j],2)))

        norm_result = 0.5*np.sum(omega_penalty)

        return norm_result


########################################################################################################################
########################################################################################################################
## Self training OWL
class STowlNonlinear:
    def __init__(self, Xall, Aall, Ball, B_mat, miss_mat, n_valid, m, uniqueIndex, dataLabel, pall):
        self.Xall = Xall
        self.Aall = Aall
        self.Ball = Ball
        self.B_mat = B_mat
        self.miss_mat = miss_mat
        self.n_valid = n_valid
        self.m = m
        self.uniqueIndex = uniqueIndex
        self.dataLabel = dataLabel
        self.pall = pall

    def iniFit(self, cost, gamma, study='single'):

        if study == 'single':
            index = []
            for i in range(self.m):
                index = np.concatenate((index, [item for sublist in np.where(self.dataLabel == i) for item in sublist]))

            index1 = [item for sublist in np.where(self.dataLabel == 0) for item in sublist]
            indexRemain = index[len(index1):]
            indexRemain = [int(indexRemain[elem]) for elem in range(len(indexRemain))]

        elif study == 'longitudinal':
            index = []
            for i in self.uniqueIndex:
                index = np.concatenate((index, [item for sublist in np.where(self.dataLabel == i) for item in sublist]))

            index1 = [item for sublist in np.where(self.dataLabel == self.m-1) for item in sublist]
            indexRemain = index[:(len(index)-len(index1))]
            indexRemain = [int(indexRemain[elem]) for elem in range(len(indexRemain))]

        X1 = self.Xall[index1, :]
        A1 = self.Aall[index1]
        B1 = self.Ball[index1]
        p1 = self.pall[index1]
        dataLabel1 = self.dataLabel[index1]

        Xremain = self.Xall[indexRemain, :]  ##ordered
        Aremain = self.Aall[indexRemain]  ##ordered
        Bremain = self.Ball[indexRemain]  ##ordered
        premain = self.pall[indexRemain]  ##ordered
        dataLabelremain = self.dataLabel[indexRemain]

        model_ini = svm.SVC(kernel='rbf', C=cost, gamma=gamma, decision_function_shape="ovo")
        model_ini.fit(X1, A1, sample_weight=B1/p1)
        pred_ini = model_ini.predict(Xremain)

        self.pred_ini = pred_ini
        self.cost = cost
        self.gamma = gamma
        self.X1 = X1
        self.A1 = A1
        self.B1 = B1
        self.p1 = p1
        self.Xremain = Xremain
        self.Aremain = Aremain
        self.Bremain = Bremain
        self.premain = premain
        self.study = study
        self.dataLabel1 = dataLabel1
        self.dataLabelremain = dataLabelremain

    ## set itertol=0, track=True to perform convergence analysis
    def fit(self, lam, itermax=50, itertol=1e-4, track=True):
        if not hasattr(self, 'pred_ini'):
            print('Run iniFit() for initial predicted labels!')
        else:
            predRemainK = np.copy(self.pred_ini)

            X_aux = np.concatenate((self.X1, self.Xremain, self.Xremain), axis=0)
            B_aux = np.concatenate((self.B1 / self.p1, self.Bremain / self.premain))
            wts_aux = np.repeat(1, self.A1.shape[0])

            if self.study == 'single':
                index = []
                for i in range(1, self.m):
                    index.append([item for sublist in np.where(self.dataLabelremain == i) for item in
                                  sublist])  # [l for l, x in enumerate(self.dataLabel) if x == i]

                for j in range(self.m - 1):
                    Btemp = self.Bremain[index[j]]
                    ptemp = self.premain[index[j]]
                    B_aux = np.concatenate((B_aux, np.repeat(np.average(Btemp / ptemp, axis=0), len(index[j]))))
                    wts_aux = np.concatenate((wts_aux, np.repeat(lam[j], len(index[j]))))

                for k in range(self.m - 1):
                    wts_aux = np.concatenate((wts_aux, np.repeat(1 - lam[j], len(index[j]))))

                wts_aux = wts_aux * B_aux

            elif self.study == 'longitudinal':
                uniqueIndex_new = self.uniqueIndex.copy()
                uniqueIndex_new.remove(self.m - 1)

                index = []
                for i in uniqueIndex_new:
                    index.append([item for sublist in np.where(self.dataLabelremain == i) for item in
                                  sublist])  # [l for l, x in enumerate(self.dataLabel) if x == i]

                for j in range(len(uniqueIndex_new)):
                    Btemp = self.Bremain[index[j]]
                    ptemp = self.premain[index[j]]
                    B_aux = np.concatenate((B_aux, np.repeat(np.average(Btemp / ptemp, axis=0), len(index[j]))))
                    wts_aux = np.concatenate((wts_aux, np.repeat(lam[j], len(index[j]))))

                for k in range(len(uniqueIndex_new)):
                    wts_aux = np.concatenate((wts_aux, np.repeat(1 - lam[k], len(index[k]))))

                wts_aux = wts_aux * B_aux

            if track:
                obj_path = np.full(itermax, np.nan)
                pred_path = np.full([itermax, (self.Aall.shape[0])], np.nan)
                model_path = []
                ts_path = np.full(itermax, np.nan)


            while (iter < itermax and rel_obj >= itertol):

                A_aux = np.concatenate((self.A1, self.Aremain, predRemainK))

                modelk = svm.SVC(kernel='rbf', C=self.cost, gamma=self.gamma, decision_function_shape="ovo")
                modelk.fit(X_aux, A_aux, sample_weight=wts_aux)

                lagMulti = modelk.dual_coef_  ## Lagrange multiplier times label y. Summation = 0.
                svIndex = modelk.support_
                sv = modelk.support_vectors_
                baseObj = RKHSnorm(sv, svIndex, lagMulti, self.gamma)
                rkhsNorm = baseObj.omegaNorm()
                xi_vec = 1-A_aux*modelk.decision_function(X_aux)
                xi_vec[xi_vec < 0] = 0

                obj_new = rkhsNorm+np.sum(xi_vec*wts_aux)*self.cost
                rel_obj = abs(obj_new - obj_old)/obj_old

                obj_old = np.copy(obj_new)
                predRemainK = modelk.predict(self.Xremain)

                tuning_stat = tuneStat_longitudinal(self.Xall, self.Aall, self.Ball, self.B_mat, self.miss_mat,
                                                    self.n_valid, self.m, self.uniqueIndex, self.dataLabel, modelk)
                ts_stat = tuning_stat.tsSSE_single()

                predk = modelk.predict(self.Xall)

                if track:
                    obj_path[iter] = obj_new
                    pred_path[iter,:] = predk
                    model_path.append(modelk)
                    ts_path[iter] = ts_stat

                iter += 1

            if iter >= itermax:
                conv = 99
            else:
                conv = iter

            self.conv = conv
            self.objConv = obj_new
            self.predConv = predk
            self.modelConv = modelk
            self.tsConv = ts_stat

            if track:
                self.objPath = obj_path
                self.predPath = pred_path
                self.modelPath = model_path
                self.tsPath = ts_path

    ## predictions on new datasets
    def predict(self, Xindpt, track):
        ## results based on final model
        if not track:
            if not hasattr(self, 'modelConv'):
                print('Please run fit() first.')
            else:
                return self.modelConv.predict(Xindpt)  ##nindpt*1

        ## results based on the whole path of models (multi-D predicted labels)
        else:
            if not hasattr(self, 'modelPath'):
                print('Please run fit() with track = True.')
            else:
                predOut = []
                for model in self.modelPath:
                    predOut.append(model.predict(Xindpt))
                return np.asarray(predOut)  ##nindpt*conv

########################################################################################################################
########################################################################################################################
## Evaluate prediction performance (accuracy).
class evalPred:

    def __init__(self, predLabel, trueLabel):
        self.pred = predLabel
        self.true = trueLabel

    def acc(self):
        # only one set of predicted labels (e.g., conv)
        if self.pred.ndim == 1:
            return np.sum(self.pred == self.true)/self.true.shape[0]

        # multiple rows of predicted labels (e.g. path)
        else:
            acc_vec = np.full(self.pred.shape[0], np.nan)
            for ii in np.arange(self.pred.shape[0]):
                acc_vec[ii] = np.sum(self.pred[ii, :] == self.true)/ self.true.shape[0]
            return(acc_vec)


########################################################################################################################
########################################################################################################################
## Evaluate tuning process.
class evalTune:
    def maxTune(self, ts_array, acc_array, method='max'):

        dim = len(ts_array.shape)

        opt_idx = []
        tune_idx = []

        for i in range(dim):
            opt_idx.append([np.where(acc_array == np.amax(acc_array))[i][0]])

        if method == 'max':
            for j in range(dim):
                tune_idx.append([np.where(ts_array == np.amax(ts_array))[j][0]])
        elif method == 'min':
            for j in range(dim):
                tune_idx.append([np.where(ts_array == np.amin(ts_array))[j][0]])

        self.tune_idx = tune_idx
        self.opt_idx = opt_idx


