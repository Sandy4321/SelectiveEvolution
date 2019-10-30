# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from dask.distributed import Client, progress
from deap import creator, base, tools, algorithms 
import random
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.externals import joblib
#from sklearn.model_selection import GridSearchCV
from dask_searchcv import GridSearchCV

import argparse
import pandas as pd
import json
import os
import errno

    
class SelectiveEvolution:
    
    def __init__(self):
        # self.client = Client(processes=False, threads_per_worker=4,
        #     n_workers=1, memory_limit='32GB')
        self.client = Client(threads_per_worker=4)

    def data_prep(self, df):
        '''
        This function takes for granted that the last column of the pandas
        data set is the dependent variable
        '''
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1:]
        X_tr, X_tst, y_tr, y_tst = train_test_split(
            X,y,test_size=0.2,random_state=np.random.randint(100))

        return X_tr, X_tst, y_tr, y_tst

    def evalOneMax(self, individual):
            return [sum(individual)]

    def genetic_algo(
        self, df=None, list_inputs = None, predictive_algorithm=None, param_grid=None,
        directory_name=None
        ):
        '''
        Genetic algorithm put in place training a random forest powered by
        Dask parallel computing

        Required:

        Returns:
        - list_gini : independent variables that ensure best predictive power 
        '''
        #####
        #SETING UP THE GENETIC ALGORITHM and CALCULATING STARTING POOL (STARTING CANDIDATE POPULATION)
        #####
        X_tr, X_tst, y_tr, y_tst = self.data_prep(df)

        if list_inputs== None:
            list_inputs = X_tr.columns
        
        if predictive_algorithm == 'random_forest':
            rf = RandomForestClassifier(criterion='gini', random_state=0)
            grid_algorithm = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)

        if predictive_algorithm == 'xgboost':
            xgb = GradientBoostingClassifier()
            grid_algorithm = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(list_inputs))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        NPOPSIZE = 10 #RANDOM STARTING POOL SIZE
        population = toolbox.population(n=NPOPSIZE)

        #####
        #ASSESSING GINI ON THE STARTING POOL
        #####
        dic_gini={}
        for i in range(np.shape(population)[0]):
            # TRASLATING DNA INTO LIST OF VARIABLES (1-81)
            var_model = []
            for j in range(np.shape(population)[0]):
                if (population[i])[j]==1:
                    var_model.append(list(list_inputs)[j])
            # ASSESSING GINI INDEX FOR EACH INVIVIDUAL IN THE INITIAL POOL 
                    
            X_train= X_tr.copy()
            Y_train= y_tr.copy()
            output_var = Y_train.columns[0]
            
            ######
            # Insert here (grid_algorithm = ...) the type of model you want to use. In this example I used an ADABoostClassifier & RF
            #####  
            
            with joblib.parallel_backend('dask'):
                print('first loop iterations')
                model=grid_algorithm.fit(X_train,Y_train.values.ravel())   
                Y_predict=model.predict_proba(X_train)
                Y_predict=Y_predict[:,1]
            ######
                    
            ######
            # CHANGE_HERE - START: GINI
            #####                
            fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_predict)
            auc = metrics.auc(fpr, tpr)
            gini_power = abs(2*auc-1)
                
            ######
            # CHANGE_HERE - END
            #####                
            
            gini=str(gini_power)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
            dic_gini[gini]=population[j]   
        list_gini=sorted(dic_gini.keys(),reverse=True)

        #GENETIC ALGORITHM MAIN LOOP - START
        # - ITERATING MANY TIMES UNTIL NO IMPROVMENT HAPPENS IN ORDER TO FIND THE OPTIMAL SET OF CHARACTERISTICS (VARIABLES)

        sum_current_gini=0.0
        sum_current_gini_1=0.0
        sum_current_gini_2=0.0
        first=0    
        OK = 1
        a=0
        while OK:  #REPEAT UNTIL IT DO NOT IMPROVE, AT LEAST A LITLE, THE GINI IN 2 GENERATIONS
            a=a+1
            print ('loop ', a)
            OK=0

            ####
            # GENERATING OFFSPRING - START
            ####
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1) #CROSS-X PROBABILITY = 50%, MUTATION PROBABILITY=10%
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population =toolbox.select(offspring, k=len(population))
            ####
            # GENERATING OFFSPRING - END
            ####

            sum_current_gini_2=sum_current_gini_1
            sum_current_gini_1=sum_current_gini
            sum_current_gini=0.0

            #####
            #ASSESSING GINI ON THE OFFSPRING - START
            #####
            for j in range(np.shape(population)[0]): 
                if population[j] not in dic_gini.values(): 
                    var_model = [] 
                    for i in range(np.shape(population)[0]): 
                        if (population[j])[i]==1:
                            var_model.append(list(list_inputs)[i])
                    
                    X_train=X_tr[var_model]
                    Y_train=y_tr[output_var] # or whatever you have called it in your own data set. 
                                        
                    ######
                    # Same model as before (grid_algorithm = ...) used to predict probabilities of TEST set. 
                    #####            
                    with joblib.parallel_backend('dask'):
                        model=grid_algorithm.fit(X_train,Y_train.values.ravel())   
                        Y_predict=model.predict_proba(X_train)
                        Y_predict=Y_predict[:,1]

                    ######

                    # CHANGE_HERE - START: GINI
                    #####                       
                    fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_predict)
                    auc = metrics.auc(fpr, tpr)
                    gini_power = abs(2*auc-1)
                    
                    # CHANGE_HERE - END
                    #####                       
                
                    gini=str(gini_power)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
                    dic_gini[gini]=population[j]  
            #####
            #ASSESSING GINI ON THE OFFSPRING - END
            #####

            #####
            #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - START
            #####           
            list_gini=sorted(dic_gini.keys(),reverse=True)
            population=[]
            for i in list_gini[:NPOPSIZE]:
                population.append(dic_gini[i])
                gini=float(i.split(';')[0])
                sum_current_gini+=gini
            #####
            #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - END
            #####           
            
            #HAS IT IMPROVED AT LEAST A LITLE THE GINI IN THE LAST 2 GENERATIONS
            print('sum_current_gini=', sum_current_gini, 'sum_current_gini_1=', sum_current_gini_1, 'sum_current_gini_2=', sum_current_gini_2)
            if(sum_current_gini>sum_current_gini_1+0.0001 or sum_current_gini>sum_current_gini_2+0.0001):
                OK=1
        #####
        #GENETIC ALGORITHM MAIN LOOP - END
        #####


        gini_max=list_gini[0]        
        gini=float(gini_max.split(';')[0])
        features=gini_max.split(';')[1]

        ####
        # PRINTING OUT THE LIST OF FEATURES
        #####
        best_vars = []
        f=0
        for i in range(len(features)):
            if features[i]=='1':
                f+=1
                print ('feature ', f, ':', list(list_inputs)[i]) #this is the BEST combination of explanatory varaibles
                best_vars.append(list(list_inputs)[i])
        print ('gini: ', gini)

        try:
            os.makedirs(directory_name)
        except OSError as exc: 
            if exc.errno == errno.EEXIST and os.path.isdir(directory_name):
                pass
        
        best_params = model.best_params_
        
        with open(os.path.join(directory_name, "best_hyper_parameters"), "w") as f:
            json.dump(best_params, f)
    
        with joblib.parallel_backend('dask'):
            if predictive_algorithm == 'random_forest':
                best_params_algorithm = RandomForestClassifier(**best_params)
            if predictive_algorithm == 'xgboost':
                best_params_algorithm = GradientBoostingClassifier(**best_params)

            model=best_params_algorithm.fit(X_tr[best_vars],Y_train.values.ravel())  
            test_predictions=model.predict_proba(X_tst[best_vars])
            self.client.close()
                
        predicted_vs_true = pd.DataFrame({'predictions': pd.Series(test_predictions[:,1])})
        predicted_vs_true['true'] = y_tst.reset_index(drop=True)

        var_importance_df = pd.DataFrame(
            {'variables': best_vars, 'importance':model.feature_importances_})

        return list_gini, var_importance_df, predicted_vs_true
