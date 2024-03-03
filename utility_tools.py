# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:06:45 2024

@author: piotd
"""

import pandas as pd
import random

#Fonction pour regrouper les données téléchargées sur plusieurs années consécutive
#sous un seul dataframe horaire et un dataframe journalié, en vérifiant qu'il
#n'y est pas de doublon pour les données
def regroupement_data(link_data,annee_debut,annee_fin):
#Création des variables pour vérifier que les données soient toutes différentes
    data_diffrent=True
    same_data=[]
    
    #On initialise les données avec la 1ere année
    data_wind_hour = pd.read_csv(link_data+str(annee_debut)+'.csv', skiprows=3) 
    
    for i in range(annee_debut+1,annee_fin+1):
        new_year=pd.read_csv(link_data+str(i)+'.csv', skiprows=3)
        
        #test qu'on est pas des données de 2 années consécutives qui soient identiques en testant 5 données aléatoires
        test_index=[random.randint(-8760,-1),random.randint(-8760,-1),random.randint(-8760,-1),random.randint(-8760,-1),random.randint(-8760,-1)]
        if data_wind_hour['wind_speed'].iloc[test_index[0]]==new_year['wind_speed'].iloc[test_index[0]]:
            if data_wind_hour['wind_speed'].iloc[test_index[1]]==new_year['wind_speed'].iloc[test_index[1]]:
                if data_wind_hour['wind_speed'].iloc[test_index[2]]==new_year['wind_speed'].iloc[test_index[2]]:
                    if data_wind_hour['wind_speed'].iloc[test_index[3]]==new_year['wind_speed'].iloc[test_index[3]]:
                        if data_wind_hour['wind_speed'].iloc[test_index[4]]==new_year['wind_speed'].iloc[test_index[4]]:
                            data_diffrent=False
                            same_data=same_data+[str(i-1)+" = "+str(i)]
                            
        data_wind_hour=pd.concat([data_wind_hour,new_year])
    
    if data_diffrent:
        print("Toutes les données sont différentes")
    else:
        print("Certaines données sont identiques :")
        print(same_data)
        
    #on convertit les jours et heures au format datetime de panda
    data_wind_hour['time'] = pd.to_datetime(data_wind_hour['time'])
    
    # Grouper par jour et agréger les données
    data_wind_day = data_wind_hour.groupby(data_wind_hour['time'].dt.date).agg({'electricity': 'mean', 'wind_speed': 'mean'})
    data_wind_day.index = pd.to_datetime(data_wind_day.index)
    
    return data_wind_hour,data_wind_day