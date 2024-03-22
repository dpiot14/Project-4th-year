# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:49:59 2024

@author: piotd
"""
import pandas as pd
import numpy as np

from utility_tools import string_to_number_and_string

#Fonction pour calculer la tendance et la saisonnalité d'une série temporelle
#sous format dataframe avec les jours comme index (la variable data)
#Renvoie la tendance(float) puis la saisonnalité(np.array)
#Paramètre methode_tend : choix de la méthode pour calculer la tendance
#'mean' : valeur moyenne sur la série (tendance si on suppose la série stationnaire) (valeur par défaut)
#'mobileI_d' : moyenne mobile sur I jours pour calculer la tendance
#Expl : pour la moyenne mobile sur 30 jours, entrer 'mobile30_d'
#Paramètre methode_saison : choix de la méthode pour calculer la saisonnalite
#'mobile28_d' : moyenne mobile sur 28 jours de la valeur moyenne par jour (valeur par défaut)
#Les méthodes 'mobile14_d' et 'mobile7_d' existent aussi
#'mean' : valeur moyenne sur chaque jour de l'année, sans moyenne mobile
def Etude_Tendance_Saisonnalite_annuelle(data, methode_tend='mean',methode_saison='mobile28_d'):
    data_copy=data.copy()
    
    ## -- Calcul de la tendance --
    if methode_tend=='mean':
        tendance=data_copy['electricity'].mean()
    elif methode_tend[:6]=='mobile':   #Si on souhaite utiliser une moyenne mobile
        int_mobile,mode_mobile=string_to_number_and_string(methode_tend[6:])
        if mode_mobile=='_d':
            tendance=data_copy['electricity'].rolling(int_mobile, center=True).mean()
        else:
            print("Erreur : mauvais argument pour methode_tend : la méthode mobile ne comprends pas "+mode_mobile+"\nVoir la documentation") 
        
    else:
        print("Erreur : Methode pour la tendance inconnue, essayez avec une autre valeur pour 'methode_tend'")
    
    ## -- Calcul de la saisonnalité
    data_copy['electricity']=data_copy['electricity']-tendance
    data_copy['day']=data_copy.index.day
    data_copy['month']=data_copy.index.month
    data_year_wind_tendance = data_copy.groupby(['month', 'day']).agg({'electricity': 'mean', 'wind_speed': 'mean'})
    
    if methode_saison[:6]=='mobile': #Si on souhaite utiliser une moyenne mobile
        int_mobile,mode_mobile=string_to_number_and_string(methode_tend[6:])
        if mode_mobile=='_d':
            saison=pd.concat([data_year_wind_tendance['electricity'][(366-int_mobile):],data_year_wind_tendance['electricity'],data_year_wind_tendance['electricity'][:(int_mobile)]]).rolling(int_mobile, center=True).mean().to_numpy()[(int_mobile):(int_mobile+366)]
        else:
            print("Erreur : mauvais argument pour methode_saison : la méthode mobile ne comprends pas "+mode_mobile+"\nVoir la documentation") 
        
    elif methode_saison=='mean':
        saison=data_year_wind_tendance
    else:
        print("Erreur : Methode pour la saisonnalité inconnue, essayez avec une autre valeur pour 'methode_saison'")

    return tendance,saison

