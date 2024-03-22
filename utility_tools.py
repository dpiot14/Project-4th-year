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
#Pour le vent : data_type = wind
#Pour les données météo : data_type = weather
def regroupement_data(link_data,annee_debut,annee_fin,data_type="wind"):
#Création des variables pour vérifier que les données soient toutes différentes
    data_diffrent=True
    same_data=[]
    
    #On initialise les données avec la 1ere année
    data_hour = pd.read_csv(link_data+str(annee_debut)+'.csv', skiprows=3) 
    
    for i in range(annee_debut+1,annee_fin+1):
        new_year=pd.read_csv(link_data+str(i)+'.csv', skiprows=3)
        
        #test qu'on est pas des données de 2 années consécutives qui soient identiques en testant 5 données aléatoires
        test_index=random.randint(-8760,-1)
        
        if data_hour['time'].iloc[test_index]==new_year['time'].iloc[test_index]:
            data_diffrent=False
            same_data=same_data+[str(i-1)+" = "+str(i)]

                                
        data_hour=pd.concat([data_hour,new_year])
    
    if data_diffrent:
        print("Toutes les données sont différentes")
    else:
        print("Certaines données sont identiques :")
        print(same_data)
        
    #on convertit les jours et heures au format datetime de panda
    data_hour['time'] = pd.to_datetime(data_hour['time'])
    
    # Grouper par jour et agréger les données
    if data_type=="wind":
        data_day = data_hour.groupby(data_hour['time'].dt.date).agg({'electricity': 'mean', 'wind_speed': 'mean'})
    elif data_type=="weather":
        #print(data_hour.head())
        data_day = data_hour.groupby(data_hour['time'].dt.date).agg({'t2m': 'mean', 'prectotland': 'mean', 'precsnoland': 'mean', 'snomas': 'mean', 'rhoa': 'mean', 'swgdn': 'mean', 'swtdn': 'mean', 'cldtot': 'mean'})
    data_day.index = pd.to_datetime(data_day.index)
    
    return data_hour,data_day



#Fonction pour découper un string débutant par un nombre entier naturel en 2 : 
#le nombre sous forme de int puis le reste du string. 
#Exemples : string_to_number_and_string("124fjdk") renvoie (124,"fjdk")
#string_to_number_and_string("4MA_1") renvoie (4,"MA_1")
#string_to_number_and_string("GMM13") renvoie (-1,"GMM13") (-1 est la valeur d'erreur)
#Cette fonction est utilisée pour 'Etude_Tendance_Saisonnalite_annuelle' de 
#'methode_etude_serie.py'
def string_to_number_and_string(text):
    number = ''
    rest = ''
    found_number = False
    
    for char in text:
        if char.isdigit() and not found_number:
            number += char
        else:
            found_number = True
            rest += char
    
    try:
        nbr = int(number)
    except ValueError:
        nbr = -1
    
    return nbr, rest
