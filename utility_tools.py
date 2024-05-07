# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:06:45 2024

@author: piotd
"""

import pandas as pd
import random
import numpy as np

'''
Fonction pour regrouper les données téléchargées sur plusieurs années consécutive 
sous un seul dataframe horaire et un dataframe journalié, en vérifiant qu'il
n'y est pas de doublon pour les données

    - Pour le vent : data_type = wind
    - Pour les données météo : data_type = weather
    - Pour le solaire : data_type = solar
'''

#A changer  : nom des dossiers (wind/weather/solar)

def regroupement_data(link_data,annee_debut,annee_fin,data_type="wind"):
# Création des variables pour vérifier que les données soient toutes différentes
    data_diffrent=True
    same_data=[]
    
    # On initialise les données avec la 1ere année
    data_hour = pd.read_csv(link_data+str(annee_debut)+'.csv', skiprows=3) 
    
    for i in range(annee_debut+1,annee_fin+1):
        new_year=pd.read_csv(link_data+str(i)+'.csv', skiprows=3)
        
        # Test qu'on est pas des données de 2 années consécutives qui soient identiques en testant 5 données aléatoires
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
        
    # On convertit les jours et heures au format datetime de panda
    data_hour['time'] = pd.to_datetime(data_hour['time'])
    
    # Grouper par jour et agréger les données
    if data_type=="wind":
        data_day = data_hour.groupby(data_hour['time'].dt.date).agg({'electricity': 'mean', 'wind_speed': 'mean'})
    elif data_type=="weather":
        #print(data_hour.head())
        data_day = data_hour.groupby(data_hour['time'].dt.date).agg({'t2m': 'mean', 'prectotland': 'mean', 'precsnoland': 'mean', 'snomas': 'mean', 'rhoa': 'mean', 'swgdn': 'mean', 'swtdn': 'mean', 'cldtot': 'mean'})
    elif data_type=="solar":
        data_day = data_hour.groupby(data_hour['time'].dt.date).agg({'electricity': 'mean', 'irradiance_direct': 'mean', 'irradiance_diffuse': 'mean', 'temperature': 'mean'})
    data_day.index = pd.to_datetime(data_day.index)
    
    return data_hour,data_day


def regroupement_data_all(link_data,annee_debut=[1980,1980,2000],annee_fin=[2023,2023,2023]):
    
    # Ouverture des données de vent, de solaire et météo
    wind_link = link_data+"/Wind/"
    data_wind_hour,data_wind_day = regroupement_data(wind_link,annee_debut[0],annee_fin[0],data_type="wind")
    
    solar_link = link_data+"/Solar/"
    data_solar_hour,data_solar_day = regroupement_data(solar_link,annee_debut[1],annee_fin[1],data_type="solar")
    
    weather_link = link_data+"/Weather/"
    data_weather_hour,data_weather_day = regroupement_data(weather_link,annee_debut[2],annee_fin[2],data_type="weather")
    
    
    #Formatage des données dans un unique dataframe df_hour
    df_hour = data_wind_hour.copy()
    df_hour = df_hour.rename(columns={'electricity': 'wind_electricity'})
    df_hour = df_hour.drop(['local_time'],axis=1)
    df_hour = df_hour.set_index('time')
    df_hour = pd.concat([df_hour,data_solar_hour.set_index('time'),data_weather_hour.set_index('time')],axis=1)
    df_hour = df_hour.rename(columns={'electricity': 'solar_electricity'})
    df_hour = df_hour.drop(['local_time','t2m'],axis=1)
    
    
    #Formatage des données dans un unique dataframe df_day
    df_day = data_wind_day.copy()
    df_day = df_day.rename(columns={'electricity': 'wind_electricity'})
    df_day = pd.concat([df_day,data_solar_day,data_weather_day],axis=1)
    df_day = df_day.rename(columns={'electricity': 'solar_electricity'})
    df_day = df_day.drop(['t2m'],axis=1)
      
    return df_hour,df_day



'''
Fonction pour découper un string débutant par un nombre entier naturel en 2 : 
    - le nombre sous forme de int 
    - le reste du string
Exemples : string_to_number_and_string("124fjdk") renvoie (124,"fjdk")
        string_to_number_and_string("4MA_1") renvoie (4,"MA_1")
        string_to_number_and_string("GMM13") renvoie (-1,"GMM13") (-1 est la valeur d'erreur)

Cette fonction est utilisée pour 'Etude_Tendance_Saisonnalite_annuelle' de 'methode_etude_serie.py'
'''

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


'''
Prend en entrée une série journalière, avec en indice une date au format pandas,
renvoie en sortie une série horaire, avec en indice la date et l'heure au format panda.

Si repartition = None, la valeur du jour dans serie_d est conservée pour les 24 heures
de la journée correspondante
Sinon, repartition doit être un vecteur numpy de taille 24, correspondant au coeff qui doit être
appliqué à chaque heure de la journée
'''

def day_to_hour(serie_d, repartition=None):
    # Créer une série horaire vide
    serie_h = pd.Series()
    
    # Parcourir les éléments de la série journalière
    for date_jour, valeur_jour in serie_d.items():
        # Si la répartition est spécifiée
        if repartition is not None:
            # Répartir la valeur du jour sur les 24 heures en fonction de la répartition
            valeurs_heure = valeur_jour * repartition.values
        else:
            # Si aucune répartition n'est spécifiée, répéter la valeur du jour pour les 24 heures
            valeurs_heure = np.full(24, valeur_jour)
        
        # Créer les index pour chaque heure de la journée
        index_heures = [date_jour.replace(hour=heure) for heure in range(24)]
        
        # Créer une série horaire pour cette journée
        serie_jour = pd.Series(valeurs_heure, index=index_heures)
        
        # Concaténer cette série horaire à la série résultante
        serie_h = pd.concat([serie_h, serie_jour])
    
    return serie_h
    
