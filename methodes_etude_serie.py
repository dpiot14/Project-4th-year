# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:49:59 2024

@author: piotd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utility_tools import string_to_number_and_string

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.api as sm

'''
Fonction pour calculer la tendance et la saisonnalité d'une série temporelle,
Sous format dataframe avec les jours comme index (la variable data)

    Renvoie la tendance(float) et la saisonnalité(np.array)

    Paramètre methode_tend : choix de la méthode pour calculer la tendance
        'mean' : valeur moyenne sur la série (tendance si on suppose la série stationnaire) (valeur par défaut)
        'mobileI_d' : moyenne mobile sur I jours pour calculer la tendance
        Expl : pour la moyenne mobile sur 30 jours, entrer 'mobile30_d'

    Paramètre methode_saison : choix de la méthode pour calculer la saisonnalite
        'mobile28_d' : moyenne mobile sur 28 jours de la valeur moyenne par jour (valeur par défaut)
        Les méthodes 'mobile14_d' et 'mobile7_d' existent aussi
    'mean' : valeur moyenne sur chaque jour de l'année, sans moyenne mobile
    'sinus1_t10' : Calcul du sinus d'interpolation de la saisonnalité sur la valeur moyenne

possibilité de remplacer le 1 par une autre valeur pour utiliser la moyenne mobile correspondante
10 correspond au seuil utilisé : 10->seuil de 1/10
'''

def Etude_Tendance_Saisonnalite_annuelle(data, methode_tend='mean',methode_saison='mobile28_d'):
    data_copy=data.copy()
    
    ## -- Calcul de la tendance 
    if methode_tend=='mean':
        tendance=float(data_copy['electricity'].mean())
    elif methode_tend[:6]=='mobile':   # Si on souhaite utiliser une moyenne mobile
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
    
    # Création des index avec jour et mois
    dates = pd.date_range(start='2024-01-01', periods=366)
    indexes = [f"{date.month}-{date.day}" for date in dates]
    
    pd.DataFrame(data_year_wind_tendance, index=indexes, columns=['electricity'])

    
    if methode_saison[:6]=='mobile': # Si on souhaite utiliser une moyenne mobile
        int_mobile,mode_mobile=string_to_number_and_string(methode_saison[6:])
        if mode_mobile=='_d':
            np_saison=pd.concat([data_year_wind_tendance['electricity'][(366-int_mobile):],data_year_wind_tendance['electricity'],data_year_wind_tendance['electricity'][:(int_mobile)]]).rolling(int_mobile, center=True).mean().to_numpy()[(int_mobile):(int_mobile+366)]
            saison=pd.DataFrame(np_saison, index=indexes, columns=['electricity'])
        else:
            print("Erreur : mauvais argument pour methode_saison : la méthode mobile ne comprends pas "+mode_mobile+"\nVoir la documentation") 
        
    elif methode_saison=='mean':
        saison=data_year_wind_tendance
        
    elif methode_saison[:5]=='sinus': # Si on souhaite utiliser une méthode sinus
        # Utilisation d'une moyenne mobile ou non pour le calcul du sinus
        int_mobile,mode_mobile=string_to_number_and_string(methode_saison[5:]) 
        if mode_mobile[:2]=='_t': # Précision du seuil
            int_threshold,mode_threshold=string_to_number_and_string(mode_mobile[2:])
            
            # Effectuer une transformée de Fourier
            if int_mobile == 1:
                fft = np.fft.fft(data_year_wind_tendance['electricity'])
            else:
                np_saison=pd.concat([data_year_wind_tendance['electricity'][(366-int_mobile):],data_year_wind_tendance['electricity'],data_year_wind_tendance['electricity'][:(int_mobile)]]).rolling(int_mobile, center=True).mean().to_numpy()[(int_mobile):(int_mobile+366)]
                fft = np.fft.fft(np_saison)
        

            # Donne les phases et amplitudes
            amplitudes = np.abs(fft)
            phases = np.angle(fft)

            # Fixer un seuil pour les petites amplitudes
            threshold = np.max(amplitudes)/int_threshold

            # Mettre à zéro les petites amplitudes
            fft[np.where(amplitudes < threshold)] = 0

            # Reconstruire le signal à l'aide de la transformée de Fourier inverse
            saison = np.fft.ifft(fft).real
        else:
            print("Erreur : mauvais argument pour methode_saison : la méthode sinus ne comprends pas "+mode_mobile+"\nVoir la documentation")
    else:
        
        print("Erreur : Methode pour la saisonnalité inconnue, essayez avec une autre valeur pour 'methode_saison'")

    return tendance,saison


'''
Fonction pour retirer la tendance et la saisonnalité à une série temporelle
Actuellement, la tendance doit être une valeur constante, au format int/float
Retourne la série nettoyée
'''
def Retrait_Tendance_Saisonnalite(data, tendance, saisonnalite):
    
    ## -- Création d'une copie de data
    data_copy=data.copy()
    
    ## -- Retrait de la tendance
    if type(tendance) != int and type(tendance) != float :
        print("Erreur : la fonction Retrait_Tendance_Saisonnalite ne fonction qu'avec une tendance constante (int ou float) actuellement")
    else:
        data_copy['electricity']-=tendance
        
    ## -- Retrait de la saisonnalité
    
    for index in data_copy.index:
        data_copy.loc[index,'electricity']-=saisonnalite.loc[str(index.month)+"-"+str(index.day),'electricity']
        
    return data_copy


'''
Fonction pour ajouter la tendance et la saisonnalité à une série temporelle
Actuellement, la tendance doit être une valeur constante, au format int/float
Retourne la série reconstruite
'''
def Ajout_Tendance_Saisonnalite(data, tendance, saisonnalite):
    
    ## -- Création d'une copie de data
    data_copy=data.copy()
    
    ## -- Ajout de la tendance
    if type(tendance) != int and type(tendance) != float :
        print("Erreur : la fonction Ajout_Tendance_Saisonnalite ne fonction qu'avec une tendance constante (int ou float) actuellement")
    else:
        data_copy['electricity']+=tendance
        
    ## -- Ajout de la saisonnalité
    
    for index in data.index:
        data_copy.loc[index,'electricity']+=saisonnalite.loc[str(index.month)+"-"+str(index.day),'electricity']
        
    return data_copy

'''
Fonction pour réaliser l'étude ARMA d'une série temporelle data
    Renvoie par défaut la prédiction réalisée à l'aide du modèle ARMA
    Les paramètres p et q sont des paramètres de la fonction ARMA

Les autres paramètres permettent de choisir les graphes que l'on souhaite
Afficher à l'écran

Note : Ici on utilise un processus ARMA car les séries temporelles étudiées sont déjà stationnaires
Nous n'avons pas besoin de les différencier, d'où le paramètre constant d=0
'''

def Arma_predict(data,p,q,graph_predict=False,graph_predict_last_year=False,graph_autocorrelation=False,error=False,d=0):
    arma = ARIMA(data, order=(p,d,q)).fit() # Par défaut, d=0 pour un processus ARMA
    pred = arma.predict()
    
    if graph_predict:
        plt.figure(figsize=(12,6))
        plt.plot(data,label="Réel")
        plt.plot(pred, color = "r",label="Prédiction")
        plt.xlabel('Temps')
        plt.ylabel('Facteur de capacité')
        plt.title("Comparaison de la série réelle et de la prédiction par modèle ARMA")
        plt.legend()
        plt.show()
        
    if graph_predict_last_year:
        # Sélection des 365 derniers jours pour chaque série
        serie_derniere_annee = data[-365:]
        pred_derniere_annee = pred[-365:]

        # Création du graphique
        plt.figure(figsize=(12,6))
        
        # Affichage de la série réelle pour les 365 derniers jours
        plt.plot(serie_derniere_annee, label="Réel")
        
        # Affichage de la prédiction pour les 365 derniers jours
        plt.plot(pred_derniere_annee, color = "r", label="Prédiction")
    
        # Ajout des légendes et titres
        plt.xlabel('Temps')
        plt.ylabel('Facteur de capacité')
        plt.title("Comparaison de la série réelle et de la prédiction par modèle ARMA pour les 365 derniers jours")
        plt.legend()
        
        # Affichage du graphique
        plt.show()
    
    if graph_autocorrelation:
        plot_pacf(data)
        plot_acf(data)
        plt.show()
        
    if error:
        # Calcul du MSE
        mse = mean_squared_error(serie_derniere_annee, pred_derniere_annee)
        
        # Calcul du MAE
        mae = mean_absolute_error(serie_derniere_annee, pred_derniere_annee)
        
        # Calcul du RMSE
        rmse = np.sqrt(mse)
        
        # Calcul du R^2
        r2 = r2_score(serie_derniere_annee, pred_derniere_annee)
        
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")
    
    return pred

'''
Fonction pour réaliser l'étude arimax d'une série temporelle data
    Renvoie par défaut la prédiction réalisée à l'aide du modèle ARIMAX
    - data correspond aux données (à predire et exogène)
    - name_predict est le nom de la variable à prédire

Les paramètres p, d et q sont des paramètres de la fonction ARIMAX
    - par défaut, d=0 (ARMA sans différenciation)
    - day_exog est le nombre de jour de décallage pour utiliser les données exogènes
    - par défaut, avce day_exog=0, on regarde les variables exogènes le jour même
    - Avec day_exog=1, on utilise non pas les données exogènes le jour même mais la veille


Ajouter la notion de jeu de données train et test
Ajouter infos sur les options
Ajouter validation croisée
Ajouter option normalisation
Ajouter l'impact des jours précédents
'''

def Arimax_predict(data, name_predict, p, q, day_exog=0, int_conf=False, int_conf_1y=False ,error=False,d=0):
    data_copy=data.copy()
    endog=data_copy[name_predict]
    if day_exog==0:
        exog=data_copy.drop(columns=[name_predict])
    elif day_exog>0:
        exog_shift = data_copy.drop(columns=[name_predict]).shift(day_exog)  # On décale les données de day_exog jours
        # On supprime les premiers jours de données car on NaN dans exog_shift à cause du décalage
        exog=exog_shift.iloc[day_exog:]
        endog=endog.iloc[day_exog:]
        exog.columns=exog.columns+"_"+str(day_exog) #On change les noms des colonnes
    else:
        print("Erreur : dans Arimax_predict, la valeur de day_exog est négative, elle doit être positive ou nulle")
    
    # Ajustement du modèle ARIMAX avec les variables exogènes choisies
    model = sm.tsa.ARIMA(endog=endog, exog=exog, order=(p, d, q))
    results = model.fit()
    
    # Prédictions avec les variables exogènes
    predictions = results.get_forecast(steps=len(endog), exog=exog)
    predicted_means = predictions.predicted_mean
    predicted_intervals = predictions.conf_int()
    
    if int_conf:
        # Dates de l'ensemble de test - pour l'axe des x
        dates = endog.index
        
        # Tracer les observations réelles
        plt.figure(figsize=(10,5))
        plt.plot(dates, endog, label='Observations réelles', color='red')
        
        # Tracer les prédictions moyennes
        plt.plot(dates, predicted_means, label='Prédictions', color='blue')
        
        # Tracer les intervalles de confiance
        plt.fill_between(dates, predicted_intervals.iloc[:, 0], predicted_intervals.iloc[:, 1], color='lightblue', alpha=0.3, label='Intervalle de confiance à 95%')
        
        # Personnaliser le graphique
        plt.title('Prédictions ARIMAX et intervalles de confiance')
        plt.xlabel('Date')
        plt.ylabel('Electricity')
        plt.legend()
        plt.tight_layout()
        
        # Afficher le graphique
        plt.show()
        
    if int_conf_1y:
        # Dates de l'ensemble de test - pour l'axe des x
        n_line=endog.shape[0]
        dates = endog.iloc[n_line-365:].index
        
        # Tracer les observations réelles
        plt.figure(figsize=(10,5))
        plt.plot(dates, endog.iloc[n_line-365:], label='Observations réelles', color='red')
            
        # Tracer les prédictions moyennes
        plt.plot(dates, predicted_means.iloc[n_line-365:], label='Prédictions', color='blue')
            
        # Tracer les intervalles de confiance
        plt.fill_between(dates, predicted_intervals.iloc[n_line-365:, 0], predicted_intervals.iloc[n_line-365:, 1], color='lightblue', alpha=0.3, label='Intervalle de confiance à 95%')
            
        # Personnaliser le graphique
        plt.title('Prédictions ARIMAX et intervalles de confiance')
        plt.xlabel('Date')
        plt.ylabel('Electricity')
        plt.legend()
        plt.tight_layout()
            
        # Afficher le graphique
        plt.show()
        
    if error:
        assert len(predicted_means) == len(endog), "La longueur des prédictions et des valeurs réelles doit être identique."


        # Calculer les métriques
        mse = mean_squared_error(endog, predicted_means)
        mae = mean_absolute_error(endog, predicted_means)
        rmse = np.sqrt(mse)  # RMSE est simplement la racine carrée de MSE
        r2 = r2_score(endog, predicted_means)
        
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")
    
    
    return predictions