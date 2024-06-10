# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:26:05 2024

@author: piotd
"""

import csv
import pandas as pd
import numpy as np

import utility_tools as ut
import methodes_etude_serie as meth
 
# Organisation du dossier à préciser pour path
# methode : méthode pour la saisonnalité
 

#En entrée : np.array de résidus sous forme de vecteur
#On créé des catégories (nbr en paramètre), régulières en nbr taille, pour générer les résidus moyens avec la probabilité de la taille de la série
class loi_proba:
    def __init__(self, residus, bin_nbr=20):
        # Calculer la taille de chaque panier
        amplitude = max(abs(np.min(residus)),np.max(residus))
        size = amplitude / bin_nbr
        # Créer un tableau pour stocker les résultats
        bin_table = np.zeros((bin_nbr, 2))
        
        bin_intervals = np.zeros(bin_nbr)
        for i in range(bin_nbr):
            bin_intervals[i] = -(1/2)*amplitude+i*size #min de l'intervalle
    
        #tri des résidus
        sorted_residus = np.sort(residus)
    
        i=0
        # Remplir le tableau avec les moyennes et les fréquences
        for j in range(len(sorted_residus)):
            if i < bin_nbr-1:
                while sorted_residus[j]>bin_intervals[i+1]:
                    i+=1
                    if i == bin_nbr-1: break

            bin_table[i,0]+=sorted_residus[j] #valeur du résidus
            bin_table[i,1]+=1 #nbr d'éléments
            
        #On transforme la somme des résidus en moyenne
        bin_table[:,0]=bin_table[:,0]/bin_table[:,1]
        #On transforme le nbr de résidus en fréquence
        bin_table[:,1]=bin_table[:,1]/np.sum(len(sorted_residus))
        bin_table = bin_table[~np.isnan(bin_table).any(axis=1)]
        
        self.residus = sorted_residus
        self.bin_table = bin_table
    
        
    def generate(self, n=1):
        #seed = np.random.randint(1,100000000)
        #print(seed)
        #np.random.seed(seed)
        rng = np.random.default_rng()
        if n>1:
            #random_state = np.random.RandomState(seed=your_favorite_seed_value)
            
            #rng = np.random.default_rng()
            
            return rng.choice(self.bin_table[:, 0], n, p=self.bin_table[:, 1])
            #print(random_state.choice(self.bin_table[:, 0], n, p=self.bin_table[:, 1]))
        else:
            #rng = np.random.default_rng()
            return rng.choice(self.bin_table[:, 0], p=self.bin_table[:, 1])
        
    def IC(self, proba = 0.95, sens = -1):
        # si sens = -1, on veut probabilité proba d'être plus grand
        # si sens = 1, on veut probabilité proba d'être plus petit
        
        n = len(self.residus)
        nbr = int(n*proba)
        
        if sens == -1:
            return self.residus[1-nbr]
        elif sens == 1:
            return self.residus[nbr]
        else:
            print("Erreur : sens de IC ne vaut ni I ni -1")
            return 0
            
            




class renewable_study:
    
    mean_day_wind=np.ones(24)
    mean_day_solar=np.ones(24)
    
    def __init__(self, path_data):
        self.path_data = path_data
        self.data_hour, self.data_day = ut.regroupement_data_all(path_data,annee_debut=[1980,1980,2000],annee_fin=[2023,2023,2023])
        
    def temporal_analysis(self, methode):
        self.methode = methode
        self.tendance_wind, self.saisonnalite_wind = meth.Etude_Tendance_Saisonnalite_annuelle(self.data_day['wind_electricity'], methode_tend='mean',methode_saison=self.methode[0],name='wind_electricity')
        self.tendance_solar, self.saisonnalite_solar = meth.Etude_Tendance_Saisonnalite_annuelle(self.data_day['solar_electricity'], methode_tend='mean',methode_saison=self.methode[1], name='solar_electricity')
        self.residus_wind = meth.Retrait_Tendance_Saisonnalite(self.data_day['wind_electricity'], tendance=self.tendance_wind,saisonnalite=self.saisonnalite_wind)
        self.residus_solar = meth.Retrait_Tendance_Saisonnalite(self.data_day['solar_electricity'], tendance=self.tendance_solar,saisonnalite=self.saisonnalite_solar)
        # self.error_arimax = a compléter avec code à coder dans fonction Arimax
        
    def arma_analysis(self, p=1, q=3):  #On effectue l'Arma, on récupère les résidus, le coeff de corrélation, et la loi des résidus (sous forme d'histogramme pour génération : découpage en une centaine de valeurs)
        self.arma_residus_wind,self.coeff_arma_wind = meth.Arma_predict(self.data_day['wind_electricity'],p=p,q=q)
        self.arma_proba_distribustion_wind = loi_proba(self.arma_residus_wind, bin_nbr = 50)
        self.arma_residus_solar,self.coeff_arma_solar = meth.Arma_predict(self.data_day['solar_electricity'],p=p,q=q)
        self.arma_proba_distribustion_solar = loi_proba(self.arma_residus_solar, bin_nbr = 50)
        
    def arimax_analysis(self, mae_wind = 0.368, mae_temp=0.246, mae_irrad=11):
        exog_wind = self.data_day['wind_speed'] + np.random.normal(loc=0, scale=mae_wind, size=len(self.data_day['wind_speed']))
        self.arimax_residus_wind = meth.Arimax_predict(self.data_day['wind_electricity'], exog_wind, p=1, q=1, normalisation=True)
        self.arimax_proba_distribustion_wind = loi_proba(self.arimax_residus_wind, bin_nbr = 50)
        
        exog_1 = self.data_day['irradiance_direct'] + np.random.normal(loc=0, scale=mae_irrad, size=len(self.data_day['irradiance_direct']))
        exog_2 = self.data_day['temperature'] + np.random.normal(loc=0, scale=mae_temp, size=len(self.data_day['temperature']))
        exog_solar = pd.concat([exog_1, exog_2], axis=1)
        self.arimax_residus_solar = meth.Arimax_predict(self.data_day['solar_electricity'], exog_solar, p=1, q=1, normalisation=True)
        self.arimax_proba_distribustion_solar = loi_proba(self.arimax_residus_solar, bin_nbr = 50)
    
    
    #def arimax_analysis(self, ...): #On effectue l'Arimax, on récupère les résidus, et la loi des résidus (sous forme d'histogramme, + Intervalle de confiance (une série discrétisée avec les valeurs pour IC))
    
    
    #Ajouter résultats Arma (à voir le format)
    def return_csv(self, name_file, path="./"):
        df = pd.concat([self.saisonnalite_wind, self.saisonnalite_solar], axis=1)
        df.columns = ["wind_electricity","solar_electricity"]
        df.loc['tendance'] = self.tendance_wind
        df.to_csv(name_file)
        
    def open_csv(self, name_file):
        # Lecture du fichier CSV
        df = pd.read_csv(name_file, index_col=0)
        
        # Mise à jour des variables de la classe
        self.saisonnalite_wind = df['wind_electricity']
        self.saisonnalite_solar = df['solar_electricity']
        self.tendance_wind = df.loc['tendance']
        
        
    # p à remplacer avec les résultats du Arma
    # Prise en compte de la date de départ start_day à ajouter
    # Remplacer le gaussien par la fonction générée pour simuler les résidus à l'aide de np.random.choice(tableau[:, 1], p=tableau[:, 0])
    def generate_load_factor_wind(self, j_asimu=365, start_day=pd.to_datetime('01-01-2025', format='%d-%m-%Y'), time_step = 'day'):
    
        """
        Fonction qui vise à prédire un facteur de charge pour un nombre de jours donné
        Args 
        -------
            tendance = la tendance de la série à reconstruire
            saisonnalite = la saisonnalité de la série à reconstruire
            j_asimu = nombre de jours à simuler  
    
        Returns
        -------
        Serie 
            Série resconstruite avec ajout de la tendance et saisonnaité sur la prédiction de résidus
    
        """
        residual = np.zeros(j_asimu) #Initialisation de la série des résidus
        p = self.coeff_arma_wind
        #On veut transformer cette série en dataframe avec en indice les dates
        indices_dates = pd.date_range(start=start_day, periods=j_asimu)
        df_residual = pd.Series(data = residual, index=indices_dates)
        
        alea_part = self.arma_proba_distribustion_wind.generate(n=j_asimu)
        
        df_residual.iloc[0] = alea_part[0]
        for j in range(1,j_asimu):
            df_residual.iloc[j] = p*df_residual.iloc[j-1] + alea_part[j]
            
            saison = self.saisonnalite_wind.loc[str(df_residual.index[j].month)+"-"+str(df_residual.index[j].day)].iloc[0]
            if df_residual.iloc[j] + self.tendance_wind + saison < 0:
                df_residual.iloc[j] = 0
            elif df_residual.iloc[j] + self.tendance_wind + saison > 1:
                df_residual.iloc[j] = 1
            #print(df_residual.iloc[j])
        
        serie = meth.Ajout_Tendance_Saisonnalite(df_residual, self.tendance_wind, self.saisonnalite_wind, bords_tendance = 'nearest') 
        
        if time_step == 'hour':
            serie = ut.day_to_hour(serie, repartition=self.mean_day_wind)
        
        return serie
    
    def generate_load_factor_solar(self, j_asimu=365, start_day=pd.to_datetime('01-01-2025', format='%d-%m-%Y'), time_step = 'day'):
    
        """
        Fonction qui vise à prédire un facteur de charge pour un nombre de jours donné
        Args 
        -------
            tendance = la tendance de la série à reconstruire
            saisonnalite = la saisonnalité de la série à reconstruire
            j_asimu = nombre de jours à simuler  
    
        Returns
        -------
        Serie 
            Série resconstruite avec ajout de la tendance et saisonnaité sur la prédiction de résidus
    
        """
        residual = np.zeros(j_asimu) #Initialisation de la série des résidus
        p = self.coeff_arma_solar
        #On veut transformer cette série en dataframe avec en indice les dates
        indices_dates = pd.date_range(start=start_day, periods=j_asimu)
        df_residual = pd.Series(data = residual, index=indices_dates)
        
        alea_part = self.arma_proba_distribustion_solar.generate(n=j_asimu)
        df_residual.iloc[0] = alea_part[0]
        for j in range(1,j_asimu):
            df_residual.iloc[j] = p*df_residual.iloc[j-1] + alea_part[j]
            saison = self.saisonnalite_solar.loc[str(df_residual.index[j].month)+"-"+str(df_residual.index[j].day)].iloc[0]
            if df_residual.iloc[j] + self.tendance_solar + saison < 0:
                df_residual.iloc[j] = 0
            elif df_residual.iloc[j] + self.tendance_solar + saison > 1:
                df_residual.iloc[j] = 1
            #print(df_residual.iloc[j])
        
        serie = meth.Ajout_Tendance_Saisonnalite(df_residual, self.tendance_solar, self.saisonnalite_solar, bords_tendance = 'nearest') 
        
        if time_step == 'hour':
            serie = ut.day_to_hour(serie, repartition=self.mean_day_solar)
            
        return serie
    
    
    def mean_day(self):
        self.mean_day_wind = self.data_hour["wind_electricity"].groupby(self.data_hour["wind_electricity"].index.hour).mean()
        self.mean_day_wind = self.mean_day_wind/np.sum(self.mean_day_wind.values)*24
        
        self.mean_day_solar = self.data_hour["solar_electricity"].groupby(self.data_hour["solar_electricity"].index.hour).mean()
        self.mean_day_solar = self.mean_day_solar/np.sum(self.mean_day_solar.values)*24
        
    
    
    
### Fonctions de stratégies


# Nombre d'heures dans l'année
H = (24 * 365)

# methode : ??
# horizon : nbr de jours de prédiction
class PredicteurGlissant():
    def __init__(self, méthode, horizon=24, cyclique=H):
        self.horizon = horizon*10   # 240 heures - 10 jours
        self.méthode = méthode  # méthode donnée en argument
        self.trainard = 0  # ???
        self.somme=0  # ???
        self.set_somme_init(start=0, stop=self.horizon)  # on démarre à 0, on va jusqu'à 240, et on prend la 1ere valeur de somme
        self.cyclique=cyclique   # On prend cyclique (pour dire qu'on a un cycle sur une année)
        self.k=0   #???

    # On fait la somme pour la méthode définie de start à stop des valeurs de méthode
    def set_somme_init(self, start, stop):
        self.somme = 0
        for k in range(start,stop-1):
            self.somme += self.méthode(k)

    # On avance à l'heure suivante
    def __next__(self):
        self.somme -= self.trainard  # On retire de la somme la dernière production
        self.trainard = self.méthode(self.k)  # On met dans trainard la production de l'heure actuelle
        self.somme += self.méthode((self.k+self.horizon-1)%self.cyclique) #On ajoute la production à l'horizon-1, en prenant en compte le cycle
        self.k = (self.k + 1)%self.cyclique #On ajoute 1 au cycle, en prenant en compte le cycle
        return self.somme #On renvoie la somme
    
    
    
    
    
    