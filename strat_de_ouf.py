# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:17:50 2024

@author: Manon Julia
"""

import Model_prediction.technologies as te
import numpy as np
import methodes_etude_serie as meth
import stratege as strat1
import pandas as pd

import renewable_study as rw

#########################
### TOUT EST EN GW(h) ###
#########################

# Nombre d'heures dans l'année
H = (24 * 365)


#Horizon est par défaut en jours maintenant
class PredicteurGlissant():
    def __init__(self, méthode, horizon=1, cyclique=H):
        self.horizon_d = horizon  #En jours
        self.horizon_h = horizon*24  #En heures
        self.méthode = méthode
        self.trainard = 0
        self.somme=0
        self.set_somme_init(start=0, stop=self.horizon_h)
        self.cyclique=cyclique
        self.k=0

    def set_somme_init(self, start, stop):
        self.somme = 0
        for k in range(start,stop-1):
            self.somme += self.méthode(k)

    def __next__(self):
        self.somme -= self.trainard
        self.trainard = self.méthode(self.k)
        self.somme += self.méthode((self.k+self.horizon_h-1)%self.cyclique)
        self.k = (self.k + 1)%self.cyclique
        return self.somme


class PredicteurRenouvelable():
    def __init__(self, méthode, horizon=1, cyclique=H, IC=[0.95,-1]):
        self.horizon_d = horizon  #En jours
        self.horizon_h = horizon*24  #En heures
        self.méthode = méthode
        self.trainard = 0
        self.somme=0
        self.set_somme_init(start=0, stop=self.horizon_h)
        self.cyclique=cyclique
        self.k=0

    def set_somme_init(self, start, stop):
        self.somme = 0
        for k in range(start,stop-1):
            self.somme += self.méthode(k)

    def __next__(self):
        self.somme -= self.trainard
        self.trainard = self.méthode(self.k)
        self.somme += self.méthode((self.k+self.horizon_h-1)%self.cyclique)
        self.k = (self.k + 1)%self.cyclique
        return self.somme
    

# Pour recharger astocker GW à l'heure k dans les technologies présentes dans liste
# On commence par recharger dans la première tech de la liste, puis la 2eme, ...
def recharge_plusieur_techs(k, liste, astocker):
    astocker_init = astocker
    for tec in liste:
        astocker -= tec.recharger(k=k, astocker=astocker)
    return astocker_init - astocker


# idem decharge_plusieur_techs mais en faisant l'inverse
def decharge_plusieur_techs(k, liste, aproduire):
    aproduire_init = aproduire
    for tec in liste:
        aproduire -= tec.décharger(k=k, aproduire=aproduire)
    return aproduire_init - aproduire


def add_error(mae, size):
    return np.random.normal(loc=0, scale=mae, size=size)


'''On a pas besoin de prendre les valeurs renouvelables'''
'''Pas le détail pour les stockages, je met donc tout dans PHS'''


#Dans les IC, le premier doit être celui pour pénurie, le 2ème pour abondance
def strat_stockage(prodres, Step, Battery, Gas, Lake, Nuclear, Conso, fac_IC, H):
    """

    """
    
    #On initialise les éléments à renvoyer dans la fonction
    # Phs,Battery,Methanation,Lake, thermal, Nucléaire
    ''''''
    Phs_tab = np.zeros(H)
    Battery_tab = np.zeros(H)
    Gas_tab = np.zeros(H)
    Lake_tab = np.zeros(H)
    Thermal_tab = np.zeros(H)
    Nuc_tab = np.zeros(H)
    ''''''
    
    # Prévision de production minimum nucléaire dans les prochaines 24 heures
    pred_nuke24_min = strat1.PredicteurGlissant(Nuclear.p_min_effective) 
    # Prévision de production maximale nucléaire dans les prochaines 24 heures
    pred_muke24_max = strat1.PredicteurGlissant(Nuclear.p_max_effective)
    # Création du Prédicteur Glissant pour la production réelle
    pred_prodres24 = strat1.PredicteurGlissant(lambda k: prodres[k])
    
    pred_prodres_IC = {}

    # Création des prédicteurs glissant pour les IC
    keys = list(fac_IC.keys())
    pred_prodres_IC['penurie'] = strat1.PredicteurGlissant(lambda k: fac_IC[keys[0]][k])
    pred_prodres_IC['abondance'] = strat1.PredicteurGlissant(lambda k: fac_IC[keys[1]][k])

    # Capacité maximale de Step + Batterie
    cap_sb_max = Step.capacité + Battery.capacité
    # Capacité dans une situation prévue normale
    cap_sb_milieu = 0.5 * cap_sb_max
    # Capacité dans une situation prévue d'abondance
    cap_sb_abondance = 0
    # Capacité dans une situation prévue de pénurie
    cap_sb_pénurie = cap_sb_max
    sb_écart = 0
    
    # Technologies de stockage sans les lacs
    tecstock = {"Battery": Battery, "Step": Step}

    # Technologies de stockage avec les lacs
    tecdestock = {"Lake": Lake, "Step": Step, "Battery": Battery}

    # On initialise le surplus sur le même nombre d'heure que la production résiduelle
    surplus = np.zeros(len(prodres))
    # On initialise le manque sur le même nombre d'heure que la production résiduelle
    manque = np.zeros(len(prodres))
    # On initialise l'état sur la même taille. 0 = pénurie, 100 = flex, 200 = abondance
    etat_tab = np.zeros(len(prodres))
    #Pour K dans toutes les heures de l'année
    for k in range(H):
        # On passe au rang suivant pour nuke24min et nuke24max
        nuke24min = pred_nuke24_min.__next__()
        
        nuke24max = pred_muke24_max.__next__() # On récupère le rang suivant
        prodres24 = pred_prodres24.__next__()
        
        sum_prodres_IC_p = pred_prodres_IC['penurie'].__next__()
        sum_prodres_IC_a =  pred_prodres_IC['abondance'].__next__()
        
        print(nuke24min,nuke24max)
        Lake.recharger(k) # On recherge les lacs
        if sum_prodres_IC_p + nuke24max - Conso[k]< 0:
            état = "pénurie"
            consigne_SB = cap_sb_pénurie 
            etat_tab[k]=0

        elif sum_prodres_IC_a + nuke24min - Conso[k] > 0:
            état = "abondance"
            consigne_SB = cap_sb_abondance
            etat_tab[k]=200

        else:
            état = "flexible"
            consigne_SB = cap_sb_milieu + (sb_écart * 0.99) # 0.99 pour variations progressives
            etat_tab[k]=100

        sb_écart = consigne_SB - cap_sb_milieu # Ecart entre la consigne et la réalisation

        prodres_k = prodres[k]
        
        temp = Lake.produire_minimum(k)
        prodres_k += temp
        Lake_tab[k] += temp
        

        stock_SB = Step.stock[k] + Battery.stock[k] # On récupère la valeur des stocks
        a_decharger_SB = stock_SB - consigne_SB
        
        ####################################
        

        if état == "pénurie":
            # Nuke au max
            temp = Nuclear.pilote_prod(k, Nuclear.Pout(k))
            print(temp)
            prodres_k += temp
            Nuc_tab[k] = temp
            
            if prodres_k > 0:
                #reliquat on recharge
                temp = recharge_plusieur_techs(k, liste=[Battery, Step, Gas], astocker=prodres_k)
                Phs_tab[k] = -temp
                prodres_k -= temp
                #reliquat on risque d'écrêter : on annule le trop
                if prodres_k > 0:
                    temp = Nuclear.pilot_annule_prod(k, prodres_k)
                    prodres_k -= temp
                    Nuc_tab[k] -= temp
                    
                surplus[k] = prodres_k

            else:
                aproduire_k = -prodres_k
                if stock_SB > 0.3 * cap_sb_max:
                    temp = decharge_plusieur_techs(k, liste=[Step, Battery, Lake, Gas], aproduire = aproduire_k)
                    aproduire_k -= temp
                    Phs_tab[k] = temp
                else:
                    temp = decharge_plusieur_techs(k, liste=[Lake, Step, Battery, Gas], aproduire=aproduire_k)
                    aproduire_k -= temp
                    Phs_tab[k] = temp
                manque[k] = aproduire_k

        elif état == "abondance":
            # nuke au min
            temp = Nuclear.pilote_prod(k, 0)
            prodres_k += temp
            print(temp)
            Nuc_tab[k] = temp
            # gaz à fond
            temp = Gas.recharger(k, Gas.Pin(k))
            prodres_k -= temp
            Gas_tab[k] -= temp

            if a_decharger_SB < 0:
                # les batteries veulent remonter à 30% tant mieux !
                temp = recharge_plusieur_techs(k, liste=[Step, Battery], astocker= -a_decharger_SB)
                prodres_k -= temp
                Phs_tab[k] -= temp
            else:
                # on prend le risque d'écrêter
                temp = decharge_plusieur_techs(k, liste=[Battery, Step], aproduire=a_decharger_SB)
                prodres_k += temp
                Phs_tab[k] += temp
                

            if prodres_k > 0:
                #on écrêterait
                temp = recharge_plusieur_techs(k, liste=[Step, Battery], astocker=prodres_k)
                prodres_k -= temp
                Phs_tab[k] -= temp
                surplus[k] = prodres_k
            else:
                aproduire = -prodres_k
                # un peu de nuke pour recharger le gas et batt
                temp = Nuclear.pilote_prod(k, aproduire)
                aproduire -= temp
                Nuc_tab[k] += temp
                # on risque la pénurie finalement : on annule la production de H2
                temp = Gas.annuler_recharger(k, aanuler= aproduire)
                aproduire -= temp
                Gas_tab[k] += temp
                # on vide les batterie sous 30% puis lac puis Gas fossile
                temp = decharge_plusieur_techs(k, liste=[Battery, Step, Lake, Gas], aproduire=aproduire)
                aproduire -= temp
                Phs_tab[k] += temp
                
                manque[k] = aproduire

        else:
            # Flexible


            #regul batteries
            if a_decharger_SB < 0:
                # les batteries veulent remonter à 50%
                temp = recharge_plusieur_techs(k, liste=[Step, Battery],
                                                     astocker=-a_decharger_SB)
                prodres_k -= temp
                Phs_tab[k] -= temp

            else:
                # on prend le risque d'écrêter
                temp = decharge_plusieur_techs(k, liste=[Battery, Step],
                                                     aproduire=a_decharger_SB)
                prodres_k += temp
                Phs_tab[k] += temp
                
            # gaz à fond
            temp = Gas.recharger(k, Nuclear.Pout(k) + prodres_k) 
            prodres_k -= temp
            Gas_tab[k] -= temp
            # max de Gaz que nucléaire + renouvelable permet

            temp = Nuclear.pilote_prod(k, -prodres_k) # Ajout du nucléaire nécessaire
            prodres_k += temp
            Nuc_tab[k]+= temp
            

            if prodres_k > 0:
                # on écrêterait
                temp = recharge_plusieur_techs(k, liste=[Step, Battery],
                                                      astocker=prodres_k)
                prodres_k -= temp
                Phs_tab[k] -= temp
                surplus[k] = prodres_k
            else:
                # risque de pénurie
                temp = decharge_plusieur_techs(k, liste=[Lake, Step, Battery, Gas],
                                                     aproduire=-prodres_k)
                prodres_k += temp
                Phs_tab[k] += temp
                manque[k] = -prodres_k
        pass

    return surplus, manque, Phs_tab, Battery_tab, Gas_tab, Lake_tab, Thermal_tab, Nuc_tab, etat_tab


# Utilisé pour calculer la production résiduelle


#On ajoute dans reg les classes représentant chaque région sous forme d'un dictionnaire

#Si IC différent de None : on génère les séries avec les intervalles de confiance corespondants
#Les IC sont au format [seuil, sens]
#On considère que les IC à générer sont les même pour toutes les sources renouvelables
#Sinon, on génère uniquement les intervalles de bases

#Ici, on doit juste changer les données utilisées

# Expl de base
IC = [[98,-1],[67,1]]
mae_wind = 0.2
mae_solar = 0.2

#mix à supprimer des paramètres

def calculer_prod_non_pilot(mix, nb, reg, j_asimu=365, IC=[["penurie",0.95,-1],["surplus",0.95,1]]):
    
    
    # Puissance d'un pion
    powOnshore = 1.4
    powOffshore = 2.4
    powPV = 3

    # On fait la somme des prods par region pour chaque techno (FacteurDeCharge * NbPions * PuissanceParPion)
    powers_renouvables = {"eolienneON": 1.4,
                          "panneauPV": 3,
                          "eolienneOFF": 2.4}
    
    # On récupère les facteurs de charge des données
    # A modifier je pense, en remplaçant avec les facteurs de charge généré
    
    fdc_on = {}
    fdc_pv = {}
    fdc_on_approx = {}
    fdc_pv_approx = {}
    for r in reg.keys():
        fdc_on[r] = reg[r].generate_load_factor_wind(j_asimu = j_asimu, time_step = 'hour')
        fdc_pv[r] = reg[r].generate_load_factor_solar(j_asimu = j_asimu, time_step = 'hour')
        fdc_on_approx[r] = fdc_on[r] + add_error(mae_wind, j_asimu*24)
        fdc_pv_approx[r] = fdc_pv[r] + add_error(mae_solar, j_asimu*24)
        
        
    fdc_on_IC = {}
    fdc_pv_IC = {}
    fdc_all_IC = {}

    for i in IC:
        fdc_on_IC[i[0]]=0
        fdc_pv_IC[i[0]]=0
        fdc_all_IC[i[0]]=0
        for r in reg.keys():
            fdc_temp = (fdc_on_approx[r] + reg[r].arma_proba_distribustion_wind.IC(proba = i[1], sens = i[2]))*nb[r]["eolienneON"]*powers_renouvables["eolienneON"]
            if fdc_temp < 0 : fdc_temp = 0
            if fdc_temp > 1 : fdc_temp = 1
            fdc_on_IC[i[0]] += fdc_temp
            fdc_all_IC[i[0]] += fdc_on_IC[i[0]]
            
            fdc_temp = (fdc_pv_approx[r] + reg[r].arma_proba_distribustion_solar.IC(proba = i[1], sens = i[2]))*nb[r]["panneauPV"]*powers_renouvables["panneauPV"]
            if fdc_temp < 0 : fdc_temp = 0
            if fdc_temp > 1 : fdc_temp = 1
            fdc_pv_IC[i[0]] += fdc_temp
            fdc_all_IC[i[0]] += fdc_temp
            
            
    fdc_IC= {}
    fdc_IC['on'] = fdc_on_IC
    fdc_IC['pv'] = fdc_pv_IC
    fdc_IC['all'] = fdc_all_IC
    
    # Je gruge pour pas avoir de bug, je le met à l'identique par rapport à fdc_on
    fdc_off = fdc_on


    reg_non_off_shore = [ "bfc", "ara", "cvl", "idf", "est"]  # liste des régions sans off_shore

    prodOnshore = np.zeros(H)
    prodOffshore = np.zeros(H)
    prodPV = np.zeros(H)

    prod = {"eolienneON": np.zeros(H),
                "panneauPV": np.zeros(H),
                "eolienneOFF": np.zeros(H)}
    prod_reg={}
    for r in reg: # Pour chaque région dans mix['unites']
        prod_reg[r] = {}  # La production de la région est initialisée en dictionnaire
        for p, pow in powers_renouvables.items():   # Pour chaque source renouvelable    pow ???
        
        # r enlevé pour l'instant, à gérer
        
        
            if p=="eolienneOFF" and reg in reg_non_off_shore:  # Si eolienne offshore sur région non offshore -> 0
                prod_reg[r][p] = np.zeros(H)
            else:
                # Sinon : prod_reg pour la région et la ressource = facteur de charge de la region * nb[reg][p] * pow
                prod_reg[r][p] = np.array(fdc_on[r]) * nb[r][p] * pow
                prod[p] += prod_reg[r][p] #On ajoute la prod de ce moyen de prodcution dans la prod totale française du moyen de production


    # carte alea MEMFDC (lance 1)
    if mix["alea"] == "MEMFDC1" or mix["alea"] == "MEMFDC2" or mix["alea"] == "MEMFDC3":
        prod_reg["cvl"]["eolienneON"] *= 0.9



    # Definition des productions electriques des rivières et lacs
    coefriv = 13.
    river = pd.read_csv("run_of_river.csv", header=None)
    river.columns = ["heures", "prod2"]
    rivprod = np.array(river.prod2) * coefriv


    #On enregistre dans les chroniques les productions de chaque moyen de production
    chroniques = {"prodOffshore": prod["eolienneOFF"],
                  "prodOnshore": prod["eolienneON"],
                  "prodPV": prod["panneauPV"],
                  "rivprod": rivprod,
                  }
    prod['regions']=prod_reg
    return chroniques, prod, fdc_IC
