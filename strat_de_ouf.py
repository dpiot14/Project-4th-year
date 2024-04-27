# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:17:50 2024

@author: Manon Julia
"""

import climix.technologies as te
import numpy as np
import methodes_etude_serie as meth

#########################
### TOUT EST EN GW(h) ###
#########################

# Nombre d'heures dans l'année
H = (24 * 365)

class PredicteurDeCharge():
    def __init__(self, méthode, horizon=24, cyclique=H):
        self.horizon = horizon*10 # ???
        self.méthode = méthode
        self.trainard = 0
        self.somme=0
        self.set_somme_init(start=0, stop=self.horizon)
        self.cyclique=cyclique
        self.k=0

    def set_somme_init(self, start, stop):
        self.somme = 0
        for k in range(start,stop-1):
            self.somme += self.méthode(k)

    def __next__(self):
        self.somme -= self.trainard
        self.trainard = self.méthode(self.k)
        self.somme += self.méthode((self.k+self.horizon-1)%self.cyclique)
        self.k = (self.k + 1)%self.cyclique
        return self.somme
    
class PredicteurDeChargeMeteo():
    
def predicteur_meteo():
    """
    Prend les données météos réelles, génère une erreur, renvoie le resultat
    
    Args
    - vent
    - couverture nuageuse
    - irradiance solaire
    - précipitations totales

    Returns
    Prévision météo


    """
    return None

def predict_load_factor(tedance, saisonnalite, j_asimu, p=0.45645229):
    
    """
    Fonction qui vise à prédire un facteur de charge pour un nombre de jours données
    Args 
    -------
        tendance = la tendnace de la série à reconstruire
        saisonnalite = la saisonnalité de la série à reconstruire
        j_asimu = nombre de jours à simuler
        p = coefficient de corrélation avec les résidus de la veille
            par défaut, p=0.45645229 qui est la valeur déterminée par notre modélisation ARMA  

    Returns
    -------
    Serie 
        Série resconstruite avec ajout de la tendance et saisonnaité sur la prédiction de résidus

    """
    # Génère un résidu gaussien (normal) avec une moyenne de 0 et un écart type de 1
    residual = np.random.normal(0, 1)
    for j in range(len(j_asimu)):
        residual = p*residual + (1-p)*np.random.normal(0, 1)
    
    serie = meth.Ajout_Tendance_Saisonnalite(residual, tendance, saisonnalite, bords_tendance = 'nearest') 
    return serie
    
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


def strat_stockage(prodres, Step, Battery, Gas, Lake, Nuclear):
    """

    """
    # Prévision de production minimum nucléaire dans les prochaines 24 heures
    pred_nuke24_min = PredicteurGlissant(Nuclear.p_min_effective) 
    # Prévision de production maximale nucléaire dans les prochaines 24 heures
    pred_muke24_max = PredicteurGlissant(Nuclear.p_max_effective)
    # Prévision de production résiduelles dans les prochaines 24 heures
    pred_prodres24 = PredicteurGlissant(lambda k: prodres[k])

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
    #Pour K dans toutes les heures de l'année
    for k in range(H):
        # On passe au rang suivant pour nuke24min et nuke24max
        nuke24min = pred_nuke24_min.__next__()
        
        nuke24max = pred_muke24_max.__next__() # On récupère le rang suivant
        prodres24 = pred_prodres24.__next__()
        Lake.recharger(k) # On recherge les lacs
        if prodres24 + nuke24max < 0:
            état = "pénurie"
            consigne_SB = cap_sb_pénurie 

        elif prodres24 + nuke24min > 0:
            état = "abondance"
            consigne_SB = cap_sb_abondance

        else:
            état = "flexible"
            consigne_SB = cap_sb_milieu + (sb_écart * 0.99) # 0.99 pour variations progressives

        sb_écart = consigne_SB - cap_sb_milieu # Ecart entre la consigne et la réalisation

        prodres_k = prodres[k]

        prodres_k += Lake.produire_minimum(k)

        stock_SB = Step.stock[k] + Battery.stock[k] # On récupère la valeur des stocks
        a_decharger_SB = stock_SB - consigne_SB
        
        ####################################
        

        if état == "pénurie":
            # Nuke au max
            prodres_k += Nuclear.pilote_prod(k, Nuclear.Pout(k)) 
            if prodres_k > 0:
                #reliquat on recharge
                prodres_k -= recharge_plusieur_techs(k, liste=[Battery, Step, Gas], astocker=prodres_k)
                #reliquat on risuqe d'écrêter : on annule le trop
                if prodres_k > 0:
                    prodres_k -= Nuclear.pilot_annule_prod(k, prodres_k)
                surplus[k]= prodres_k

            else:
                aproduire_k = -prodres_k
                if stock_SB > 0.3 * cap_sb_max:
                    aproduire_k -= decharge_plusieur_techs(k, liste=[Step, Battery, Lake, Gas], aproduire=aproduire_k)
                else:
                    aproduire_k -= decharge_plusieur_techs(k, liste=[Lake, Step, Battery, Gas], aproduire=aproduire_k)
                manque[k] = aproduire_k

        elif état == "abondance":
            # nuke au min
            prodres_k += Nuclear.pilote_prod(k, 0)
            # gaz à fond
            prodres_k -= Gas.recharger(k, Gas.Pin(k))

            if a_decharger_SB < 0:
                # les batteries veulent remonter à 30% tant mieux !
                prodres_k -= recharge_plusieur_techs(k, liste=[Step, Battery], astocker= -a_decharger_SB)
            else:
                # on prend le risque d'écrêter
                prodres_k += decharge_plusieur_techs(k, liste=[Battery, Step], aproduire=a_decharger_SB)

            if prodres_k > 0:
                #on écrêterait
                prodres_k -= recharge_plusieur_techs(k, liste=[Step, Battery], astocker=prodres_k)
                surplus[k] = prodres_k
            else:
                aproduire = -prodres_k
                # un peu de nuke pour recharger le gas et batt
                aproduire -= Nuclear.pilote_prod(k, aproduire)
                # on risque la pénurie finalement : on annule la production de H2
                aproduire -= Gas.annuler_recharger(k, aanuler= aproduire)
                # on vide les batterie sous 30% puis lac puis Gas fossile
                aproduire -= decharge_plusieur_techs(k, liste=[Battery, Step, Lake, Gas], aproduire=aproduire)
                manque[k] = aproduire

        else:
            # Flexible


            #regul batteries
            if a_decharger_SB < 0:
                # les batteries veulent remonter à 50%
                prodres_k -= recharge_plusieur_techs(k, liste=[Step, Battery],
                                                     astocker=-a_decharger_SB)

            else:
                # on prend le risque d'écrêter
                prodres_k += decharge_plusieur_techs(k, liste=[Battery, Step],
                                                     aproduire=a_decharger_SB)
            # gaz à fond
            prodres_k -= Gas.recharger(k, Nuclear.Pout(k) + prodres_k) 
            # max de Gaz que nucléaire + renouvelable permet

            prodres_k += Nuclear.pilote_prod(k, -prodres_k) # Ajout du nucléaire nécessaire

            if prodres_k > 0:
                # on écrêterait
                prodres_k -=  recharge_plusieur_techs(k, liste=[Step, Battery],
                                                      astocker=prodres_k)
                surplus[k] = prodres_k
            else:
                # risque de pénurie
                prodres_k += decharge_plusieur_techs(k, liste=[Lake, Step, Battery, Gas],
                                                     aproduire=-prodres_k)
                manque[k] = -prodres_k
        pass






    return surplus, manque
