import os
import numpy as np
import pandas as pd

import technologies as te
dataPath = os.path.dirname(os.path.realpath(__file__))+"/"

#########################
### TOUT EST EN GW(h) ###
#########################

# Nombre d'heures dans l'année
H = (24 * 365)


# Horizon est par défaut en jours maintenant
class PredicteurGlissant():
    def __init__(self, méthode, horizon=1, cyclique=H):
        self.horizon_d = horizon  # En jours
        self.horizon_h = horizon*24  # En heures
        self.méthode = méthode
        self.trainard = 0
        self.somme = 0
        self.set_somme_init(start=0, stop=self.horizon_h)
        self.cyclique = cyclique
        self.k = 0

    def set_somme_init(self, start, stop):
        self.somme = 0
        for k in range(start, stop-1):
            self.somme += self.méthode(k)

    def __next__(self):
        self.somme -= self.trainard
        self.trainard = self.méthode(self.k)
        self.somme += self.méthode((self.k+self.horizon_h-1) % self.cyclique)
        self.k = (self.k + 1) % self.cyclique
        return self.somme


# Pour recharger astocker GW à l'heure k dans les technologies présentes dans liste
# On commence par recharger dans la première tech de la liste, puis la 2eme, ...
def recharge_plusieur_techs(k, liste, astocker, return_dico=False, liste_dico=[]):
    astocker_init = astocker
    dico = {}
    astocker_temp = astocker
    if liste_dico == []:
        liste_dico = liste
    for i in range(len(liste)):
        astocker -= liste[i].recharger(k=k, astocker=astocker)
        dico[liste_dico[i]] = astocker_temp-astocker
        astocker_temp = astocker
    if return_dico:
        return astocker_init - astocker, dico
    else:
        return astocker_init - astocker


# idem decharge_plusieur_techs mais en faisant l'inverse
def decharge_plusieur_techs(k, liste, aproduire, return_dico=False, liste_dico=[]):
    aproduire_init = aproduire
    dico = {}
    aproduire_temp = aproduire
    if liste_dico == []:
        liste_dico = liste
    for i in range(len(liste)):
        aproduire -= liste[i].décharger(k=k, aproduire=aproduire)
        dico[liste_dico[i]]= aproduire_temp-aproduire
        aproduire_temp= aproduire
    if return_dico:
        return aproduire_init - aproduire, dico
    else:
        return aproduire_init - aproduire


'''
prodres : production résiduelle
Step : ??
Battery : ??
Gas : ??
Lake : ??
Nuclear : ??
'''

'''
Avant : - ouverture/génération des classes de chaque région -> a terme l'ouverture est la meilleure option


Dans calcul_prod_non_pilot : - ajouter la génération des données/l'ouverture de données générées précédemment         OK
                             - ajout de l'erreur sur les données                                                      OK
                             - Génération des séries avec les intervalles de confiance nécessaire pour la suite       OK
                             - passage des séries journalières à horaires : production moyennée sur la journée        OK

Dans PredicteurGlissant : - Utile surtout quand on combinera les méthodes -> A voir plus tard

Dans strat stockage : - ajout du paramètre prodres_prev en entrée                                                     OK
                      - différentiation dans le code de la production prévue et de la production réelle
                      : on fait la stratégie sur la production prévue et on a les résultats de la production réelle.

                      Adaptation du code horaire en jour -> On produit autant jour que nuit pour l'installation       OK


                             - ajouter la génération des données estimée en fonction de la méthode/pour les deux méthodes si on combine

Limitations :
On considère un facteur de charge constant sur chaque journée
On prends pas en compte les corrélations entre les régions

'''
def strat_stockage(prodres, Step, Battery, Gas, Lake, Nuclear):  # On utilise la strat_stockage dans l'année
    """

    """

    # Phs,Battery,Methanation,Lake, thermal, Nucléaire
    Phs_tab= np.zeros(H)
    Battery_tab= np.zeros(H)
    Gas_tab= np.zeros(H)
    Lake_tab= np.zeros(H)
    Thermal_tab= np.zeros(H)
    Nuc_tab= np.zeros(H)

    # On initialise le prédicteur glissant pour le nucléaire min
    pred_nuke24_min= PredicteurGlissant(Nuclear.p_min_effective, horizon = 10)
    # On initialise le prédicteur glissant pour le nucléaire max
    pred_muke24_max= PredicteurGlissant(Nuclear.p_max_effective, horizon = 10)
    # On initialise le prédicteur glissant pour les prod résiduelle
    pred_prodres24= PredicteurGlissant(lambda k: prodres[k], horizon = 1)

    # Capacité maximale de Step + Batterie
    cap_sb_max=Step.capacité + Battery.capacité
    # Capacité dans une situation prévue normale
    cap_sb_milieu=0.5 * cap_sb_max
    # Capacité dans une situation prévue d'abondance
    cap_sb_abondance=0
    # Capacité dans une situation prévue de pénurie
    cap_sb_pénurie=cap_sb_max
    # ???
    sb_écart=0

    # Technologies de stockage sans les lacs
    tecstock={"Battery": Battery, "Step": Step}

    # Technologies de stockage avec les lacs
    tecdestock={"Lake": Lake, "Step": Step, "Battery": Battery}

    # On initialise le surplus sur le même nombre d'heure que la production résiduelle
    surplus=np.zeros(len(prodres))
    # On initialise le manque sur le même nombre d'heure que la production résiduelle
    manque=np.zeros(len(prodres))
    # On initialise l'état sur la même taille. 0 = pénurie, 100 = flex, 200 = abondance
    etat_tab=np.zeros(len(prodres))
    # Pour K dans toutes les heures de l'année
    for k in range(H):
        # Production minimale nucléaire 24h = next de pred glissant
        nuke24min=pred_nuke24_min.__next__()
        # Production maximale nucléaire 24h = next de pred glissant
        nuke24max=pred_muke24_max.__next__()
        # Production résiduelle prochaine 24h = next de pred glissant
        prodres24=pred_prodres24.__next__()
        # On recharge les lacs
        Lake.recharger(k)
        # print(nuke24min,nuke24max)
        # print(Nuclear.p_min_effective(k),Nuclear.p_max_effective(k))
        if prodres24 + nuke24min < 0:
            état="pénurie"
            consigne_SB=cap_sb_pénurie
            etat_tab[k]=0

        elif prodres24 + nuke24max > 0:
            état="abondance"
            consigne_SB=cap_sb_abondance
            etat_tab[k]=100

        else:
            état="flexible"
            consigne_SB=cap_sb_milieu + (sb_écart * 0.99)
            etat_tab[k]=50


        sb_écart=consigne_SB - cap_sb_milieu

        prodres_k=prodres[k]

        temp=Lake.produire_minimum(k)
        prodres_k += temp
        Lake_tab[k] += temp

        stock_SB=Step.stock[k] + Battery.stock[k]
        a_decharger_SB=stock_SB - consigne_SB

        # ""

        if état == "pénurie":
            # Nuke au max
            temp=Nuclear.pilote_prod(k, Nuclear.Pout(k))
            prodres_k += temp
            Nuc_tab[k]=temp
            # print(temp)

            if prodres_k > 0:
                # reliquat on recharge
                temp, dico=recharge_plusieur_techs(
                    k, liste=[Battery, Step, Gas], astocker=prodres_k, return_dico=True, liste_dico=["Battery", "Step", "Gas"])
                Phs_tab[k] -= dico['Step']
                Battery_tab[k] -= dico['Battery']
                Gas_tab[k] -= dico['Gas']
                prodres_k -= temp
                # reliquat on risque d'écrêter : on annule le trop
                if prodres_k > 0:
                    temp=Nuclear.pilot_annule_prod(k, prodres_k)
                    prodres_k -= temp
                    Nuc_tab[k] -= temp

                surplus[k]=prodres_k

            else:
                aproduire_k=-prodres_k
                if stock_SB > 0.3 * cap_sb_max:
                    temp, dico=decharge_plusieur_techs(
                        k, liste=[Step, Battery, Lake, Gas], aproduire=aproduire_k, return_dico=True, liste_dico=["Step", "Battery", "Lake", "Gas"])
                    aproduire_k -= temp
                    Phs_tab[k] += dico['Step']
                    Battery_tab[k] += dico['Battery']
                    Lake_tab[k] += dico['Lake']
                    Gas_tab[k] += dico['Gas']
                else:
                    temp, dico=decharge_plusieur_techs(
                        k, liste=[Lake, Step, Battery, Gas], aproduire=aproduire_k, return_dico=True, liste_dico=["Lake", "Step", "Battery", "Gas"])
                    aproduire_k -= temp
                    Phs_tab[k] += dico['Step']
                    Battery_tab[k] += dico['Battery']
                    Lake_tab[k] += dico['Lake']
                    Gas_tab[k] += dico['Gas']
                manque[k]=aproduire_k

        elif état == "abondance":
            # nuke au min
            temp=Nuclear.pilote_prod(k, 0)
            prodres_k += temp
            # print(temp)
            Nuc_tab[k]=temp
            # gaz à fond
            temp=Gas.recharger(k, Gas.Pin(k))
            prodres_k -= temp
            Gas_tab[k] -= temp

            if a_decharger_SB < 0:
                # les batteries veulent remonter à 30% tant mieux !
                temp, dico=recharge_plusieur_techs(
                    k, liste=[Step, Battery], astocker=-a_decharger_SB, return_dico=True, liste_dico=['Step', 'Battery'])
                prodres_k -= temp
                #print(dico)
                Phs_tab[k] -= dico['Step']
                Battery_tab[k] -= dico['Battery']
            else:
                # on prend le risque d'écrêter
                temp, dico=decharge_plusieur_techs(
                    k, liste=[Battery, Step], aproduire=a_decharger_SB, return_dico=True, liste_dico=['Battery', 'Step'])
                prodres_k += temp
                #print(dico)
                Phs_tab[k] += dico['Step']
                Battery_tab[k] += dico['Battery']

            if prodres_k > 0:
                # on écrêtarait
                temp, dico=recharge_plusieur_techs(
                    k, liste=[Step, Battery], astocker=prodres_k, return_dico=True, liste_dico=['Step', 'Battery'])
                prodres_k -= temp
                #print(dico)
                Phs_tab[k] -= dico['Step']
                Battery_tab[k] -= dico['Battery']
                surplus[k]=prodres_k
            else:
                aproduire=-prodres_k
                # un peu de nuke pour recharger le gas et batt
                temp=Nuclear.pilote_prod(k, aproduire)
                aproduire -= temp
                Nuc_tab[k] += temp
                # on risque la pénurie finalement : on annule la production de H2
                temp=Gas.annuler_recharger(k, aanuler=aproduire)
                aproduire -= temp
                Gas_tab[k] += temp
                # on vide les batterie sous 30% puis lac puis Gas fossile
                temp, dico=decharge_plusieur_techs(
                    k, liste=[Battery, Step, Lake, Gas], aproduire=aproduire, return_dico=True, liste_dico=["Battery", "Step", "Lake", "Gas"])
                aproduire -= temp
                Phs_tab[k] += dico['Step']
                Battery_tab[k] += dico['Battery']
                Lake_tab[k] += dico['Lake']
                Gas_tab[k] += dico['Gas']

                manque[k]=aproduire

        else:
            # Normal


            # regul batteries
            if a_decharger_SB < 0:
                # les batteries veulent remonter à 50%
                temp, dico=recharge_plusieur_techs(k, liste=[Step, Battery],
                                                     astocker=-a_decharger_SB, return_dico=True, liste_dico=["Step", "Battery"])
                prodres_k -= temp
                Phs_tab[k] -= dico['Step']
                Battery_tab[k] -= dico['Battery']

            else:
                # on prend le risque d'écrêter
                temp, dico=decharge_plusieur_techs(k, liste=[Battery, Step],
                                                     aproduire=a_decharger_SB, return_dico=True, liste_dico=["Battery", "Step"])
                prodres_k += temp
                Phs_tab[k] -= dico['Step']
                Battery_tab[k] -= dico['Battery']

            # gaz à fond
            temp=Gas.recharger(k, Nuclear.Pout(k) + prodres_k)
            prodres_k -= temp
            Gas_tab[k] -= temp
            # max de Gaz que nucléaire + renouvelable permet

            # Ajout du nucléaire nécessaire
            temp=Nuclear.pilote_prod(k, -prodres_k)
            prodres_k += temp
            Nuc_tab[k] += temp

            if prodres_k > 0:
               # on écrêterait
               temp, dico=recharge_plusieur_techs(k, liste=[Step, Battery],
                                                     astocker=prodres_k, return_dico=True, liste_dico=["Step", "Battery"])
               prodres_k -= temp
               Phs_tab[k] -= dico['Step']
               Battery_tab[k] -= dico['Battery']
               surplus[k]=prodres_k
            else:
               # risque de pénurie
               temp, dico=decharge_plusieur_techs(k, liste=[Lake, Step, Battery, Gas],
                                                    aproduire=-prodres_k, return_dico=True, liste_dico=["Lake", "Step", "Battery", "Gas"])
               prodres_k += temp
               Phs_tab[k] += dico['Step']
               Battery_tab[k] += dico['Battery']
               Lake_tab[k] += dico['Lake']
               Gas_tab[k] += dico['Gas']
               manque[k]=-prodres_k
        pass

    return surplus, manque, Phs_tab, Battery_tab, Gas_tab, Lake_tab, Thermal_tab, Nuc_tab, etat_tab


def extraire_chroniques(s, p, prodres, S, B, G, L, N):
    chroniques={"s": -s, "p": p, "prodResiduelle": prodres}

    for tech in (S, B, G):
        chroniques[tech.nom[0] + "prod"]=tech.décharge
        chroniques[tech.nom[0] + "cons"]=-tech.recharge
        chroniques[tech.nom[0] + "stored"]=tech.stock

    chroniques[L.nom[0] + "prod"]=L.décharge
    chroniques[L.nom[0] + "stored"]=L.stock
    chroniques[N.nom[0] + "prod"]=N.décharge

    return chroniques


# Utilisé pour calculer la production résiduelle


# mix : données du plateau
# nb : nbr de pions de chaque catégorie
# pow dans la boucle -> puissance d'un pion

# Ici, on doit juste changer les données utilisées
def calculer_prod_non_pilot(mix, nb):

    # On récupère les facteurs de charge des données
    # A modifier je pense, en remplaçant avec les facteurs de charge généré
    fdc_on=pd.read_csv(dataPath + "mix_data/fdc_on.csv")
    fdc_off=pd.read_csv(dataPath + "mix_data/fdc_off.csv")
    fdc_pv=pd.read_csv(dataPath + "mix_data/fdc_pv.csv")


    # Puissance d'un pion
    powOnshore=1.4
    powOffshore=2.4
    powPV=3

    # On fait la somme des prods par region pour chaque techno (FacteurDeCharge * NbPions * PuissanceParPion)
    powers_renouvables={"eolienneON": 1.4,
                          "panneauPV": 3,
                          "eolienneOFF": 3}
    # note de Hugo, je ne sais pas à quoi sert cette ligne, l'effet de la carte aléa correspondant est déjà écrit à un autre endroit.
    # Alea +15% prod PV
    if "innovPV" in mix:
        fdc_pv += mix["innovPV"] * fdc_pv

    # liste des régions sans off_shore
    reg_non_off_shore=["bfc", "ara", "cvl", "idf", "est"]

    prodOnshore=np.zeros(H)
    prodOffshore=np.zeros(H)
    prodPV=np.zeros(H)

    prod={"eolienneON": np.zeros(H),
                "panneauPV": np.zeros(H),
                "eolienneOFF": np.zeros(H)}
    prod_reg={}
    for reg in mix['unites']:  # Pour chaque région dans mix['unites']
        # La production de la région est initialisée en dictionnaire
        prod_reg[reg]={}
        for p, pow in powers_renouvables.items():   # Pour chaque source renouvelable    pow ???
            if p == "eolienneOFF" and reg in reg_non_off_shore:  # Si eolienne offshore sur région non offshore -> 0
                prod_reg[reg][p]=np.zeros(H)
            else:
                # Sinon : prod_reg pour la région et la ressource = facteur de charge de la region * nb[reg][p] * pow
                prod_reg[reg][p]=np.array(fdc_on[reg]) * nb[reg][p] * pow
                # On ajoute la prod de ce moyen de prodcution dans la prod totale française du moyen de production
                prod[p] += prod_reg[reg][p]


    # carte alea MEMFDC (lance 1)
    if mix["alea"] == "MEMFDC1" or mix["alea"] == "MEMFDC2" or mix["alea"] == "MEMFDC3":
        prod_reg["cvl"]["eolienneON"] *= 0.9



    # Definition des productions electriques des rivières et lacs
    coefriv=13.
    river=pd.read_csv(dataPath + "mix_data/run_of_river.csv", header=None)
    river.columns=["heures", "prod2"]
    rivprod=np.array(river.prod2) * coefriv


    # On enregistre dans les chroniques les productions de chaque moyen de production
    chroniques={"prodOffshore": prod["eolienneOFF"],
                  "prodOnshore": prod["eolienneON"],
                  "prodPV": prod["panneauPV"],
                  "rivprod": rivprod,
                  }
    prod['regions']=prod_reg
    return chroniques, prod



def result_ressources(mix, save, nbPions, nvPions, ):

    Sol=(nbPions["eolienneON"] * 300 + nbPions["eolienneOFF"] * 400 + nbPions["panneauPV"] * 26 +
           nbPions["centraleNuc"] * 1.5 + nbPions[
               "biomasse"] * 0.8)  # occupation au sol de toutes les technologies (km2)

    Uranium=save["scores"]["Uranium"]  # disponibilite Uranium initiale
    if nbPions["centraleNuc"] > 0 or nbPions["EPR2"]:
        Uranium -= 10  # à chaque tour où on maintient des technos nucleaires
    if nvPions["EPR2"] > 0:
        Uranium -= nvPions["EPR2"]
        # carte alea MEGC (lance 2)
    if actions['alea']['actuel'] == "MEGC2" or actions['alea']['actuel'] == "MEGC3":
        Uranium -= 10

    save["scores"]["Uranium"]=Uranium  # actualisation du score Uranium

    Hydro=save["scores"]["Hydro"]  # disponibilite Hydrocarbures et Charbon
    if save["prodGazFossile"][str(mix["annee"])] > 0:
        Hydro -= 20  # à chaque tour où on consomme du gaz fossile

    # carte alea MEMP (lance 2)
    if actions['alea']['actuel'] == "MEMP2" or actions['alea']['actuel'] == "MEMP3":
        Hydro -= 20

    save["scores"]["Hydro"]=Hydro  # actualisation du score Hydro

    Bois=save["scores"]["Bois"]  # disponibilite Bois
    recup=save["scores"]["totstockbois"] - Bois

    if nbPions["biomasse"] > 0:
        Bois -= nbPions["biomasse"]
    if nbPions["biomasse"] > 0 and recup >= 0:
        # au nombre de centrales Biomasse on enlève 1 quantite de bois --> au tour suivant 1/2 des stocks sont recuperes
        Bois += 1 / 2 * recup
    # carte alea MEMP (lance 1)
    if actions['alea']['actuel'] == "MEMP1" or actions['alea']['actuel'] == "MEMP2" or actions['alea']['actuel'] == "MEMP3":
        Bois -= 20

    # carte alea MEVUAPV  (lance de 1 / 2)
    if actions['alea']['actuel'] == "MEVUAPV1" or actions['alea']['actuel'] == "MEVUAPV2" or actions['alea']['actuel'] == "MEVUPV3":
        Bois -= 10
        save["scores"]["totstockbois"] -= 10

    save["scores"]["Bois"]=Bois  # actualisation du score Bois

    dechet=save["scores"]["Dechet"]
    # dechet += nbTherm*2 + nbNuc*1 #dechets generes par les technologies Nucleaires et Thermiques
    dechet += nbPions["centraleNuc"] + nbPions["EPR2"]
    save["scores"]["Dechet"]=dechet


    result={"sol": round((Sol / 551695) * 100, 4),
              "scoreUranium": Uranium, "scoreHydro": Hydro, "scoreBois": Bois, "scoreDechets": dechet,
              }

    return result


def simulation(scenario, mix, save, nbPions, nvPions, nvPionsReg, electrolyse):
    """ Optimisation de strategie de stockage et de destockage du Mix energetique

    Args:
        scenario (array) : scenario de consommation heure par heure
        mix (dict) : donnees du plateau
        save (dict) : donnees du tour precedent
        nbPions (dict) : nombre de pions total pour chaque techno
        nvPions (dict) : nombre de nouveaux pions total pour chaque techno ce tour-ci
        nvPionsReg (dict) : nombre de pions total pour chaque techno
        electrolyse (float) : demande en electrolyse du scenar (kWh)
    Returns:
        result (dict) : dictionnaire contenant les résultats d'une seule année (result sans s à la fin)
    """

    # carte alea MEVUAPV  (lance de 1 / 2)
    # if actions['alea']['actuel'] == "MEVUAPV1" or actions['alea']['actuel'] == "MEVUAPV2" or actions['alea']['actuel'] == "MEVUAPV3":
    #     save["varConso"] = 9e4
    # scenario += np.ones(H) * (save["varConso"]/H)

    if actions['alea']['actuel'] == "MEVUAPV2" or actions['alea']['actuel'] == "MEVUAPV3":
        mix["innovPV"]=0.15

    # carte alea MEMDA (lance 3)
    if actions['alea']['actuel'] == "MEMDA3":
        scenario=0.95 * scenario

    chroniques={"demande": scenario,
                  "electrolyse": electrolyse}

    # On update les chroniques avec calculer_prod_non_pilot
    chroniques.update(calculer_prod_non_pilot(save, mix, nbPions))

    # Calcul de la production residuelle
    # prodresiduelle = prod2006_offshore + prod2006_onshore + prod2006_pv + rivprod - scenario
    prodresiduelle=chroniques["prodOffshore"] + chroniques["prodOnshore"] + \
        chroniques["prodPV"] + chroniques["rivprod"] - scenario
    # On calcule prodresiduelle avec les chroniques

    # Definition des differentes technologies

    # Techno params : name, stored, prod, etain, etaout, Q, S, vol

    S=te.TechnoStep()
    B=te.TechnoBatteries(nb_units=mix["stock"])
    G=te.TechnoGaz(nb_units=mix["methanation"])
    L=te.TechnoLacs()

    # reacteurs nucleaires effectifs qu'après 1 tour
    nbProdNuc=mix["centraleNuc"]
    # nbProdNuc2 = (nbPions["EPR2"] - nvPions["EPR2"])
    nbProdNuc2=mix["EPR2"]

    N=te.TechnoNucleaire(nb_units_EPR=nbProdNuc, nb_units_EPR2=nbProdNuc2)

    if mix["alea"] == "MEMFDC3":
        N.PoutMax *= 45 / 60

    s, p=strat_stockage(prodres=prodresiduelle, Step=S, Battery=B,
                          Gas=G, Lake=L, Nuclear=N)  # On utilise start_stockage ici

    chroniques.update(extraire_chroniques(s=s, p=p, prodres=prodresiduelle,
                                          S=S, B=B, G=G, L=L, N=N))

    result={}
    # result = result_prod_region(mix=mix, save=save, nbPions=nbPions, nvPionsReg=nvPionsReg,
    #                             chroniques=chroniques, L=L, N=N, G=G, S=S, B=B)
    # result.update(result_couts(mix, save, nbPions, nvPions, nvPionsReg, B, S, N))

    # result.update(result_ressources(mix, save, nbPions, nvPions))


    return result, save, chroniques


def simuler(demande, electrolyse, mix, nb):
    """ Optimisation de strategie de stockage et de destockage du Mix energetique

    Args:
        demande (array) : scenario de consommation heure par heure
        mix (dict) : donnees du plateau
        save (dict) : donnees du tour precedent
        nbPions (dict) : nombre de pions total pour chaque techno
        nvPions (dict) : nombre de nouveaux pions total pour chaque techno ce tour-ci
        nvPionsReg (dict) : nombre de pions total pour chaque techno
        electrolyse (float) : demande en electrolyse du scenar (kWh)
    Returns:
        result (dict) : dictionnaire contenant les résultats d'une seule année (result sans s à la fin)
    """

    # carte alea MEVUAPV  (lance de 1 / 2)
    # if mix["alea"] == "MEVUAPV1" or mix["alea"] == "MEVUAPV2" or mix["alea"] == "MEVUAPV3":
    #     save["varConso"] = 9e4
    # scenario += np.ones(H) * (save["varConso"]/H)

    if mix["alea"] == "MEVUAPV2" or mix["alea"] == "MEVUAPV3":
        mix["innovPV"]=0.15

    # carte alea MEMDA (lance 3)
    if mix["alea"] == "MEMDA3":
        demande=0.95 * demande

    # On obtient les chroniques avec calculer_prod_non_pilot
    chroniques, prod_renouvelables=calculer_prod_non_pilot(mix, nb)

    chroniques.update({"demande": demande,
                  "electrolyse": electrolyse})

    # Calcul de la production residuelle
    # prodresiduelle = prod2006_offshore + prod2006_onshore + prod2006_pv + rivprod - scenario
    prodresiduelle=chroniques["prodOffshore"] + chroniques["prodOnshore"] + chroniques["prodPV"] + chroniques[
        "rivprod"] - demande  # prodresiduelle = somme des prods dans les chroniques

    # Definition des differentes technologies

    # Techno params : name, stored, prod, etain, etaout, Q, S, vol

    S=te.TechnoStep()
    B=te.TechnoBatteries(nb_units=mix["stock"])
    G=te.TechnoGaz(nb_units=nb["methanation"])
    L=te.TechnoLacs()

    # reacteurs nucleaires effectifs qu'après 1 tour
    nbProdNuc=nb["centraleNuc"]
    # nbProdNuc2 = (nbPions["EPR2"] - nvPions["EPR2"])
    nbProdNuc2=nb["EPR2"]

    N=te.TechnoNucleaire(nb_units_EPR=nbProdNuc, nb_units_EPR2=nbProdNuc2)
    if mix["alea"] == "MEMFDC3":
        N.PoutMax *= 45. / 60.
        N.fc_nuke *= 45. / 60.


    s, p=strat_stockage(prodres=prodresiduelle, Step=S, Battery=B,
                          Gas=G, Lake=L, Nuclear=N)  # On utilise strat_stockage ici

    chroniques.update(extraire_chroniques(s=s, p=p, prodres=prodresiduelle,
                                          S=S, B=B, G=G, L=L, N=N))

    puissances={Lettre: tech.PoutMax for Lettre, tech in {
        'N': N, 'G': G, 'L': G, 'S': S, 'B': B}.items()}

    return chroniques, prod_renouvelables, puissances
