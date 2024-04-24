import os
import numpy as np
import pandas as pd

import climix.technologies as te
dataPath = os.path.dirname(os.path.realpath(__file__))+"/"

#########################
### TOUT EST EN GW(h) ###
#########################

# Nombre d'heures dans l'année
H = (24 * 365)


class PredicteurGlissant():
    def __init__(self, méthode, horizon=24, cyclique=H):
        self.horizon = horizon*10
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


def recharge_plusieur_techs(k, liste, astocker):
    astocker_init = astocker
    for tec in liste:
        astocker -= tec.recharger(k=k, astocker=astocker)
    return astocker_init - astocker


def decharge_plusieur_techs(k, liste, aproduire):
    aproduire_init = aproduire
    for tec in liste:
        aproduire -= tec.décharger(k=k, aproduire=aproduire)
    return aproduire_init - aproduire


def strat_stockage(prodres, Step, Battery, Gas, Lake, Nuclear):
    """

    """
    pred_nuke24_min = PredicteurGlissant(Nuclear.p_min_effective)
    pred_muke24_max = PredicteurGlissant(Nuclear.p_max_effective)
    pred_prodres24 = PredicteurGlissant(lambda k: prodres[k])

    cap_sb_max = Step.capacité + Battery.capacité
    cap_sb_milieu = 0.5 * cap_sb_max
    cap_sb_abondance = 0
    cap_sb_pénurie = cap_sb_max
    sb_écart = 0

    tecstock = {"Battery": Battery, "Step": Step}

    tecdestock = {"Lake": Lake, "Step": Step, "Battery": Battery}

    surplus = np.zeros(len(prodres))
    manque = np.zeros(len(prodres))
    for k in range(H):
        nuke24min = pred_nuke24_min.__next__()
        nuke24max = pred_muke24_max.__next__()
        prodres24 = pred_prodres24.__next__()
        Lake.recharger(k)
        if prodres24 + nuke24max < 0:
            état = "pénurie"
            consigne_SB = cap_sb_pénurie

        elif prodres24 + nuke24min > 0:
            état = "abondance"
            consigne_SB = cap_sb_abondance

        else:
            état = "flexible"
            consigne_SB = cap_sb_milieu + (sb_écart * 0.99)

        sb_écart = consigne_SB - cap_sb_milieu

        prodres_k = prodres[k]

        prodres_k += Lake.produire_minimum(k)

        stock_SB = Step.stock[k] + Battery.stock[k]
        a_decharger_SB = stock_SB - consigne_SB

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
                #on écrêtarait
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
            # Normal


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

            prodres_k += Nuclear.pilote_prod(k, -prodres_k)

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


def extraire_chroniques(s, p, prodres, S, B, G, L, N):
    chroniques = {"s": -s, "p": p, "prodResiduelle": prodres}

    for tech in (S, B, G):
        chroniques[tech.nom[0] + "prod"] = tech.décharge
        chroniques[tech.nom[0] + "cons"] = -tech.recharge
        chroniques[tech.nom[0] + "stored"] = tech.stock

    chroniques[L.nom[0] + "prod"] = L.décharge
    chroniques[L.nom[0] + "stored"] = L.stock
    chroniques[N.nom[0] + "prod"] = N.décharge

    return chroniques


def calculer_prod_non_pilot(mix, nb):

    fdc_on = pd.read_csv(dataPath + "mix_data/fdc_on.csv")
    fdc_off = pd.read_csv(dataPath + "mix_data/fdc_off.csv")
    fdc_pv = pd.read_csv(dataPath + "mix_data/fdc_pv.csv")


    # Puissance d'un pion
    powOnshore = 1.4
    powOffshore = 2.4
    powPV = 3

    # On fait la somme des prods par region pour chaque techno (FacteurDeCharge * NbPions * PuissanceParPion)
    powers_renouvables = {"eolienneON": 1.4,
                          "panneauPV": 3,
                          "eolienneOFF": 3}
    # note de Hugo, je ne sais pas à quoi sert cette ligne, l'effet de la carte aléa correspondant est déjà écrit à un autre endroit.
    # Alea +15% prod PV
    if "innovPV" in mix:
        fdc_pv += mix["innovPV"] * fdc_pv

    reg_non_off_shore = [ "bfc", "ara", "cvl", "idf", "est"]

    prodOnshore = np.zeros(H)
    prodOffshore = np.zeros(H)
    prodPV = np.zeros(H)

    prod = {"eolienneON": np.zeros(H),
                "panneauPV": np.zeros(H),
                "eolienneOFF": np.zeros(H)}
    prod_reg={}
    for reg in mix['unites']:
        prod_reg[reg] = {}
        for p, pow in powers_renouvables.items():
            if p=="eolienneOFF" and reg in reg_non_off_shore:
                prod_reg[reg][p] = np.zeros(H)
            else:
                prod_reg[reg][p] = np.array(fdc_on[reg]) * nb[reg][p] * pow
                prod[p] += prod_reg[reg][p]


    # carte alea MEMFDC (lance 1)
    if mix["alea"] == "MEMFDC1" or mix["alea"] == "MEMFDC2" or mix["alea"] == "MEMFDC3":
        prod_reg["cvl"]["eolienneON"] *= 0.9



    # Definition des productions electriques des rivières et lacs
    coefriv = 13.
    river = pd.read_csv(dataPath + "mix_data/run_of_river.csv", header=None)
    river.columns = ["heures", "prod2"]
    rivprod = np.array(river.prod2) * coefriv


    chroniques = {"prodOffshore": prod["eolienneOFF"],
                  "prodOnshore": prod["eolienneON"],
                  "prodPV": prod["panneauPV"],
                  "rivprod": rivprod,
                  }
    prod['regions']=prod_reg
    return chroniques, prod



def result_ressources(mix, save, nbPions, nvPions, ):

    Sol = (nbPions["eolienneON"] * 300 + nbPions["eolienneOFF"] * 400 + nbPions["panneauPV"] * 26 +
           nbPions["centraleNuc"] * 1.5 + nbPions[
               "biomasse"] * 0.8)  # occupation au sol de toutes les technologies (km2)

    Uranium = save["scores"]["Uranium"]  # disponibilite Uranium initiale
    if nbPions["centraleNuc"] > 0 or nbPions["EPR2"]:
        Uranium -= 10  # à chaque tour où on maintient des technos nucleaires
    if nvPions["EPR2"] > 0:
        Uranium -= nvPions["EPR2"]
        # carte alea MEGC (lance 2)
    if actions['alea']['actuel'] == "MEGC2" or actions['alea']['actuel'] == "MEGC3":
        Uranium -= 10

    save["scores"]["Uranium"] = Uranium  # actualisation du score Uranium

    Hydro = save["scores"]["Hydro"]  # disponibilite Hydrocarbures et Charbon
    if save["prodGazFossile"][str(mix["annee"])] > 0:
        Hydro -= 20  # à chaque tour où on consomme du gaz fossile

    # carte alea MEMP (lance 2)
    if actions['alea']['actuel'] == "MEMP2" or actions['alea']['actuel'] == "MEMP3":
        Hydro -= 20

    save["scores"]["Hydro"] = Hydro  # actualisation du score Hydro

    Bois = save["scores"]["Bois"]  # disponibilite Bois
    recup = save["scores"]["totstockbois"] - Bois

    if nbPions["biomasse"] > 0:
        Bois -= nbPions["biomasse"]
    if nbPions["biomasse"] > 0 and recup >= 0:
        Bois += 1 / 2 * recup  # au nombre de centrales Biomasse on enlève 1 quantite de bois --> au tour suivant 1/2 des stocks sont recuperes
    # carte alea MEMP (lance 1)
    if actions['alea']['actuel'] == "MEMP1" or actions['alea']['actuel'] == "MEMP2" or actions['alea']['actuel'] == "MEMP3":
        Bois -= 20

    # carte alea MEVUAPV  (lance de 1 / 2)
    if actions['alea']['actuel'] == "MEVUAPV1" or actions['alea']['actuel'] == "MEVUAPV2" or actions['alea']['actuel'] == "MEVUPV3":
        Bois -= 10
        save["scores"]["totstockbois"] -= 10

    save["scores"]["Bois"] = Bois  # actualisation du score Bois

    dechet = save["scores"]["Dechet"]
    # dechet += nbTherm*2 + nbNuc*1 #dechets generes par les technologies Nucleaires et Thermiques
    dechet += nbPions["centraleNuc"] + nbPions["EPR2"]
    save["scores"]["Dechet"] = dechet


    result = {"sol": round((Sol / 551695) * 100, 4),
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
        mix["innovPV"] = 0.15

    # carte alea MEMDA (lance 3)
    if actions['alea']['actuel'] == "MEMDA3":
        scenario = 0.95 * scenario

    chroniques = {"demande": scenario,
                  "electrolyse": electrolyse}

    chroniques.update(calculer_prod_non_pilot(save, mix, nbPions))

    # Calcul de la production residuelle
    # prodresiduelle = prod2006_offshore + prod2006_onshore + prod2006_pv + rivprod - scenario
    prodresiduelle = chroniques["prodOffshore"] + chroniques["prodOnshore"] + chroniques["prodPV"] + chroniques["rivprod"] - scenario

    # Definition des differentes technologies

    # Techno params : name, stored, prod, etain, etaout, Q, S, vol

    S = te.TechnoStep()
    B = te.TechnoBatteries(nb_units=mix["stock"])
    G = te.TechnoGaz(nb_units=mix["methanation"])
    L = te.TechnoLacs()

    # reacteurs nucleaires effectifs qu'après 1 tour
    nbProdNuc = mix["centraleNuc"]
    #nbProdNuc2 = (nbPions["EPR2"] - nvPions["EPR2"])
    nbProdNuc2 = mix["EPR2"]

    N = te.TechnoNucleaire(nb_units_EPR=nbProdNuc, nb_units_EPR2=nbProdNuc2)

    if mix["alea"] == "MEMFDC3":
        N.PoutMax *= 45 / 60

    s, p = strat_stockage(prodres=prodresiduelle, Step=S, Battery=B,
                          Gas=G, Lake=L, Nuclear=N)

    chroniques.update(extraire_chroniques(s=s, p=p, prodres=prodresiduelle,
                                          S=S, B=B, G=G, L=L, N=N))

    result= {}
    #result = result_prod_region(mix=mix, save=save, nbPions=nbPions, nvPionsReg=nvPionsReg,
    #                             chroniques=chroniques, L=L, N=N, G=G, S=S, B=B)
    #result.update(result_couts(mix, save, nbPions, nvPions, nvPionsReg, B, S, N))

    #result.update(result_ressources(mix, save, nbPions, nvPions))


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
        mix["innovPV"] = 0.15

    # carte alea MEMDA (lance 3)
    if mix["alea"] == "MEMDA3":
        demande = 0.95 * demande

    chroniques, prod_renouvelables = calculer_prod_non_pilot(mix, nb)

    chroniques.update({"demande": demande,
                  "electrolyse": electrolyse})

    # Calcul de la production residuelle
    # prodresiduelle = prod2006_offshore + prod2006_onshore + prod2006_pv + rivprod - scenario
    prodresiduelle = chroniques["prodOffshore"] + chroniques["prodOnshore"] + chroniques["prodPV"] + chroniques[
        "rivprod"] - demande

    # Definition des differentes technologies

    # Techno params : name, stored, prod, etain, etaout, Q, S, vol

    S = te.TechnoStep()
    B = te.TechnoBatteries(nb_units=mix["stock"])
    G = te.TechnoGaz(nb_units=nb["methanation"])
    L = te.TechnoLacs()

    # reacteurs nucleaires effectifs qu'après 1 tour
    nbProdNuc = nb["centraleNuc"]
    # nbProdNuc2 = (nbPions["EPR2"] - nvPions["EPR2"])
    nbProdNuc2 = nb["EPR2"]

    N = te.TechnoNucleaire(nb_units_EPR=nbProdNuc, nb_units_EPR2=nbProdNuc2)
    if mix["alea"] == "MEMFDC3":
        N.PoutMax *= 45. / 60.
        N.fc_nuke *= 45. / 60.


    s, p = strat_stockage(prodres=prodresiduelle, Step=S, Battery=B,
                          Gas=G, Lake=L, Nuclear=N)

    chroniques.update(extraire_chroniques(s=s, p=p, prodres=prodresiduelle,
                                          S=S, B=B, G=G, L=L, N=N))

    puissances = { Lettre: tech.PoutMax for Lettre, tech in {'N':N, 'G':G, 'L':G, 'S':S, 'B':B}.items()}

    return chroniques, prod_renouvelables, puissances
