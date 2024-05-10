import os
import numpy as np
import pandas as pd



chemin = os.path.dirname(os.path.realpath(__file__))
chemin_donnees = chemin


# classe de base technologies
class Techno:
    """Classe regroupant toutes les technologies de stockage et de production pilotables

    """
    H = 24*365

    # On initialise avec les infos de base sur la techno
    def __init__(self, nom, stock, etain, etaout, PoutMax, PinMax, capacité, H=H):
        """ Création de la Technologie de stockage/déstockage/production
        Args:
            nom (str): nom de la techno
            stock (array) : niveau des stocks à chaque heure
            etain (float) : coefficient de rendement a la charge
            etaout (float) : coefficient de rendement a la decharge
            PoutMax (float) : puissance maximale de décharge (flux sortant GW)
            PinMax (float) : puissance maximale de charge (flux entrant GW)
            capacité (float) : capacite maximale de stockage (GWh)

        Returns:
            l'objet de la classe créé (Techno)
        """
        self.nom = nom  # . nom de la techno
        if stock is None :
            stock=0
        if np.isscalar(stock):
            self.stock = np.ones(H) * stock
        else:
            self.stock = stock

        self.etain = etain  # . coefficient de rendement à la charge
        self.etaout = etaout  # . coefficient de rendement à la decharge
        self.PoutMax = PoutMax  # . capacite installee (flux sortant)
        self.PinMax = PinMax  # . capacite de charge d'une techno de stockage (flux entrant)
        self.capacité = capacité  # . Capacite maximale de stockage de la techno (Volume max)
        self.H = H
        self.set_décharge()
        self.set_recharge()


    def set_décharge(self, décharge=0):
        if np.isscalar(décharge):
            self.décharge = np.ones(self.H) * décharge
        else:
            self.décharge = décharge

    def set_recharge(self, recharge=0):
        if np.isscalar(recharge):
            self.recharge = np.ones(self.H) * recharge
        else:
            self.recharge = recharge

    def get_pertes(self):
        return self.décharge / self.etaout * (1 - self.etaout) + self.recharge * (1-self.etain)

    def Pout(self, k):
        """Renvoie la puissance maximum de décharge à l'heure k """
        return min(self.stock[k] * self.etaout, self.PoutMax-self.décharge[k]+self.recharge[k])

    def Pin(self, k):
        """Renvoie la puissance maximum de recharge à l'heure k """
        return min((self.capacité-self.stock[k]) / self.etain, self.PinMax-self.recharge[k]+self.recharge[k])

    def annuler_recharger(self, k, aanuler):
        """

        """
        vaannuler = min(aanuler, self.recharge[k])
        self.recharge[k] -= vaannuler
        self.stock[k:] -= vaannuler * self.etain
        return vaannuler

    def annuler_décharger(self, k, aanuler):
        # annulation d'ordre
        vaannuler = min(aanuler, self.décharge[k])
        self.décharge[k] -= vaannuler
        self.stock[k:] += vaannuler / self.etaout
        return vaannuler

    def recharger(self, k, astocker):
        """Recharge les moyens de stockage quand on a trop d'energie

        Args:
            k (int) : heure courante
            astocker (float) : qte d'energie a stocker (GWh/h)

        Returns:
            out (flout) : partie de astocker qui n'a pas pu être stockée
        """
        if astocker < 0:

            out = 0

        else:
            relargue = 0
            if self.décharge[k] > 0:
                relargue = self.annuler_décharger(k, astocker)
                astocker -= relargue

            vastoker = min(astocker , self.Pin(k))
            self.stock[k:] = self.stock[k] + vastoker * self.etain
            self.recharge[k] += vastoker
            out = vastoker + relargue

        return out


    def décharger(self, k, aproduire):
        """ Decharge les moyens de stockage quand on a besoin d'energie

        Args:
            k (int) : heure courante
            aproduire (float) : qte d'energie à récupérer

        Return:
            out (float) : le reste de ce qui n'a pas été produit
        """
        if aproduire <= 0:
            out = 0

        else:
            relargue = 0
            if self.recharge[k] > 0:
                relargue = self.annuler_recharger(k, aproduire)
                aproduire -= relargue

            vaproduire = min(aproduire, self.Pout(k))

            self.stock[k:] = self.stock[k] - vaproduire/self.etaout
            self.décharge[k] += vaproduire

            out = vaproduire + relargue

        return out


#Technologie Step
class TechnoStep(Techno):
    etain = 0.95
    etaout = 0.9
    PoutMax = 9.3
    PinMax = 9.3
    capacité = 180

    def __init__(self, nom='Step', stock=16,
                 etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=PinMax, capacité=capacité, H=Techno.H):
        super().__init__(nom=nom, stock=stock, etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=PinMax, capacité=capacité,
                         H=H)

#Technologie batterie
class TechnoBatteries(Techno):
    """
    Calculé pour une capacité totale de 10 unités
    """
    etain = 0.95
    etaout = 0.9
    PoutMaxParUnite = 20.08/10.
    PinMaxParUnite = PoutMaxParUnite
    capaciteParUnite = 74.14/10.

    def __init__(self, nom='Batteries', stock=None,
                 etain=0.95, etaout=0.9, PoutMax=None, PinMax=None,
                 capacité=None, nb_units=1, H=Techno.H):

        self.nb_units = nb_units
        if PoutMax is None:
            PoutMax = nb_units * TechnoBatteries.PoutMaxParUnite
        if PinMax is None:
            PinMax = nb_units * TechnoBatteries.PinMaxParUnite
        if capacité is None:
            capacité = nb_units * TechnoBatteries.capaciteParUnite
        if stock is None:
            stock = capacité/2.

        super().__init__(nom=nom, stock=stock,
                         etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=PinMax,
                         capacité=capacité, H=H)


# Technologie gaz
class TechnoGaz(Techno):
    capacité = 10000000.
    init_gaz = capacité / 2.
    # Methanation : 1 pion = 10 unites de 100 MW = 1 GW
    # T = Techno('Centrale thermique', None, np.zeros(H), None, 1, 0.7725*nbTherm, None, None)
    # Puissance : 1.08 GWe (EDF)
    # Meme rendement
    etain = 0.59
    etaout = 0.45
    PoutMax = 34.44
    PinMaxParUnite = 1.
    def __init__(self, nom='Gaz', stock=init_gaz,
                 etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=None,
                 capacité=capacité, nb_units=0, H=Techno.H):
        self.nb_units = nb_units

        if PinMax is None:
            PinMax = nb_units * TechnoGaz.PinMaxParUnite

        super().__init__(nom=nom, stock=stock,
                         etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=PinMax,
                         capacité=capacité, H=H)


# Technologie lacs : on change rien
class TechnoLacs(Techno):
    # Puissance centrales territoire : 18.54 GWe repartis sur 24 centrales (EDF)
    # Rendement meca (inutile ici) : ~35% generalement (Wiki)
    duree_mois = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])*24
    etain = 1
    etaout = 1
    PoutMax = 18.54
    PinMax = 18.54
    capacité = TechnoStep.capacité

    def __init__(self, nom='Lacs', stock=None,
                 etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=PinMax,
                 capacité=capacité, H=Techno.H):


        super().__init__(nom=nom, stock=stock,
                         etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=PinMax,
                         capacité=capacité, H=H)

        if stock is None:
            self.set_stock_et_cons_from_csv()

    def Pout(self, k):
        """Renvoie la puissance maximum de décharge à l'heure k """
        return min(self.stock[k] * self.etaout, self.PoutMax-self.décharge[k])

    def set_stock_et_cons_from_csv(self, fichier=chemin_donnees + "/lake_inflows.csv"):
        lake = pd.read_csv(fichier, header=None)
        lake.columns = ["month", "prod2"]
        lakeprod = np.array(lake.prod2)

        # Calcul de ce qui est stocke dans les lacs pour chaque mois
        self.stock = np.ones(self.H)*self.capacité
        k = 0
        for m in range(12):
            ksuiv = k + TechnoLacs.duree_mois[m]
            self.recharge[k:ksuiv] = lakeprod[m]
            k = ksuiv

    def recharger(self, k):
        max_charge = self.capacité - self.stock[k]
        recharge = min(max_charge, self.recharge[k])
        self.stock[k:] += recharge
        self.recharge[k] -= recharge
        return self.recharge[k]

    def produire_minimum(self, k):
        self.stock[k:] += self.recharge[k]
        produit = self.décharger(k, self.recharge[k])
        self.recharge[k] -= produit
        return produit

    def décharger(self, k, aproduire):
        """ Decharge les moyens de stockage quand on a besoin d'energie

        Args:
            self : technologie de stockage a utiliser pour la production (batterie, phs, ...)
            k (int) : heure courante
            aproduire (float) : qte d'energie a fournir
            endmonthlake (array) : qte d'energie restante dans les lacs jusqu'a la fin du mois
            executer (boolean) : indique si l'energie dechargee est a prendre en compte pour la production globale (faux pour les echanges internes)
        """
        if aproduire <= 0:
            out = 0

        else:

            vaproduire = min(aproduire, self.Pout(k))

            self.stock[k:] = self.stock[k] - vaproduire/self.etaout
            self.décharge[k] += vaproduire

            out = vaproduire

        return out

# Ancienne fonction de lacs

class OldTechnoLacs(Techno):
    # Puissance centrales territoire : 18.54 GWe repartis sur 24 centrales (EDF)
    # Rendement meca (inutile ici) : ~35% generalement (Wiki)

    horlake = np.array(
        [0, 31,
         31 + 28,
         31 + 28 + 31,
         31 + 28 + 31 + 30,
         31 + 28 + 31 + 30 + 31,
         31 + 28 + 31 + 30 + 31 + 30,
         31 + 28 + 31 + 30 + 31 + 30 + 31 ,
         31 + 28 + 31 + 30 + 31 + 30 + 31 + 31,
         31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30,
         31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 ,
         31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30,
         31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31]) * 24


    def __init__(self, nom='Lacs', stock=None,
                 etain=1, etaout=1, PoutMax=10, PinMax=10,
                 capacité=2000, H=Techno.H):

        self.endmonthlake = np.zeros(H)
        for k in range(12):
            self.endmonthlake[TechnoLacs.horlake[k]:TechnoLacs.horlake[k + 1]] = int(TechnoLacs.horlake[k + 1])

        super().__init__(nom=nom, stock=stock,
                         etain=etain, etaout=etaout, PoutMax=PoutMax, PinMax=PinMax,
                         capacité=capacité, H=H)

        if stock is None:
            self.set_stock_from_csv()

    def set_stock_from_csv(self, fichier=chemin_donnees+"/lake_inflows.csv"):
        lake = pd.read_csv(fichier, header=None)
        lake.columns = ["month", "prod2"]
        lakeprod = np.array(lake.prod2)

        # Calcul de ce qui est stocke dans les lacs pour chaque mois
        self.stock = np.zeros(self.H)
        for k in range(12):
            self.stock[TechnoLacs.horlake[k]:TechnoLacs.horlake[k + 1]] = 1000. * lakeprod[k]


    def décharger(self, k, aproduire):
        """ Decharge les moyens de stockage quand on a besoin d'energie

        Args:
            self : technologie de stockage a utiliser pour la production (batterie, phs, ...)
            k (int) : heure courante
            aproduire (float) : qte d'energie a fournir
            endmonthlake (array) : qte d'energie restante dans les lacs jusqu'a la fin du mois
            executer (boolean) : indique si l'energie dechargee est a prendre en compte pour la production globale (faux pour les echanges internes)
        """
        if aproduire <= 0:
            out = 0

        else:

            vaproduire = min(aproduire, self.Pout(k))

            self.stock[k:int(self.endmonthlake[k])] = self.stock[k] - vaproduire/self.etaout
            self.décharge[k] = vaproduire

            out = vaproduire

        return out


# Nucléaire : on change rien

def fc_min_max_nuke(k, Pmax=1):
    """ Renvoie la puissance dispo actuellement pour le nucleaire, par rapport a la puissance max



    Args:
        k (int) : heure courante

    Sur ATH : pas de penurie avec 56 centrales min.
    Sur ATL : 28 centrales min. (50%)
    Diviser en 4 ou 8 groupes (plutôt 8 pour les besoins humains)
    1/8 = 0.125, 7/8 = 0.875
    2180, 2920, 3650, 4400, 5130, 5900, 6732, 7580
    Tiers de 8760 : 2920(4*730), 5840, 8460
    DANS le dernier tiers : 50% croissance lineaire min, 25% baisse de 20% prod min/max, 25% arret
    """

    # La production nucleaire est divisee en 8 groupes, chacun a son cycle d'arret.
    # Dans cette fonction, on regarde dans quel partie du cycle on est pour chaque groupe,
    # pour calibrer la production min et max.

    H = 8760
    N = 8
    n = 1 / N

    # Intervalles des 3 parties importantes du cycle, pour chaque groupe
    A_ranges = [((2180 + 6570) % H, (2180 + 8030) % H), ((2920 + 6570) % H, (2920 + 8030) % H),
                ((3650 + 6570) % H, (3650 + 8030) % H), ((4400 + 6570) % H, (4400 + 8030) % H),
                ((5130 + 6570) % H, (5130 + 8030) % H), ((5900 + 6570) % H, (5900 + 8030) % H),
                ((6732 + 6570) % H, (6732 + 8030) % H), ((7580 + 6570) % H, (7580 + 8030) % H)]

    B_ranges = [((2180 + 8030) % H, 2180), ((2920 + 8030) % H, 2920), ((3650 + 8030) % H, 3650),
                ((4400 + 8030) % H, 4400), ((5130 + 8030) % H, 5130), ((5900 + 8030) % H, 5900),
                ((6732 + 8030) % H, 6732), ((7580 + 8030) % H, 7580)]

    C_ranges = [(2180, (2180 + 730) % H), (2920, (2920 + 730) % H), (3650, (3650 + 730) % H),
                (4400, (4400 + 730) % H), (5130, (5130 + 730) % H), (5900, (5900 + 730) % H),
                (6732, (6732 + 730) % H), (7580, (7580 + 730) % H)]

    inA = [lower <= k < upper for (lower, upper) in A_ranges]
    inB = [lower <= k < upper for (lower, upper) in B_ranges]
    inC = [lower <= k < upper for (lower, upper) in C_ranges]

    sMin = 0
    sMax = 0

    # Pour chaque groupe, on regarde sa zone et on ajuste son min et son max
    for i in range(N):
        if inA[i]:
            start = A_ranges[i][0]
            sMin += n * (0.2 + 0.00054795 * (k - start))
            sMax += n * 1
        elif inB[i]:
            start = B_ranges[i][0]
            sMin += n * (1 - 0.00027397260274 * (k - start))
            sMax += n * (1 - 0.00027397260274 * (k - start))
        elif inC[i]:
            sMin += n * 0
            sMax += n * 0
        else:
            sMin += n * 0.2
            sMax += n * 1

    return sMin*Pmax, sMax*Pmax

class TechnoNucleaire(Techno):
    PoutMaxParUniteEPR2 = 1.67
    PoutMaxParUniteEPR = 1.08

    ramp_up = 0.25           # % de PoutMax /heure
    ramp_down = ramp_up

    def __init__(self, nom='Nucléaire', PoutMax=None, décharge_init=None,
                 nb_units_EPR=0, nb_units_EPR2=0, H=Techno.H, ramp_up=ramp_up, ramp_down=ramp_down):

        self.nb_units_EPR = nb_units_EPR
        self.nb_units_EPR2 = nb_units_EPR2

        if PoutMax is None:
            PoutMax = (TechnoNucleaire.PoutMaxParUniteEPR * nb_units_EPR +
                       TechnoNucleaire.PoutMaxParUniteEPR2 * nb_units_EPR2)

        self.ramp_up = ramp_up * PoutMax            #GW/h
        self.ramp_down = ramp_down * PoutMax        #GW/h

        super().__init__(nom=nom, stock=None,
                         etain=None, etaout=1, PoutMax=PoutMax, PinMax=None,
                         capacité=None, H=H)

        if décharge_init is not None:
            self.set_décharge_init(décharge_init)

        self.fc_nuke = [fc_min_max_nuke(k, self.PoutMax) for k in range(self.H)]

    def set_décharge_init(self, décharge_init):
        if np.isscalar(décharge_init):
            self.décharge = décharge_init*np.ones(self.H)
        else:
            self.décharge = décharge_init


    def get_pout_effectives(self, k):
        return self.fc_nuke[k]

    def p_min_effective(self, k):
        return self.fc_nuke[k][0]
    def p_max_effective(self, k):
        return self.fc_nuke[k][1]

    def Pout(self, k):
        if k > 0:
            return min(self.p_max_effective(k) - self.décharge[k],
                       self.décharge[k-1]+self.ramp_up - self.décharge[k])
        else:
            return self.p_max_effective(k) - self.décharge[k]

    def Pout_min(self, k):
        if k > 0:
            return max(self.p_min_effective(k) - self.décharge[k],
                       self.décharge[k-1] - self.ramp_down - self.décharge[k])
        else:
            return self.p_min_effective(k) - self.décharge[k]

    def pilote_prod(self, k, aproduire):
        """ Lance la production des centrales nucleaires

        Args:
            self : objet à utiliser pour la production  (ici, Techno Nucleaire)
            k (int) : heure courante
            aproduire (float) : qte d'energie a fournir
        """
        # Si la demande est trop faible ou nulle, on produit quand meme à hauteur de 20%

        le_max = self.Pout(k)
        le_min = self.Pout_min(k)

        if aproduire > le_max:
            temp = le_max
        elif aproduire > le_min:
            temp = aproduire
        else:
            temp = le_min

        self.décharge[k] += temp

        return temp

    def pilot_annule_prod(self, k, aannuler):
        vaannuler = min(aannuler, - self.Pout_min(k))
        self.décharge[k] -= vaannuler
        return vaannuler

# Production centrales thermiques

def thermProd(tec, k, aproduire):
    """ Lance la production des centrales thermiques

    Args:
        tec : objet à utiliser pour la production  (ici, Techno Thermique)
        k (int) : heure courante
        aproduire (float) : qte d'energie a fournir
    """
    if aproduire <= 0:
        out = 0

    else:
        temp = min(aproduire / tec.etaout, tec.PoutMax / tec.etaout)
        tec.décharge[k] = temp * tec.etaout
        out = aproduire - tec.décharge[k]

    return out


def test_technoLacs():
    import matplotlib.pyplot as plt
    L = TechnoLacs()
    plt.title("Test des lacs")
    plt.subplot(313)
    plt.grid()
    plt.plot(L.recharge, 'k--')
    plt.ylabel('stock avant et après')
    reste = np.zeros(Techno.H)
    aproduire = reste.copy()
    for k in range(Techno.H):
        L.recharger(k)
        L.produire_minimum(k)
        if 2000 < k < 3600:
            val=15
            aproduire[k] +=val
            reste[k] += val - L.décharger(k=k, aproduire=val)

        if 6000 < k < 6400:
            val = 1
            aproduire[k] +=val
            reste[k] += val - L.décharger(k=k, aproduire=val)

        if 6200 < k < 6800:
            val = 1
            aproduire[k] +=val
            reste[k] += val- L.décharger(k=k, aproduire=val)

        if 4000 < k < 5000:
            val = 10
            aproduire[k] += val
            reste[k] += val - L.décharger(k=k, aproduire=val)

        if 8000 < k < 8500:
            val = -5
            aproduire[k] += val
            reste[k] += val - L.décharger(k=k, aproduire=val)
        else:
            val = 0
            aproduire[k] += val
            reste[k] += val - L.décharger(k=k, aproduire=0)

    plt.subplot(311)
    plt.plot(L.stock, 'r')
    plt.subplot(312)
    plt.plot(aproduire, 'b--')
    plt.plot(reste, 'r')
    plt.legend(['aproduire', 'reste'])
    plt.grid()
    plt.subplot(313)
    plt.plot(L.décharge, 'g')
    plt.plot(aproduire-reste, 'b.')
    plt.legend(['décharge(prod)', 'aprod - reste'])
    plt.show()

def test_generique(T=None):
    import matplotlib.pyplot as plt
    N= 50

    if T is None:
        T= Techno(nom="generique", stock=50, etain=0.75, etaout=0.25, PoutMax=1, PinMax=10, capacité=100, H=50)

    astocker = np.zeros(N)
    reste = np.zeros(N)
    Pin = np.zeros(N)
    Pout = np.zeros(N)

    for k in range(N):
        val = 0
        if 10 < k <= 20:
            val = (T.capacité-T.stock[0])/10/T.etain *2
        if 30 < k <= 40:
            val = - T.capacité /10*T.etaout *2

        if val>0:
            reste[k] = val - T.recharger(k,val)
        else:
            reste[k] = -val -T.décharger(k, -val)

        start_surcroit = 32
        if start_surcroit< k <=start_surcroit+10 :
            val += -val / 2.
            reste[k] = val - T.recharger(k, val )

        if start_surcroit+5< k <=start_surcroit+10 :
            val += -val * 2
            reste[k] = val - T.recharger(k, val )

        astocker[k] = val

        Pin[k] += T.Pin(k)
        Pout[k] += T.Pout(k)

    plt.subplot(311)
    plt.plot(T.stock, 'k--')
    plt.grid()
    plt.subplot(312)
    plt.plot(astocker, 'k--')
    plt.plot(-T.décharge,'b')
    plt.plot(T.recharge, 'r')
    plt.plot(reste, 'g--')

    plt.subplot(313)
    plt.plot(astocker, 'k--')
    plt.plot(reste, 'g--')
    plt.show()

def test_technoNucleaire():
    import matplotlib.pyplot as plt

    N = TechnoNucleaire(nb_units_EPR=46, nb_units_EPR2=0, décharge_init=0.1)
    fc_nuke = [N.get_pout_effectives(k) for k in range(1, 365 * 24)]
    aproduire = np.zeros(N.H)
    reste = np.zeros(N.H)
    Pout = np.zeros(N.H)

    for k in range(N.H):
        aproduire[k]=0.2*N.PoutMax
        if 500 < k <= 2000:
            aproduire[k] = aproduire[k-1] + 1/1000.*N.PoutMax
        if 2000 < k <= 3500:
            aproduire[k] = aproduire[k - 1] - 1 / 1000.*N.PoutMax

        reste[k] +=  aproduire[k] - N.pilote_prod(k, aproduire[k])
        if 5000 < k <= 8000:
            reste[k] += 0.4*N.PoutMax - N.pilote_prod(k, 0.4*N.PoutMax)
        if 6000 < k <= 8000:
            reste[k] += 0.2*N.PoutMax - N.pilote_prod(k, 0.2*N.PoutMax)
        if 7000 < k <= 8000:
            reste[k] += 0.4*N.PoutMax - N.pilote_prod(k, 0.4*N.PoutMax)

        Pout[k] = N.Pout(k)

    plt.subplot(211)
    plt.plot(fc_nuke)
    plt.plot(N.décharge,'b-+')
    plt.grid()
    plt.subplot(212)
    plt.plot(aproduire, 'k')
    plt.plot(reste, 'g')
    plt.plot(Pout,'r')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_technoLacs()
    #test_technoNucleaire()
    #test_generique()
    #test_generique(TechnoStep(H=50))
    #test_generique(TechnoBatteries(nb_units=1, H=50))
    #test_generique(TechnoGaz(nb_units=10, H=50))