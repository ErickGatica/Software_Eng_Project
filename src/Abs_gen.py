"""
This is the script with the functions to generate the Absorption spectra of the molecules

It will take temperature, pressure, molecule, and other parameters as input and return the absorption spectra of the molecule
The absorption spectra will be generated using the HITRAN database and the HITRAN API

"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hapi import *

# Impor the variables from the Variables.py script
from Variables import args
from Variables import args_dict

# Constant
R=8.31446261815324 #J/(mol K)

# Functions

def spectrum(P,T,length,numin,numax,molecule_id,isotopo_id,method_,wavestep,molar):

    molecule=moleculeName(molecule_id); #Getting the name of the molecule from the id
    #if isotopo_id is None:
    #    isotopo_id = 1

    name_isoto=isotopologueName(molecule_id,isotopo_id); #Getting the name of the isotopologue from the id

    fetch(molecule,molecule_id,isotopo_id,numin,numax)
    # Lets go with the data
    if method_=="HT":
        nu,coef=absorptionCoefficient_HT(SourceTables=molecule,WavenumberStep=wavestep,Diluent={'self':molar,'air':1-molar},Environment={'T':T,'p':P,'l':length})
    elif method_=="V":
        nu,coef=absorptionCoefficient_Voigt(SourceTables=molecule,WavenumberStep=wavestep,Diluent={'self':molar,'air':1-molar},Environment={'T':T,'p':P,'l':length})
    elif method_=="L":
        nu,coef=absorptionCoefficient_Lorentz(SourceTables=molecule,WavenumberStep=wavestep,Diluent={'self':molar,'air':1-molar},Environment={'T':T,'p':P,'l':length})
    elif method_=="D":
        nu,coef=absorptionCoefficient_Doppler(SourceTables=molecule,WavenumberStep=wavestep,Diluent={'self':molar,'air':1-molar},Environment={'T':T,'p':P,'l':length})

#nu,coef=absorptionCoefficient_Gauss(SourceTables=molecule,Diluent={'air':1.0},Environment={'T':T,'p':P,'l':length})
#-> absorptionCoefficient_HT
#-> absorptionCoefficient_Voigt
#-> absorptionCoefficient_Lorentz
#-> absorptionCoefficient_Doppler

    #Getting the spectrums
    # Lets go with absorption Spectrum
    nu,absorp=absorptionSpectrum(nu,coef)
    # Lets go with transmittance Spectrum
    nu,trans=transmittanceSpectrum(nu,coef)
    # Lets go with radiance Spectrum
    nu,radi = radianceSpectrum(nu,coef)

    # Transforming the coef data to absorption
    

    # Lets define a class to save the data
    class Data_spectrum:
        def __init__(self, nu, coef, absorp, trans, radi,name_isoto):
            self.nu = nu
            self.coef = coef
            self.absorp = absorp
            self.trans = trans
            self.radi = radi
            self.name_isoto=name_isoto
    
    # Set the font to resemble LaTeX
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        '''
        # Creating the Figure 1 coefficient
        fig1, ax1 = plt.subplots()
        # Creating the plot
        ax1.plot(nu,coef)
        # Add a legend
        ax1.legend(loc='upper right')
        # Label the axes
        ax1.set_xlabel(r'$\nu$ $cm^{-1}$')
        ax1.set_ylabel(r'Coefficient $cm^2/molecule$')
        # Title
        ax1.set_title('Absorption coefficient for '+ name_isoto+ '$T$='+str(T)+'K $P$='+str(P)+'atm')

        # Creating the Figure 2 absorption spectrum
        fig2, ax2 = plt.subplots()
        # Creating the plot
        ax2.plot(nu,absorp)
        # Add a legend
        ax2.legend(loc='upper right')
        # Label the axes
        ax2.set_xlabel(r'$\nu$ $cm^{-1}$')
        ax2.set_ylabel(r'Absorbance')
        # Title
        ax2.set_title('Absorption spectrum for '+ name_isoto+ '$T$='+str(T)+'K $P$='+str(P)+'atm')

        # Creating the Figure 3 transmittance spectrum
        fig3, ax3 = plt.subplots()
        # Creating the plot
        ax3.plot(nu,trans)
        # Add a legend
        ax3.legend(loc='upper right')
        # Label the axes
        ax3.set_xlabel(r'$\nu$ $cm^{-1}$')
        ax3.set_ylabel(r'Transmittance')
        # Title
        ax3.set_title('Transmittance spectrum for '+ name_isoto+ '$T$='+str(T)+'K $P$='+str(P)+'atm')

        # Creating the Figure 4 radiance spectrum
        fig4, ax4 = plt.subplots()
        # Creating the plot
        ax4.plot(nu,radi)
        # Add a legend
        ax4.legend(loc='upper right')
        # Label the axes
        ax4.set_xlabel(r'$\nu$ $cm^{-1}$')
        ax4.set_ylabel(r'Radiance')
        # Title
        ax4.set_title('Radiance spectrum for '+ name_isoto+ '$T$='+str(T)+'K $P$='+str(P)+'atm')
        #Showing the plot
        plt.show()
        '''

    return Data_spectrum(nu, coef, absorp, trans, radi, name_isoto)

#spectrum(P,T,length,numin,numax,molecule_id,isotopo_id)


# Function to plot using the data that was generated

def plot_spectra(data):
    # Set the font to resemble LateX
    fig1, ax1 = plt.subplots()
    # Creating the plot
    ax1.plot(data.nu,data.coef)
    