import numpy as np


class Molecule:
  """
  Class 'Molecule' contains all the information about a molecule which we may need.
  """

  def __init__ (self, data):
    """
    The initializatrion of the Molecule object with its own observable properties.
    """
    try:
      self.label = data['label']
    except:
      self.label = None
    try:
      self.atomTypes = data['atomTypes']
    except:
      self.atomTypes = None
    try:
      self.coordinates = data['coordinates']
    except:
      self.coordinates = None
    try:
      self.rotationalConstants = data['rotationalConstants']     # GHz
    except:
      self.rotationalConstants = None
    try:
      self.dipole = data['dipole']                               # D
    except:
      self.dipole = None
    try:
      self.isotropicPolarization = data['isotropicPolarization'] # bohr^{3}
    except:
      self.isotropicPolarization = None
    try:
      self.homo = data['homo']                                   # Ha
    except:
      self.homo = None
    try:
      self.lumo = data['lumo']                                   # Ha
    except:
      self.lumo = None
    try:
      self.gap = data['gap']                                     # Ha
    except:
      self.gap = None
    try:
      self.spatialRadius = data['spatialRadius']                 # bohr^{2}
    except:
      self.spatialRadius = None
    try:
      self.ZPVE = data['ZPVE']                                   # Ha
    except:
      self.ZPVE = None
    try:
      self.internalEnergy0K = data['internalEnergy0K']           # Ha
    except:
      self.internalEnergy0K = None
    try:
      self.internalEnerg298K = data['internalEnerg298K']         # Ha
    except:
      self.internalEnerg298K = None
    try:
      self.enthalpy298K = data['enthalpy298K']                   # Ha
    except:
      self.enthalpy298K = None
    try:
      self.freeEnergy298K = data['freeEnergy298K']               # Ha
    except:
      self.freeEnergy298K = None
    try:
      self.heatCapacity298K = data['heatCapacity298K']           # Ha ???
    except:
      self.heatCapacity298K = None


  def getLabel(self):
    """
    Return the label.
    """

    return self.label

  def getNumberOfAtoms(self):
    """
    Return the number of atoms in the molecule.
    """

    return len(self.getAtomType())

  def getAtomType(self, index = None):
    """
    Return atom type(s).
    """
    if self.atomTypes == None:
      return self.atomTypes
    else:
      if index == None:
        return self.atomTypes
      elif index < self.getNumerOfAtoms():
        return self.atomTypes[index]
      else:
        return "No atom number %d in the molecule %d which contains only %d atoms" % (index, self.getLabel(), self.getNumerOfAtoms())

  def getCoordinate(self, atom = None, axis = None):
    """
    Return atom(s) coordinate(s).
    """

    if self.coordinates == None:
      return self.coordinates
    else:
      if atom == None:
        if axis == None:
          return self.coordinates[:,:]
        else:
          return self.coordinates[:,axis]
      else:
        if axis == None:
          return self.coordinates[atom,:]
        else:
          return self.coordinates[atom,axis]


  def getRotationalConstants(self, axis = None):
    """
    Return rotational constant(s).
    """

    if self.rotationalConstants == None:
      return self.rotationalConstants
    else:
      if axis == None:
        return self.rotationalConstants[:]
      else:
        return self.rotationalConstants[axis]


  def getDipole(self):
    """
    Return dipol.
    """

    return self.dipole


  def getIsotropicPolarization(self):
    """
    Return isotropic polarization.
    """

    return self.isotropicPolarization


  def getHOMO(self):
    """
    Return HOMO.
    """

    return self.homo

  def getLUMO(self):
    """
    Return LUMO.
    """

    return self.lumo


  def getGap(self):
    """
    Return gap.
    """

    return self.gap


  def getSpatialRadius(self):
    """
    Return spatial radius.
    """

    return self.spatialRadius


  def getZPVE(self):
    """
    Return ZPVE.
    """

    return self.ZPVE


  def getInternalEnergy0K(self):
    """
    Return internal energy at 0 Kelvin.
    """

    return self.internalEnergy0K


  def getInternalEnergy298K(self):
    """
    Return internal energy at 298 Kelvin.
    """

    return self.internalEnergy298K


  def getEnthalpy298K(self):
    """
    Return internal enthalpy at 298 Kelvin.
    """

    return self.enthalpy298K


  def getFreeEnergy298K(self):
    """
    Return internal free energy at 298 Kelvin.
    """

    return self.freeEnergy298K


  def getHeatCapacity298K(self):
    """
    Return internal heat capacity at 298 Kelvin.
    """

    return self.heatCapacity298K

def main():

  data1 = {'freeEnergy298K': -422.584313, 'enthalpy298K': -422.545318, 'internalEnerg298K': -422.546262, 'isotropicPolarization': 73.03, 'heatCapacity298K': 29.396, \
          'internalEnergy0K': -422.553381, 'ZPVE': 0.158717, 'atomTypes': ['C', 'C', 'C', 'C', 'C', 'O', 'C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], \
          'dipole': 1.0386, 'coordinates': np.array([[  1.88673383e-01,   8.14560510e-01,   2.24931418e-01], [  1.53777407e+00,   1.56510644e-01,   7.16897595e-02], \
                                                  [  2.75639735e+00,   1.00324046e+00,  -3.35485395e-01], [  3.68821260e+00,  -2.21188019e-01,  -4.40565578e-01], \
                                                  [  2.83441087e+00,  -1.09724010e+00,  -1.38924483e+00], [  1.47560262e+00,  -8.23116355e-01,  -9.99789794e-01], \
                                                  [  3.49702916e+00,  -8.78251119e-01,   9.32239547e-01], [  3.13871171e+00,  -1.84320510e-03,   2.00796156e+00], \
                                                  [  2.08899984e+00,  -6.42684510e-01,   1.26751852e+00], [  2.12986946e-01,   1.54235614e+00,   1.04147333e+00], \
                                                  [ -9.76158861e-02,   1.32811133e+00,  -6.97111859e-01], [ -5.79255051e-01,   6.84058935e-02,   4.51496586e-01], \
                                                  [  2.59302695e+00,   1.49894282e+00,  -1.29484031e+00], [  3.06579236e+00,   1.72151337e+00,   4.20807123e-01], \
                                                  [  4.71368727e+00,  -5.35568646e-02,  -7.69972105e-01], [  2.97683272e+00,  -8.19086861e-01,  -2.44091963e+00], \
                                                  [  3.04027038e+00,  -2.16964357e+00,  -1.28266093e+00], [  4.04099497e+00,  -1.77600509e+00,   1.21057194e+00], \
                                                  [  1.46214724e+00,  -1.34134890e+00,   1.81315146e+00]]), 'label': 2271, 'lumo': 0.08787, 'homo': -0.24353, \
          'spatialRadius': 927.662, 'gap': 0.33140000000000003, 'rotationalConstants': np.array([ 2.4401308,  2.1270733,  1.5384848])}

  data2 = {'freeEnergy298K': -422.584313, 'enthalpy298K': -422.545318, 'internalEnerg298K': -422.546262, 'isotropicPolarization': 73.03, 'heatCapacity298K': 29.396, \
          'internalEnergy0K': -422.553381, 'ZPVE': 0.158717, 'atomTypes': ['C', 'C', 'C', 'C', 'C', 'O', 'C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], \
          'dipole': 1.0386, 'coordinates': np.array([[  1.88673383e-01,   8.14560510e-01,   2.24931418e-01], [  1.53777407e+00,   1.56510644e-01,   7.16897595e-02], \
                                                  [  2.75639735e+00,   1.00324046e+00,  -3.35485395e-01], [  3.68821260e+00,  -2.21188019e-01,  -4.40565578e-01], \
                                                  [  2.83441087e+00,  -1.09724010e+00,  -1.38924483e+00], [  1.47560262e+00,  -8.23116355e-01,  -9.99789794e-01], \
                                                  [  3.49702916e+00,  -8.78251119e-01,   9.32239547e-01], [  3.13871171e+00,  -1.84320510e-03,   2.00796156e+00], \
                                                  [  2.08899984e+00,  -6.42684510e-01,   1.26751852e+00], [  2.12986946e-01,   1.54235614e+00,   1.04147333e+00], \
                                                  [ -9.76158861e-02,   1.32811133e+00,  -6.97111859e-01], [ -5.79255051e-01,   6.84058935e-02,   4.51496586e-01], \
                                                  [  2.59302695e+00,   1.49894282e+00,  -1.29484031e+00], [  3.06579236e+00,   1.72151337e+00,   4.20807123e-01], \
                                                  [  4.71368727e+00,  -5.35568646e-02,  -7.69972105e-01], [  2.97683272e+00,  -8.19086861e-01,  -2.44091963e+00], \
                                                  [  3.04027038e+00,  -2.16964357e+00,  -1.28266093e+00], [  4.04099497e+00,  -1.77600509e+00,   1.21057194e+00], \
                                                  [  1.46214724e+00,  -1.34134890e+00,   1.81315146e+00]]), 'lumo': 0.08787, 'homo': -0.24353, \
          'spatialRadius': 927.662, 'gap': 0.33140000000000003, 'rotationalConstants': np.array([ 2.4401308,  2.1270733,  1.5384848])}

  data3 = {'freeEnergy298K': -422.584313, 'enthalpy298K': -422.545318, 'internalEnerg298K': -422.546262, 'isotropicPolarization': 73.03, 'heatCapacity298K': 29.396, \
          'internalEnergy0K': -422.553381, 'ZPVE': 0.158717, 'atomTypes': ['C', 'C', 'C', 'C', 'C', 'O', 'C', 'O', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], \
          'dipole': 1.0386, 'lumo': 0.08787, 'homo': -0.24353, \
          'spatialRadius': 927.662, 'gap': 0.33140000000000003, 'rotationalConstants': np.array([ 2.4401308,  2.1270733,  1.5384848])}

  molecule1 = Molecule(data1)
  print 'Molecule 1: %s,  %s' % (molecule1.getLabel(), molecule1.getCoordinate(2))
  print 'Molecule 1' % (molecule1.getAtomType())



  molecule2 = Molecule(data2)
  print 'Molecule 2: %s,  %s' % (molecule2.getLabel(), molecule2.getCoordinate(2))

  molecule3 = Molecule(data3)
  print 'Molecule 3: %s,  %s' % (molecule3.getLabel(), molecule3.getCoordinate(2))



if __name__ == '__main__':
  # For debugging purposes.

  main()