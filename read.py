import os, sys
import numpy as np


def getFileNames(path='./', seach = ''):
  """
  Return paths of all files (all files which contain 'search' in names)
  from the folder 'path'.
  """

  try:
    items = os.listdir(path)
  except:
    print 'Wrong data base path: %s' % path
    sys.exit(0)

  files = []

  # This would print all the files and directories
  for file in items:
    try:
      if seach == '':
        file.index(seach)
        files.append(path + '/' + file)
      else:
        files.append(path + '/' + file)
    except:
      pass

  return files


def readFile(name):
  """
  Return all the data from the file 'name'.
  """

  file = open(name, 'r')
  data = []

  for line in file:
    if '\n' in line:
      line = line[:line.index('\n')]
    while '\t' in line:
      index = line.index('\t')
      line = line[:index] + ' ' + line[index + 1:]
    line = line.strip()
    line = line.split(' ')
    while '' in line:
      del line[line.index('')]
    data.append(line)

  file.close()

  return data


def extractRequiredData(data):
  """
  Extracting data we need for our purposes.
  """

  #Label
  label = int(data[1][1])

  #Atom types and coordinates
  atomTypes, coordinates = [], []
  for line in data:
    if len(line) == 5:
      atomTypes.append(line[0])
      coordinates.append([float(line[1]), float(line[2]), float(line[3])])
  coordinates = np.array(coordinates)

  # Other properties
  otherProperties=[]
  for j in range (2,17):
    otherProperties.append(float(data[1][j]))

  dic = {'label' : label, 'atomTypes' : atomTypes, 'coordinates' : coordinates, 'rotationalConstants' : np.array(otherProperties[0:3]), \
         'dipole' : otherProperties[3], 'isotropicPolarization' : otherProperties[4], 'homo' : otherProperties[5], \
         'lumo' : otherProperties[6], 'gap' : otherProperties[7], 'spatialRadius' : otherProperties[8], 'ZPVE' : otherProperties[9], \
         'internalEnergy0K' : otherProperties[10], 'internalEnerg298K' : otherProperties[11], 'enthalpy298K' : otherProperties[12], \
         'freeEnergy298K' : otherProperties[13], 'heatCapacity298K' : otherProperties[14]}

  return dic


def main():

  path = 'dsgdb9nsd.xyz'    #dataBase
  if len(sys.argv) > 1:
    path = sys.argv[1]

  files = getFileNames(path)

  for file in files[:1]:
    print extractRequiredData(readFile(file))

if __name__ == '__main__':
  # For debugging purposes.

  main()
