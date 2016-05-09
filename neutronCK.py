from class_molecule import Molecule
from read import *
import numpy as np
import random
import time
import sys


def main():

  # Input data for the script
  path = 'dsgdb9nsd.xyz'  #dataBase
  seach = ''
  if len(sys.argv) > 1:
    path = sys.argv[1]
  elif len(sys.argv) > 2:
    path = sys.argv[1]
    seach = sys.argv[2]


  files = getFileNames(path,seach)

  molecules = []

  for file in files:
    try:
      molecules.append(Molecule(extractRequiredData(readFile(file))))
    except:
      print 'A problem occurred while reading the file "%s"' % file

  print 'We have %s molecules in our list.' % len(molecules)


  with open("input.dat", "r") as ins:
     inputs = []
     for line in ins:
       inputs.append(line)

  num_to_select = int(inputs[0])
  lambda_krr = float(inputs[1])
  iLog2  = float(inputs[2])


  sigma_krr = 2.0**iLog2
  print 'training set size', num_to_select
  print 'lambda, iLog2, sigma', lambda_krr, iLog2, sigma_krr

  n_predict = 500

 # create an output files with the property values of interest

 # target = open ('atomisation.dat', 'w')

 # for molecule in molecules:

 #     print target.write("%s\n" %molecule.getIsotropicPolarization())
 # target.close()

  #print 'label molecule removed', molecules[0].getLabel()
  #molecules.remove(molecules[0])
  #print 'label oneMoleculeList', oneMoleculeList[0].getLabel()

  #spatialRList = []

#  for molecule1 in molecules:
#    tempV = molecule1.getSpatialRadius()
#    spatialRList.append(tempV)
    #if tempV < 19.36:
     # print 'label small molecule', molecule1.getLabel()

#  spatialRarray = np.array(spatialRList)
#  spatialRarraySquaredRoot = np.sqrt(spatialRarray)
#  np.save('radius.dat', spatialRarraySquaredRoot )
#  print 'molecules average radius', spatialRarraySquaredRoot
#  print 'len ', len(spatialRList)
#  print 'max and min radius', np.amax(spatialRarraySquaredRoot), np.amin(spatialRarraySquaredRoot)
#  Define dictionary for atomic species and relative charges

  atomic_charge = {'H' : 1,'C' : 6, 'O' : 8, 'N' : 7, 'F' : 9}
  scatter_lengh = {'H' : -3.74,'C' : 6.65, 'O' : 5.80, 'N' : 9.37, 'F' : 5.65}



  radLengh = 3.2 # in Angstrom
  print 'wave length in Angstrom', radLengh
  k_inNorm = (2*np.pi)/radLengh
  print 'k_in', k_inNorm

  n_SpherePoints = 500

  sphereCount = 0

  kList = []
  np.random.seed(7)
  while sphereCount <  n_SpherePoints:

    s1 = np.random.uniform(-1,1)
    s2 = np.random.uniform(-1,1)
   # print 's1 and s2', s1, s2

    s_tot = s1**2 + s2**2
    k_vect = []
    if s_tot < 1.0:

      sphereCount = sphereCount + 1
     # print 'I, s_tot', sphereCount, s_tot
      s_par = (1.0-s_tot)**0.5
      kx = 2.0*s1*s_par
      k_vect.append(kx)
      ky = 2.0*s2*s_par
      k_vect.append(ky)
      kz = 1.0 - 2.0* s_tot
      #print kx,ky,kz
      #print np.sqrt(kx**2+ky**2+kz**2)
      k_vect.append(kz)
      k_vect = np.array(k_vect)
      k_vect = k_inNorm * k_vect
      kList.append(k_vect)

  #print 'k List', kList
  kListArray = np.array(kList)
  np.save('kVectorList.dat', kList)
  #print 'k list length', len(kList)




  #print 'norms', np.linalg.norm(kList[0]), np.linalg.norm(kList[9]), np.linalg.norm(kList[5])
  im = 1j
  #print 'exp immaginary', np.exp(im*(np.pi/4))

  k_z = np.array([0.0,0.0,1.0])
  k_z = k_inNorm * k_z

  qList= []

  for waveV in kListArray:
    tempWaveV = waveV - k_z
    #print  tempWaveV[0], tempWaveV[1], tempWaveV[2]
    qList.append(tempWaveV)

  #print 'q list', qList
  dLengh = len(qList)
  print 'q len list', dLengh


  print 'number of descriptor components', dLengh

#  interAtomList = []

#  for molecule2 in molecules:
#    sizeM = molecule2.getNumberOfAtoms()
#    for i in range(0,sizeM):
#      for j in range(0,sizeM):
#        if i != j:

#         vquant = molecule2.getCoordinate()[i]- molecule2.getCoordinate()[j]
#         magVquant = np.linalg.norm(vquant)
#         interAtomList.append(magVquant)

#  interAtomArray = np.array(interAtomList)
#  np.save('interatomic.dat', interAtomArray)
#  print 'interatomic distance mim and max', np.amin(interAtomArray), np.amax(interAtomArray)
#  print 'average interatomic distance', np.mean(interAtomArray)



  def diff(a,b):



    b = set(b)
    return [aa for aa in a if aa not in b]

  def remove_duplicate(l):
    return list(set(l))

  print 'num to select', num_to_select
  lg = 0


  trainingMolecules = []

  random.seed(5)
  print 'Achtung seed Chosen aqui', random.random()

  while lg < num_to_select:
    frameran = random.sample(molecules,1)
    trainingMolecules.append(frameran[0])
    trainingMolecules = remove_duplicate(trainingMolecules)
    lg = len(trainingMolecules)



#  print len(trainingMolecules)
#  print len(set(trainingMolecules))
  if len(trainingMolecules) != len(set(trainingMolecules)):
    print 'Warning: hay duplicates in the training set!!'

  print 'trainingSet size', lg

  trainingMolecules.sort(key=lambda x: x.label, reverse=True)

  if lg != num_to_select:
    print 'WARNING: training list length different from input'
    print 'training set size, input size : %s,  %s' % (lg, num_to_select)


  print 'We have %s molecules in our list.' % len(trainingMolecules)

  #trainingLabelList = []
  trainingPropertyList = []
  dimensionD = len(kList)

  dlist = []

  for molecule1 in trainingMolecules:
    labelTemp = molecule1.getLabel()
    #print 'label', labelTemp
    #trainingPropertyList.append(molecule1.getEnthalpy298K())
    #fact = molecule1.getSpatialRadius()
    #fact1 = (0.08/fact)
    #fact2 = 1 + fact1
    #fact2sq = - fact1**2
    #factBis = np.exp(fact2sq)
    #trainingPropertyList.append(molecule1.getEnthalpy298K())
    trainingPropertyList.append(molecule1.getIsotropicPolarization())
    #trainingLabelList.append(labelTemp)
    sizeMolecule = molecule1.getNumberOfAtoms()
    #sqSizeMolecule = np.sqrt(sizeMolecule)
    #oneOverSize = 1.0/sqSizeMolecule
    sumTerm = 0.0

    #print 'molecule label', labelTemp
    for qTemp in qList:
      #print 'qtemp ****', qTemp
      for i in range(0,sizeMolecule):

          rMol = molecule1.getCoordinate()[i]
          #print 'i, r_i', i, rMol
          b_i = scatter_lengh[molecule1.getAtomType()[i]]
          #b_i = atomic_charge[molecule1.getAtomType()[i]]
          #bTerm = (bTerm + b_i)
          #print 'i, b_i', i, molecule1.getAtomType()[i], b_i
          arg = np.dot(qTemp,rMol)
          #print 'scalar product', arg
          expTemp = b_i* np.exp(im*arg)
          #print 'bi*exp', i, expTemp
          #print 'earlier sumTerm', sumTerm
          sumTerm = sumTerm + expTemp
          #print 'partial sum', sumTerm
          #sumTerm = sumTerm*oneOverSize
          #sumTerm = sumTerm*bTerm
          #print 'partial', sumTerm
      dlist.append(sumTerm)



  #print 'dlist length', len(dlist)
  dlistArray = np.array(dlist)
  #print 'dlist array', dlistArray
  dlistArray = np.absolute(dlistArray)**2
  #print 'dlist array squared', dlistArray
  #print 'max value', np.amax(dlistArray)
  descriptorList = [dlistArray[i:i+dimensionD] for i in range(0,len(dlistArray),dimensionD)]
  #print 'desciptor List size', len(descriptorList), type(descriptorList)
  #print descriptorList

  blist = []
  for moleculeTr in trainingMolecules:
    labelTempTr = moleculeTr.getLabel()
    #print 'LABEL', labelTempTr
    sizeMoleculeTr = moleculeTr.getNumberOfAtoms()
    oneOverSize = 1.0/sizeMoleculeTr
    #print oneOverSize
    bTerm = 0.0
    for i in range(0,sizeMoleculeTr):
         # print 'bterm', bTerm
          b_i = scatter_lengh[moleculeTr.getAtomType()[i]]
          #print 'b_i', b_i,  i, moleculeTr.getAtomType()[i]
          #b_i = atomic_charge[molecule1.getAtomType()[i]]
          bTerm = bTerm + b_i
          #print 'bterm after', bTerm

    bTerm = bTerm*oneOverSize
    blist.append(bTerm)

  #print 'blist', blist
  #print 'size', len(blist)
  blistArray = np.array(blist)
  blistArray = blistArray**2
  #print 'blist squared', blistArray

  descriptorList = [a*b for a,b in zip(blistArray,descriptorList)]
  #print descriptorList


  print 'molecular training property values', trainingPropertyList

  moleculesPropertyArray = np.array(trainingPropertyList)
  np.save('trainingPropertyArray', moleculesPropertyArray)
  #print 'allMoleculesPropertyArray size, dimension', allMoleculesPropertyArray.shape, allMoleculesPropertyArray.ndim
  #print 'abs dlist', dlistArray
  #maxDlist = np.amax(dlistArray)
  #invMaxDlist = 1.0/maxDlist
  #dlistArray = invMaxDlist*dlistArray
  #print 'min and max', np.amin(dlistArray), np.amax(dlistArray)
  #dlistArray = 1e+9*dlistArray


  descriptorNList = []
  for item in descriptorList:
    #print 'item', item
    itemMax = np.amax(item)
    #print 'max', itemMax
    itemN = item/itemMax
   # print 'normalized item', itemN
    descriptorNList.append(itemN)


  #print 'type', descriptorNList

  zetaTotList = []


  four_pi = 4.0* np.pi
  fourPiSq = four_pi**2
  overPiSq = 1.0/fourPiSq
  #print fourPiSq
  for molecule2 in trainingMolecules:
    #print 'label', molecule2.getLabel()
    size1Molecule = molecule2.getNumberOfAtoms()
    zetaPart = 0
    for ind in range(0,size1Molecule):
      zetaTemp = atomic_charge[molecule2.getAtomType()[ind]]
     # print 'i,charge', ind, zetaTemp
      zetaPart = zetaPart + zetaTemp
    zetaPart = (zetaPart**2)*overPiSq
    zetaTotList.append(zetaPart)

  #print 'list Charges length', len(zetaTotList)
  #print 'zeta total list', zetaTotList

 # zetaTotArray = np.array(zetaTotList)
 # zetaTotArray = zetaTotArray**2
 # print zetaTotArray

  neutronD = [a*b for a, b in zip(zetaTotList,descriptorNList)]
  #print 'Training Neutron descriptor D', neutronD





  def gaussian(x,y, sigmaGauss):


    twoSigmaInv = -1.0/float((2.0*sigmaGauss**2.0))

    return np.exp(twoSigmaInv*np.linalg.norm(x-y)**2.0)
    #return np.exp(twoSigmaInv*(np.linalg.norm(x-y)/(1.0 + np.linalg.norm(x-y))) **2.0)

  def laplacian(x,y, sigmaLaplacian):


    sigmaInv = -1.0/float(sigmaLaplacian)

    return np.exp(sigmaInv*np.linalg.norm((x-y),ord = 1))




  outOfSampleMolecules = diff(molecules, trainingMolecules)

  print 'length of outOfSample', len(outOfSampleMolecules)
  testMolecules = []
  lg1 = 0

  random.seed(7)

  while lg1 <  n_predict:
    frameran1 = random.sample(outOfSampleMolecules,1)
    testMolecules.append(frameran1[0])
    testMolecules = remove_duplicate(testMolecules)
    lg1 = len(testMolecules)

  if len(testMolecules) != len(set(testMolecules)):
    print 'Warning: hay duplicates in the  outOfsample set!!'


  varBool = set(testMolecules).isdisjoint(trainingMolecules)
  if varBool == True:
    print 'list intersection is empty'
  else:
    print '!!!hay (training) lists overlap aqui !!!'

  testMolecules.sort(key=lambda x: x.label, reverse=True)

  n = num_to_select
  K = np.zeros((n,n),dtype=np.float64)


  for l in range(0,n):
    for m in range(0,n):
      # include sigma
       K[l,m] =  laplacian(neutronD[l],neutronD[m], sigma_krr)

  print 'Kernel matrix', K
  print 'K dimensions', K.shape



  Id = np.eye(n)

  Ud = np.zeros((n,n),dtype=np.float64)
  entryConst = 1.0
  Ud.fill(entryConst)

  print 'Ud matrix', Ud
  #print 'size Ud matrix', len(Ud)
  #print 'Ud reduced', Ud_reduced
  oneOvern =1.0/n
  oneOvern2 = oneOvern**2

  Ktilde = np.dot(K, Ud)
  Ktilde1 = np.dot(Ud,Ktilde)
  Ktilde1 = oneOvern2*Ktilde1

  Ktilde2 = np.dot(Ud,K)
  Ktilde2 = -oneOvern*Ktilde2

  Ktilde3 = -oneOvern*Ktilde

  K_tilde = K + Ktilde1 + Ktilde2 + Ktilde3

  print 'K_tilde', K_tilde

  #sys.exit('ciao')

  K1 = K_tilde + lambda_krr*Id

  y_mean =  np.mean(moleculesPropertyArray)

  moleculesPropertyArray_minus = moleculesPropertyArray - y_mean




  #start = time.time()

  c = np.linalg.solve(K1,moleculesPropertyArray_minus)

 # print '\n Inversion time: %s sec' % (round(time.time() - start, 1))

  print 'c', type(c)
  print 'regression coefficients', c
  np.save('regressionCoef.dat',c)
  print 'length regression array', len(c)



  #referenceLabelList = []
  referencePropertyList = []


  dlist2 = []

  for molecule2 in testMolecules:
    labelTemp2 = molecule2.getLabel()
    #print 'LABEL', labelTemp2
    #referencePropertyList.append(molecule2.getEnthalpy298K())
    referencePropertyList.append(molecule2.getIsotropicPolarization())
    #fact = molecule2.getSpatialRadius()
    #fact1 = (0.08/fact)
    #fact2 = 1 + fact1
    #fact2sq = - fact1**2
    #factBis = np.exp(fact2sq)
    #referenceLabelList.append(labelTemp2)
    sizeMolecule2 = molecule2.getNumberOfAtoms()
    #print 'size molecule', sizeMolecule2
    #print 'isotropic polariz', molecule2.getIsotropicPolarization()


    sumTerm2 = 0.0

    for qTemp2 in qList:
      #print 'qtemp ****', qTemp
      for i in range(0,sizeMolecule2):

          rMol2 = molecule2.getCoordinate()[i]
         # print 'i, r_i', i, rMol2
          b_i2 = scatter_lengh[molecule2.getAtomType()[i]]
         # b_i2 = atomic_charge[molecule2.getAtomType()[i]]


          arg2 = np.dot(qTemp2,rMol2)
          #print 'scalar product', arg2
          expTemp2 = b_i2* np.exp(im*arg2)
          #print 'bi*exp', i, expTemp
          #print 'earlier sumTerm', sumTerm
          sumTerm2 = sumTerm2 + expTemp2
          #print 'partial sum', sumTerm
          #sumTerm2 = sumTerm*oneOverSize2

          #print sumTerm
      dlist2.append(sumTerm2)



  dlistArray2 = np.array(dlist2)
  dlistArray2 = np.absolute(dlistArray2)**2
  #print 'dlistArray2', dlistArray2
  #print 'max dlistArray2', np.amax(dlistArray2)
  descriptorList2 = [dlistArray2[i:i+dimensionD] for i in range(0,len(dlistArray2),dimensionD)]
  #print 'descriptor2 List length', len(descriptorList2)
  #print 'descriptor2', descriptorList2


  referencePropertyArray = np.array(referencePropertyList)
  print 'reference property list length', referencePropertyArray
  np.save('ReferenceProperty.dat', referencePropertyArray)

  maxRefProperty = np.amax(referencePropertyArray)
  minRefProperty = np.amin(referencePropertyArray)
  refPropRange = maxRefProperty - minRefProperty

  blist2 = []
  for moleculeTest in testMolecules:
    labelTempTest = moleculeTest.getLabel()
    #print 'LABEL2', labelTempTest
    sizeMoleculeTest = moleculeTest.getNumberOfAtoms()
    oneOverSizeTest = 1.0/sizeMoleculeTest
    bTermTest = 0.0
    for i in range(0,sizeMoleculeTest):
          #print 'bterm', bTerm
          b_ii = scatter_lengh[moleculeTest.getAtomType()[i]]
          #b_i = atomic_charge[molecule1.getAtomType()[i]]
          bTermTest = bTermTest + b_ii
          #print 'i, b_i, bTerm', i, moleculeTr.getAtomType()[i], b_i, bTerm
    bTermTest = bTermTest*oneOverSizeTest
    blist2.append(bTermTest)

  #print 'blist2', blist2
  #print 'size', len(blist)
  blist2Array = np.array(blist2)
  blist2Array = blist2Array**2
  #print 'blist2 squared', blist2Array

  descriptorList2 = [a*b for a,b in zip(blist2Array,descriptorList2)]
  #print len(descriptorList2), descriptorList2




  descriptorNList2 = []

  for item2 in descriptorList2:
    #print 'item2', item
    itemMax2 = np.amax(item2)
    #print 'max', itemMax2
    itemN2 = item2/itemMax2
    #print 'normalized item', itemN
    descriptorNList2.append(itemN2)


  #print 'test normalized descriptor', descriptorNList2

  zetaTotList2 = []



  for molecule3 in testMolecules:
    #print 'LABEL 3', molecule3.getLabel()
    sizeMolecule3 = molecule3.getNumberOfAtoms()
    zetaPart3 = 0
    for indd in range(0,sizeMolecule3):
      zetaTemp3 = atomic_charge[molecule3.getAtomType()[indd]]
      zetaPart3 = zetaPart3 + zetaTemp3
    zetaPart3 = (zetaPart3**2)*overPiSq
    zetaTotList2.append(zetaPart3)

  #print len(zetaTotList2)
  #print zetaTotList2

 # zetaTotArray = np.array(zetaTotList)
 # zetaTotArray = zetaTotArray**2
 # print zetaTotArray

  neutronD1 = [a1*b1 for a1, b1 in zip(zetaTotList2,descriptorNList2)]

  #print 'test neutron descriptor D1', neutronD1

  CK = np.zeros((n_predict,n),dtype=np.float64)

  for l in range(0,n_predict):
    for m in range(0,n):
      # include sigma
       CK[l,m] =  laplacian(neutronD1[l],neutronD[m], sigma_krr)



  yPredictionArray = np.dot(CK,c)
  print 'property mean', y_mean
  yPredictionArray = yPredictionArray + y_mean

  recipr = 1.0/float(n_predict)
  print 'CK matrix dimension', CK.shape
  print 'CK', CK
  print 'Property Predictions', yPredictionArray
  print 'Prediction array size', len(yPredictionArray)
  np.save('predictionProperty.dat',yPredictionArray)





  pc1 = np.dot(referencePropertyArray,yPredictionArray)
  pc1 = n_predict* pc1
  pc_sum1 = np.sum(referencePropertyArray)
  pc_sum2 = np.sum(yPredictionArray)
  pc_prod = pc_sum1*pc_sum2
  pc_num = pc1 - pc_prod
  pc_squ1 = np.dot(referencePropertyArray,referencePropertyArray)
  pc_squ1 = n_predict*pc_squ1
  pc_squ2 = np.dot(yPredictionArray,yPredictionArray)
  pc_squ2 = n_predict*pc_squ2
  pc1term = pc_squ1- pc_sum1**2
  pc1term = np.sqrt(pc1term)
  pc2term = pc_squ2 -pc_sum2**2
  pc2term = np.sqrt(pc2term)
  pc_den = pc1term*pc2term
  Pr = pc_num/pc_den
  PrSq = Pr**2
  PrDiff = 1.0 - PrSq




  invRefPropRange = 1.0/refPropRange
  fun = referencePropertyArray- yPredictionArray
  funAbs = np.absolute(fun)
  funSq = fun**2
  RMSE = np.sqrt(recipr*np.sum(funSq))
  MAE = recipr*np.sum(funAbs)
  PercentError = recipr*invRefPropRange*MAE*100
  print 'average percent error', PercentError

  print 'MAE, RMSE', MAE, RMSE
  print 'Pearson squared', PrSq

  f = open('errorstat.out','w')
  print >> f, 'MAE', MAE
  print >> f, 'RMSE', RMSE
  print >> f, 'sigma', sigma_krr
  print >> f, 'lambda', lambda_krr
  print >> f, 'training size N', num_to_select
  print >> f, 'R square Pearson', PrSq, PrDiff
  print >> f, ' index i (log2)', iLog2
  print >> f, ' wavelength in Ang', radLengh
  print >> f, ' no of components', dLengh


  sys.exit('cool')












  print '\nThe gap values of the first ten atoms.'
  for molecule in molecules[:4]:
    print molecule.getGap()

  molecules = sorted(molecules, key=lambda listElement: listElement.getGap())   # sort by gap





  print '\nThe gap values of the first ten atoms after sorting by gap.'
  for molecule in molecules[:5]:
    print molecule.getGap()

  items = []

  for item in molecules:
    items.append(item.gap)

  print len(items)
  print max(items)
  print min(items)

 # Different way for making a list of object attributes (Comprehension)
  listTest = [x.gap for x in molecules[:5]]
  print listTest
  print random.choice(listTest)



  num_to_select = 3
  list_of_randomValues = random.sample(items,num_to_select)


  print list_of_randomValues









if __name__ == '__main__':

  start = time.time()

  main()

  print '\n Total time: %s sec' % (round(time.time() - start, 1))

  print '\n  Chao!'


