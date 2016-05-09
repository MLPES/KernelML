#!/bin/bash


for f in `cat list`
do
SEQ1=`echo $f | awk -F- '{ print $1 }'`
SEQ2=`echo $f | awk -F- '{ print $2 }'`


cat>ML.$f.sh<<TOFILE
#/bin/bash
### run in /bin/bash
#$ -S /bin/bash
### join stdout and stderr
#$ -j y
### change to current working dir
#$ -cwd
### send no mail
#$ -m n 
### my email address
#$ -M marchetti@fhi-berlin.mpg.de
### Parallel Environment
### Job name
#$ -N machine_learn$f
#$ -pe impi_hydra 32
### wallclock, e.g. 3600 seconds
#$ -l h_rt=86400
### virtual memory (45G is max. on AIMS)
#$ -l h_vmem=22G


module load python27/python/2.7 python27/scipy/2015.10


for f in \` seq $SEQ1 0.5 $SEQ2 \`
do
mkdir \$f
cp neutron.py  read.py class_molecule.py \$f/
pushd \$f/

cat >input.dat<<TOEND
5000
1e-12
\$f
TOEND

cat >THREAD.sh<<TOEDF
#!/bin/bash

stdbuf -oL python /u/gima/neutronML/\$f/neutron.py /u/gima/dsgdb9nsd.xyz > out.\$f.log
exit
TOEDF
chmod +x THREAD.sh

./THREAD.sh &


popd
done


wait
TOFILE

qsub ML.$f.sh 

sleep 1
done







