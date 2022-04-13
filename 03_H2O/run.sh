currentpath=`pwd`
rootpath=`cd .. && pwd`

source /home/export/online1/mdt00/shisuan/swhfnl/julia/setenv

cd /home/export/online1/mdt00/shisuan/swhfnl/julia/usr/lib/julia_hybrid
./link_to_6a_lapack.sh
./link_to_6a_openblas.sh

cd $rootpath
bsub -akernel -n 1 -q q_swhfnl -exclu -J pkg_install -I julia install.jl

cd $currentpath

system=`ls param*`

for i_core  in 6
do
rm -rf test$i_core
mkdir test$i_core
cd test$i_core


        cp ../../SWQCQC.jl ./
        cp ../$system ./

		bsub -akernel -b -n $i_core -q q_swhfnl -cgsp 64 -cache_size 0 -exclu -o out julia SWQCQC.jl $system 

cd ..
done
