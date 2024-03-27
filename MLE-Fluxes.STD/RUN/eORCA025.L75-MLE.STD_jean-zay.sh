#!/bin/bash
#SBATCH --ntasks=1009
#SBATCH -J nemo_jean-zay
#SBATCH -e nemo_jean-zay.e%j
#SBATCH -o nemo_jean-zay.o%j
#SBATCH -A cli@cpu
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --exclusive

# Python environment -- complete/modify/comment as you wish
# ---------------------------------------------------------
source $P_PYOASIS_DIR/python/init.sh
source $P_PYOASIS_DIR/python/init.csh
source ${HOME}/.bash_profile
conda activate morays
# ----------- 

set -x
ulimit -s 
ulimit -s unlimited

CONFIG=eORCA025.L75
CASE=MLE.STD

CONFCASE=${CONFIG}-${CASE}
CTL_DIR=$PDIR/RUN_${CONFIG}/${CONFCASE}/CTL

##################################################################
#  WARNING: On Jean-Zay, It seems that having NEMO and XIOS_SERVER
#        running on the same node, lead to freezing in iom_init
#        A workaround is to put all the xios on a separate node 
#        using NB_NCORE_DP /= 0 
#################################################################
# Following numbers must be consistant with the header of this job
export NB_NPROC=999    # number of cores used for NEMO
export NB_NPROC_IOS=0  # number of cores used for xios (number of xios_server.exe)
export NB_NCORE_DP=0   # activate depopulated core computation for XIOS. If not 0, RUN_DP is
                       # the number of cores used by XIOS on each exclusive node.

# OASIS coupling - set to 0 if not
export NB_NPROC_PYCPL=10   # number of cores used for coupled python script

# Rebuild process 
export MERGE=0         # 1 = on the fly rebuild, 0 = dedicated job
export NB_NPROC_MER=20 # number of cores used for rebuild on the fly  (1/node is a good choice)
export NB_NNODE_MER=20  # number of nodes used for rebuild in dedicated job (MERGE=0). One instance of rebuild per node will be used.
export WALL_CLK_MER=2:00:00   # wall clock time for batch rebuild
export ACCOUNT=cli@cpu # account to be used

date
#
echo " Read corresponding include file on the HOMEWORK "
.  ${CTL_DIR}/includefile.sh

. $RUNTOOLS/lib/function_4_all.sh
. $RUNTOOLS/lib/function_4.sh
#  you can eventually include function redefinitions here (for testing purpose, for instance).
. $RUNTOOLS/lib/nemo4.sh
