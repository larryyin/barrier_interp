#!/bin/sh

export NODES=1
export PPN=10
export DIR_WORK=`pwd`

qsub -l nodes=${NODES}:ppn=${PPN} -N log -d ${DIR_WORK} -v PATH,LD_LIBRARY_PATH ${DIR_WORK}/job_execute2
