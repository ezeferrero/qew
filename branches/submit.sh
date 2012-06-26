#!/bin/sh

# request Bourne shell as shell for job

#$ -S /bin/bash

# assume current working directory as paths

#$ -cwd

#./main-EW.A1.296s10k > [EW,8192,8192,10000,131072,8192,0.f,1.f,1.f,1.296f,0.025f,GPU55].dat 
./main-EW-1.01fc > [EW,4096,8192,10000,32768,1024,0.f,1.f,1.f,1.01f,0.1f,GPU55].dat 


