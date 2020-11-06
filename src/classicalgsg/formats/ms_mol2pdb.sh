#!/bin/bash


molpath="../mol_files/Star_NonStar/guowei/NonStar/mols"
pdbpath="../mol_files/Star_NonStar/guowei/NonStar/pdbs"



# LOG="train_molfiles.txt"
# for entry in "$molpath"/*
# do
#	filename="${entry##*/}"
#	basename="${filename%.*}"
#	echo $basename 1>> $LOG 2>> $LOG
# done


mkdir -p "$pdbpath"


#cat $LOG | parallel -j 10 /Applications/MarvinSuite/bin/molconvert  pdb "$molpath/{1}.mol2"  -o "$pdbpath/{1}.pdb"


for entry in "$molpath"/*
do
	filename="${entry##*/}"
	basename="${filename%.*}"
	echo "processing $basename"
	/Applications/MarvinSuite/bin/molconvert  pdb "$entry"  -o "$pdbpath/$basename.pdb"
done
