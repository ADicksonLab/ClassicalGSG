#!/bin/bash

dataset_path="../mol_files/Star_NonStar/guowei/NonStar"
search_dir="$dataset_path/pdbs"

mkdir "$dataset_path/gaffmol2"


count=0
for entry in "$search_dir"/*
do
	filename="${entry##*/}"
	basename="${filename%.*}"
	echo "processing $basename"
	antechamber -i "$entry" -fi pdb -o "$dataset_path/gaffmol2/$basename.mol2" -fo mol2 -c bcc -s 2

	# (( count ++ ))

	# if (( count = 20 )); then
	#	wait
	#	count=0
	# fi
done



# dataset='test/Star'
# pdbpath="../molfiles/Guo_Wei/$dataset/pdbs"
# molpath="../molfiles/Guo_Wei/$dataset/gaffmol2"
# logpath="../molfiles/Guo_Wei/$dataset/gafflogs"

# rm -r $molpath
# mkdir -p $molpath


# LOG="molfiles.txt"

# rm $LOG
# for entry in "$pdbpath"/*
# do
#		filename="${entry##*/}"
#		basename="${filename%.*}"
#		echo $basename 1>> $LOG 2>> $LOG
# done

# # cat $LOG | parallel --results $logpath  -j 20 antechamber -i "$pdbpath/{1}.pdb" -fi pdb -o "$molpath/{1}.mol2" -fo mol2 -c bcc -s 2; echo done {1}


# cat $LOG |parallel --citation -j 20 antechamber -i "$pdbpath/{1}.pdb" -fi pdb -o "$molpath/{1}.mol2" -fo mol2 -c bcc -s 2


# count=0
# for entry in "$pdbpath"/*
# do
#         filename="${entry##*/}"
#         basename="${filename%.*}"
#         echo "processing $basename"  1>> $LOG 2>> $LOG
#         antechamber -i "$entry" -fi pdb -o "$molpath/$basename.mol2" -fo mol2 -c bcc -s 2
#         (( count ++ ))

#         if (( count = 20 )); then
#                 wait
#                 count=0
#         fi
# done
