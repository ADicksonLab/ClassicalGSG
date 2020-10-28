#!/bin/bash --login

module load schrodinger/2019-2

database_path="../database/Guo_Wei/training"

mkdir "$database_path/pdbs"

search_dir="$database_path/mol2"
for entry in "$search_dir"/*
do
	filename="${entry##*/}"
	basename="${filename%.*}"
	structconvert -imol "$entry" -opdb "$database_path/pdbs/$basename.pdb"
	echo "$basname done"
done
