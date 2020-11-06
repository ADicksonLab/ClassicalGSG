#!/bin/bash


pdbpath="../mol_files/Star_NonStar/NonStar"
search_dir="$pdbpath/smiles"

mkdir -p "$pdbpath/mol2"
mkdir -p "$pdbpath/pdbs"

for entry in "$search_dir"/*
do
	filename="${entry##*/}"
	basename="${filename%.*}"
	echo "processing $basename"
	/Applications/MarvinSuite/bin/molconvert  -3 mol2:H "$entry"  -o "$pdbpath/mol2/$basename.mol2"
	/Applications/MarvinSuite/bin/molconvert  -3 pdb:H "$entry"  -o "$pdbpath/pdbs/$basename.pdb"
done
