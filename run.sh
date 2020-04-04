#!/bin/bash
if [[ "test" = $1 ]]
then
	if [ ! -e "test.py" ]
	then
		echo "File 'test.py' not found"
	elif [ ! -e "model/classifier.pkl" ]
	then
		echo "Model file 'model/classifier' not found"
	else
		echo "Start Testing"
		python test.py
	fi
elif [[ "train" = $1 ]]
then
	if [ -e "train.py" ]
	then
		if [ -e "model/classifier.pkl" ]
		then
		  if [[ "retrain" = $2 ]]
		  then
			  mv model/classifier.pkl model/classifier.pkl.bak
			  echo "Start training..."
			  python train.py
			else
			  echo "Already Trained. If you want to retrain, type './run.sh train retrain'"
			fi
		else
		  echo "Start training..."
		  python train.py
		fi

	else
		echo "File 'train.py' not found"
	fi
else
	echo "Usage: ./run.sh [train|test]"
fi	
