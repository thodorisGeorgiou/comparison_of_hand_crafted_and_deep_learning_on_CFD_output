#Old script, is here for reference

#!/usr/bin/env bash
NF=$(wc -l < "sliceIndeces.txt")
for (( i=1 ; $i < $NF + 2; i=$((i+1)) )) do
	cp sampleDict system/
	IND=`sed "${i}q;d" sliceIndeces.txt`
	echo $IND
	COM1="python createSamplingDict.py system/sampleDict "$IND
	COM2="postProcess -func sampleDict"
	$COM1
	$COM2
done

0.8
1.4
1.5

30.349357178
118.439245524
120.609225691

START=$(date +%s.%N)
postProcess -funcs '(sampleDictLow)'
END=$(date +%s.%N)
DIFF3=$(echo "$END - $START" | bc)
