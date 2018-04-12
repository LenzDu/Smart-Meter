
source ~/.bash_profile

mkdir add add2 halfhourly hhblock other

sudo pip install pandas

aws s3 cp s3://smart-meters/halfhourly/ halfhourly  --recursive --acl public-read
aws s3 cp s3://smart-meters/hhblock_dataset/ hhblock --recursive --acl public-read
aws s3 cp s3://smart-meters/other/ other --recursive --acl public-read

echo "import sys
import pandas as pd
import glob
import sys

data = []
globbed_files = glob.glob("*.csv")


import os

for csv in globbed_files:
    frame = pd.read_csv(csv, low_memory = False)
    frame['filename'] = os.path.basename(csv)
    name = "../%s/new" % (sys.argv[1])
    frame.to_csv(name + csv)
    print csv" >> merge.py

cp merge.py halfhourly/
cp merge.py hhblock/

cd halfhourly 

python merge.py add

cd ../add

for i in *.csv; do mongoimport -d smart -c energy --type csv --file $i --headerline ; done

cd ../hhblock

python merge.py add2

cd ../add2

for i in *.csv; do mongoimport -d wide -c energy --type csv --file $i --headerline ; done

cd ../other

for i in *.csv; do mongoimport -d other -c $i --type csv --file $i --headerline ; done



