#launch：
./spark-ec2 -k msan697 -i ../msan697.pem -s 4 -t c3.2xlarge -r us-west-2 -z us-west-2a launch sparklingwater
#login：
./spark-ec2 -k msan697 -i ../msan697.pem -r us-west-2 -z us-west-2a login sparklingwater 	
# install s3
curl "https://s3.amazonaws.com/aws-cli/awscli-bundle.zip" -o "awscli-bundle.zip"
unzip awscli-bundle.zip
./awscli-bundle/install -b ~/bin/aws
/root/bin/aws s3 cp s3://smart-meters/halfhourly/ /mnt/sparklingwater/smart_long --recursive --acl public-read
/root/bin/aws s3 cp s3://smart-meters/hhblock_dataset/ /mnt/sparklingwater/smart_wide --recursive --acl public-read 

# install mongo on ec2
sudo vi /etc/yum.repos.d/mongodb-org-3.2.repo
[mongodb-org-3.2]
name=MongoDB Repository 
baseurl=https://repo.mongodb.org/yum/amazon/2013.03/mongodb-org/3.2/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://www.mongodb.org/static/pgp/server-3.2.asc
sudo yum install -y mongodb-org
# To allow it to be accessed by outside.
# comment out bind_ip variable at the /etc/mongodb.conf file.
sudo vi /etc/mongodb.conf
sudo service mongod start

# install jupiter notebok
wget http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
bash Anaconda3–4.1.1-Linux-x86_64.sh
source .bashrc
conda install python=2.7
conda update jupyter notebook

#Pyspark with mongo						
pyspark --packages org.mongodb.spark:mongo-spark-connector_2.11:2.2.0 