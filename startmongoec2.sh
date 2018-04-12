
echo "[10gen]
name=10gen Repository
baseurl=http://downloads-distro.mongodb.org/repo/redhat/os/x86_64
gpgcheck=0" >> /etc/yum.repos.d/10gen.repo

sudo yum install mongo-10gen and mongo-10gen-server

sudo mkdir -p /data/db/

sudo mongod