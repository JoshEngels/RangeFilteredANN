```bash
git clone https://github.com/microsoft/MSVBASE.git
cd MSVBASE
git submodule update --init --recursive
./scripts/patch.sh

# start docker
sudo systemctl start docker
# build. I have to copy paste it to terminal for it to work.
# ./scripts/dockerbuild.sh
sudo docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t vbase_open_source -f Dockerfile .

# start MSVBASE docker container
sudo ./scripts/dockerrun.sh
```


Run on commandline:
```bash
sudo docker exec -it --privileged --user=root vbase_open_source bash
psql -U vectordb
# start writing postsql
create table t_table(id int, price int, vector_1 float8[10], vector_2 float8[10]);
```