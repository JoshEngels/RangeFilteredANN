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

Convert datasets to tsv format, and copy to docker container.
```bash
python3 generate_datasets/convert_to_msvbase.py 
sudo docker cp experiments/index_cache/output.tsv vbase_open_source:/
```

Run on commandline:
```bash
sudo docker exec -it --privileged --user=root vbase_open_source bash
psql -U vectordb
```
```sql
-- start writing postsql
create table t_table(id int, filter float8, vector_1 float8[3]);
copy t_table from '/output.tsv' DELIMITER E'\t' csv quote e'\x01';
select * from t_table;
```