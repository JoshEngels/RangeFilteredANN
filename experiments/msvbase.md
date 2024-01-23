Install python packages:
```bash
# to connect to PostgreSQL from Python
pip3 install psycopg2-binary
```

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

# test if success
python3 tests/test_msvbase_connection.py 
```


Run 
```bash
python3 run_msvbase.py
```