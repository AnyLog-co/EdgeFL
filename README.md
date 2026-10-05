# EdgeFL

## Overview

The following is instructions to simulate the continuous Federated Learning (FL) lifecycle 
consisting of three training nodes and one aggregator node. Each node will utilize its
own EdgeLake node, such that we will deploy four EdgeLake nodes, three of which have 
operator roles and one with the master role. The master role is a normal EdgeLake operator
node but also emulates the same blockchain-like functionality of the blockchain-back shared
metadata layer. For more information about EdgeLake and how it operates, check the [EdgeLake website](https://edgelake.github.io/).

There are three demos supported, where each operator will locally train a ML model
utilizing its local data and EdgeFL will dynamically facilitate model sharing and aggregation
via the aggregator node. The value here is that there is no data movement. 
Since the demo instructions
are for a single machine, although they can be easily adapted to execute on multiple
physical machine, we will deploy multiple Postgres databases, one for each EdgeLake operator,
to emulate physically distributed data. Nevertheless, each node will utilize its own EdgeLake 
operator node (running in a Docker container) to truly simulate a distributed environment.

Before you get started, please follow the configuration steps precisely.

# Configuration
Assumptions:
 - Downloaded / cloned the repository.
 - Have Docker installed.

Install all necessary Python packages. Tested on Python3.12. Make sure to add all required Python packages to 
your `requirement.txt` file. In addition, you can also utilize a Python virtual environment, see the [venv docs](https://docs.python.org/3/library/venv.html).  
```bash
cd EdgeFL
pip install -r requirements.txt
```

## Deploy Postgres container
You will need one Postgres container for each EdgeLake operator. 
Postgres will become available on your inet IP address. You can determine this through the `ifconfig` command. 
```bash
* Start Docker *
cd EdgeLake/postgres
make up NAME=postgres1 HOST_PORT=5432 VOLUME=pgdata1
make up NAME=postgres2 HOST_PORT=5433 VOLUME=pgdata2
make up NAME=postgres3 HOST_PORT=5434 VOLUME=pgdata3
```
Note that you can change the default POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB in the Makefile
or add them as arguments to the `make up` command. 
To kill these services:
```bash
make clean NAME=postgres1
make clean NAME=postgres2
make clean NAME=postgres3
```

## Deploy EdgeLake Master node
```bash
cd EdgeLake/
make up EDGELAKE_TYPE=master TAG=1.3.2501 EDGELAKE_SERVER_PORT=32048 EDGELAKE_REST_PORT=32049 NODE_NAME=master
```
Now we need to determine the Master node's Docker IP address. Issue the following command
```bash
cd EdgeLake/
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' master
```

With this IP, we can now deploy our three EdgeLake operator nodes. For example, let's assume it's `192.1.1.1`.

## Deploy EdgeLake Operator node
Update line 61 (LEDGER_CONN) value in the file `EdgeLake/docker_makefile/edgelake_operator1.env`
to be `LEDGER_CONN=192.1.1.1:32048` (note that you do not need to change the port).
In addition, update the `DB_IP` in line 31 with the Docker network IP of the Postgres container. 

Do the same for the following files:
- `EdgeLake/docker_makefile/edgelake_operator2.env`
- `EdgeLake/docker_makefile/edgelake_operator3.env`

Now we can start the operator nodes.
```bash
cd EdgeLake/
make up EDGELAKE_TYPE=operator TAG=1.3.2501 EDGELAKE_SERVER_PORT=32148 EDGELAKE_REST_PORT=32149 NODE_NAME=operator1
make up EDGELAKE_TYPE=operator TAG=1.3.2501 EDGELAKE_SERVER_PORT=32248 EDGELAKE_REST_PORT=32249 NODE_NAME=operator2
make up EDGELAKE_TYPE=operator TAG=1.3.2501 EDGELAKE_SERVER_PORT=32348 EDGELAKE_REST_PORT=32349 NODE_NAME=operator3
```

## Validating your EdgeLake network is properly setup
To  validate your EdgeLake network is properly setup, execute the following commands:
```bash
docker attach master
```
Hit the [enter/return] key on your keyboard. 
You should now see `EL master +>`. 
Now type into the CLI `test network` and press [enter/return]. 
You should see the  following print out:
```bash
EL master +> test network

Test Network
[****************************************************************]

EL master +>
Address          Node Type Node Name Status
----------------|---------|---------|------|
172.19.0.2:32048|master   |master   |  +   |
172.19.0.3:32148|operator |operator1|  +   |
172.19.0.4:32248|operator |operator2|  +   |
172.19.0.5:32348|operator |operator3|  +   |
```
The `+` signifies that the nodes are all members of EdgeLake's p2p network. If you do not see that, then 
please contact the EdgeLake maintainers through [EdgeLake's Slack Channel](https://lfedge.org/projects/edgelake/)
(the join link is at the bottom of the page).

More over, make sure each operator is connected to the DBMS (the demos utilize the DBMS `mnist_fl`).
```bash
docker attach operator1
EL operator1 +> get databases

Active DBMS Connections
Logical DBMS Database Type Owner  IP:Port            Configuration                                   Storage
------------|-------------|------|------------------|-----------------------------------------------|----------|
almgm       |psql         |system|192.168.1.125:5433|Autocommit On, Failed to pull Fsync            |Persistent|
mnist_fl    |psql         |user  |192.168.1.125:5433|Autocommit Off, Failed to pull Fsync           |Persistent|
system_query|psql         |system|192.168.1.125:5433|Autocommit Off, Unflagged, Failed to pull Fsync|Persistent|
```
If you don't see the `mnist_fl` line, then you will need to execute the following EdgeLake CLI command
and re-execute the above command:
```bash
connect dbms mnist_fl where type = psql and user = [user] and password = [password] and ip = [ip] and port = [port] and memory = true 
```

# Demos
The demo instructions can be found the the `EdgeFL/Demo-READMEs/` directory.

We currently have two demos:
- [MNIST handwriting dataset demo](Demo-READMEs/MNIST.md)
- [Winniio demo on temperature prediction](Demo-READMEs/WINNIIO.md), a telemetry dataset. Thank you to your partners [Winniio homepage](https://www.winniio.io)!
- [Xray detection bounding box](Demo-READMEs/Chest-Xray-BoundingBox.md)
- [Shakespeare next-character prediction](Demo-READMEs/Shakespeare.md)


## Resolving common issues
After executing the init `curl` request, if your training nodes do not print out model weights,
then they do not have access to the data. 
The first step is to double check that you loaded data into your Postgres instance.
Make sure that you have the following:
1. `mnist_fl` database
2. Make sure there's actually data in those tables.

Another step to double check is that the EdgeLake operator nodes are connected to the database.  
To do so you can docker attach to each container and check.
For example, to check `operator1` is connected to the PSQL database, issue the following commands:
```bash
docker attach operator1
```
press [enter/return] so you see `EL master +>`.
Now to validate your PSQL connection, do the following:
```bash
get databases
```
You should see the following:
```bash
EL operator1 +> get databases

Active DBMS Connections
Logical DBMS Database Type Owner  IP:Port            Configuration                       Storage
------------|-------------|------|------------------|-----------------------------------|----------|
almgm       |psql         |system|192.168.1.125:5432|Autocommit On, Fsync on            |Persistent|
mnist_fl    |psql         |user  |192.168.1.125:5432|Autocommit Off, Fsync on           |Persistent|
system_query|psql         |system|192.168.1.125:5432|Autocommit Off, Unflagged, Fsync on|Persistent|
```

If you don't see `mnist_fl`, then execute the following EdgeLake specific command:
```bash
connect dbms mnist_fl where type = psql and user = demo and password = passwd and ip = 192.1.1.1 and port = 5432 and memory = true 
```
where the `192.1.1.1` is your inet ip from above. It should output `database connected`. 
If not, then your IP may be wrong. 

You can also view all your database tables for a certain DBMS
```bash
EL operator1 +> get tables where dbms = mnist_fl

Database Table name                                           Local DBMS Blockchain
--------|----------------------------------------------------|----------|----------|
mnist_fl|mnist_test                                          | V        | V        |
        |mnist_train                                         | V        | V        |
        |par_mnist_train_2025_07_01_d14_insert_timestamp     | V        | -        |
        |par_room_12004_test_2025_07_01_d14_insert_timestamp | V        | -        |
        |par_room_12004_train_2025_07_01_d14_insert_timestamp| V        | -        |
        |room_12004_test                                     | V        | V        |
        |room_12004_train                                    | V        | V        |
        |room_12055_test                                     | -        | V        |
        |room_12055_train                                    | -        | V        |
        |room_12090_test                                     | -        | V        |
        |room_12090_train                                    | -        | V        |
```

View the tables hosted by each EdgeLake operator:
```bash
EL operator1 +> get data nodes

Company     DBMS     Table            Cluster ID                       Cluster Status Node Name Member ID External IP/Port    Local IP/Port    Main Node Status
-----------|--------|----------------|--------------------------------|--------------|---------|---------|-------------------|----------------|----|-----------|
New Company|mnist_fl|room_12055_train|1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
New Company|mnist_fl|room_12055_test |1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
New Company|mnist_fl|room_12004_train|1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
New Company|mnist_fl|room_12004_test |1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
New Company|mnist_fl|room_12090_train|1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
New Company|mnist_fl|room_12090_test |1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
New Company|mnist_fl|mnist_train     |1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
New Company|mnist_fl|mnist_test      |1a4a2c6f59161cf5f7b242abcacf6ba2|active        |operator1|       79|104.60.100.77:32148|172.17.0.3:32148| +  |active     |
           |        |                |                                |active        |operator2|      208|104.60.100.77:32248|172.17.0.4:32248| +  |active     |
           |        |                |                                |active        |operator3|       65|104.60.100.77:32348|172.17.0.5:32348| +  |active     |
```

View row count by operator:
```bash
EL operator1 +> get rows count where dbms=mnist_fl

DBMS Name Table Name                                           Rows Count
---------|----------------------------------------------------|----------|
mnist_fl |mnist_test                                          |         0|
         |mnist_train                                         |         0|
         |par_mnist_train_2025_07_01_d14_insert_timestamp     |        50|
         |par_room_12004_test_2025_07_01_d14_insert_timestamp |      1576|
         |par_room_12004_train_2025_07_01_d14_insert_timestamp|      6300|
         |room_12004_test                                     |         0|
         |room_12004_train                                    |         0|

EL operator1 +> run client (192.168.1.125:32148) sql mnist_fl select count(*) from mnist_train
[39]
EL operator1 +>
{"Query":[{"count(*)":50}],
"Statistics":[{"Count": 1,
                "Time":"00:00:00",
                "Nodes": 1}]}
```

Note, to detach from EdgeLake, press ctrl+p+q simultaneously. 

## Redoing simulation / Clean up
To redo the simulation, you need to delete the `edgefl/file_write` directory.
In addition, you need to kill and restart the EdgeLake operators and master node.
To do so, follow the following instructions:
```bash
cd EdgeLake/
make clean EDGELAKE_TYPE=master TAG=1.3.2501 EDGELAKE_SERVER_PORT=32048 EDGELAKE_REST_PORT=32049 NODE_NAME=master
make clean EDGELAKE_TYPE=operator TAG=1.3.2501 EDGELAKE_SERVER_PORT=32148 EDGELAKE_REST_PORT=32149 NODE_NAME=operator1
make clean EDGELAKE_TYPE=operator TAG=1.3.2501 EDGELAKE_SERVER_PORT=32248 EDGELAKE_REST_PORT=32249 NODE_NAME=operator2
make clean EDGELAKE_TYPE=operator TAG=1.3.2501 EDGELAKE_SERVER_PORT=32348 EDGELAKE_REST_PORT=32349 NODE_NAME=operator3
```
Note that you do not need to restart Postgres.
After this step, if you want to restart the simulation follow the Deploy EdgeLake Operator/Master Node from above.

To stop Postgres:
```bash
cd EdgeLake/postgres
docker compose down
```

## Docker Containerization of APIs

The APIs are containerized using Docker. Before starting the APIs, ensure that 
```bash
edgefl/env_files/mnist-docker/mnist1.env
edgefl/env_files/mnist-docker/mnist2.env
edgefl/env_files/mnist-docker/mnist3.env
```
are configured like this:
```bash
GITHUB_DIR=/app/edgefl

TRAINING_APPLICATION_DIR=platform_components/data_handlers
MODULE_NAME=MnistDataHandler

PORT=<operator port num> #(aggregator port num + operator num = operator port num)
SERVER_TYPE=node
TMP_DIR=tmp_dir/node<operator number>/
# External IP Address for CURL commands to Edgelake
EXTERNAL_IP="<EdgeLake ip:port>"
EXTERNAL_TCP_IP_PORT="<EdgeLake ip:port>"

# node that system_query resides on
QUERY_NODE_URL="<EdgeLake ip:port>"
# Edge Node containing data
EDGE_NODE_URL="<EdgeLake ip:port>"
# Logical database name
LOGICAL_DATABASE=mnist_fl
# Table containing trained data
TRAIN_TABLE=room_12055_train
# Table containing test data
TEST_TABLE=room_12055_test

FILE_WRITE_DESTINATION="file_write"

EDGELAKE_DOCKER_RUNNING="True"
EDGELAKE_DOCKER_CONTAINER_NAME="operator<operator number>"
DOCKER_FILE_WRITE_DESTINATION="/app/file_write"
```
The aggregator env file,
```bash
edgefl/env_files/mnist-docker/mnist-agg.env
```

Should be configured like this:
```bash
GITHUB_DIR=/app/edgefl/

TRAINING_APPLICATION_DIR=platform_components/data_handlers
MODULE_NAME=MnistDataHandler

PORT=8080
SERVER_TYPE=aggregator

TMP_DIR=tmp_dir/agg/
# External IP Address for CURL commands to Edgelake
EXTERNAL_IP="<host ip>:32049"
EXTERNAL_TCP_IP_PORT="<host ip>:32048"

# LOCAL PSQL DB NAME
PSQL_DB_NAME="mnist_fl"
PSQL_DB_USER="demo"
PSQL_DB_PASSWORD="passwd"
PSQL_HOST=<host ip>
PSQL_PORT="5432"

FILE_WRITE_DESTINATION="file_write"
AGG_NAME=agg

EDGELAKE_DOCKER_RUNNING="True"
EDGELAKE_DOCKER_CONTAINER_NAME=master
DOCKER_FILE_WRITE_DESTINATION="/app/file_write"
```
To build the image, run the following command from the root directory of the project:

```bash
docker build -t edgefl:latest -f api-containers/Dockerfile .
```

You can run any of the APIs using Docker Compose. The `docker-compose.yml` file in the `api-containers` directory defines the services for the aggregator and nodes.

To run all of the APIs:
```bash
cd api-containers
docker compose up -d
```

The run only specific API services in the `docker-compose.yml` file, you can add the `--no-deps` flag to avoid starting dependent services. This is useful for testing or development purposes. 
The template for running a set of services is as follows:
```bash
cd api-containers
docker compose up -d --no-deps <service1> <service2> ...
```
Where `<service1>`, `<service2>`, etc. are the names of services defined in the `docker-compose.yml` file.


Example A: to run only the Aggregator
```bash
cd api-containers
docker compose up --no-deps -d aggregator
```
Example B: to run the aggregator and two nodes
```bash
cd api-containers
docker compose up --no-deps -d aggregator node1 node2
```
You can then add a node by running the following command:
```bash
docker compose up --no-deps -d node3
```

To see the endpoints and interact with the APIs, you can use the following URLs:
```bash
127.0.0.1:8080/docs # aggregator

127.0.0.1:8081/docs # nodes
127.0.0.1:8082/docs
127.0.0.1:8083/docs
```

To take down the containers, simply run:
```bash
docker compose down
```














# ============ Please Ignore below, README being refactored ============== 

<!-- To Do

- Need to add different commands for Mac & Windows
  -->

## Why EdgeLake:

- **Efficiency:** Reduces data transfer by sharing model parameters instead of sharing entire datasets.

- **Performance:** Our distributed system enables high-computation training to be split across multiple nodes, significantly enhancing speed and scalability compared to single-node processing. .

- **Privacy:** Keeps data on nodes, minimizing exposure risks and enhancing security with privacy-preserving technologies such as blockchain.

# AnyLog-Edgelake Setup Guide

This guide will walk you through setting up and running the AnyLog-Edgelake system.

<!-- ## Prerequisites -->
<!-- we need specify more stuff here -->

## Software Requirements

- Python 3.12
- Git
- Ability to run shell scripts
- cURL (for API requests)
- EdgeLake (to run the aggregator and nodes)
- _Optional_: PyCharm \[Professional Edition\]

## Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/royshadmon/Anylog-Edgelake-CSE115D.git
   ```

2. Configure Environment Variables:

    Note that there are currently two datasets in `edgefl/data`: _mnist_ and _winniio_.
    For this setup, we will use the winniio dataset and configure for it.

    - Navigate to `Anylog-Edgelake-CSE115D/edgefl/env_files`

   ```bash
   cd Anylog-Edgelake-CSE115D/edgefl/env_files
   ```

    - Locate `winniio.env` in the directory
    - Modify these variables in the file:

        - `EXTERNAL_IP`: Replace the IP portion with your machine's IP address.
                You can fetch them in the terminal using the command `ifconfig` on
                Mac/Linux, found under `inet`/`en0`(`ipconfig` on Windows under `IPv4`).
        - `EXTERNAL_TCP_IP_PORT`: Do the same process done for `EXTERNAL_IP`.
        - `PSQL_DB_USER`: Set this to the user for the database that will be used.
        - `PSQL_DB_PASSWORD`: If you have a password for the user, set this to it.
        - `FILE_WRITE_DESTINATION`: Set this path to `[/path/to]/Anylog-Edgelake-CSE115D/edgegl/file_write`

    - Notes:
    
        - If your database will not be locally hosted, set `PSQL_DB_HOST` accordingly.
        - If you have specified the port of your database, set `PSQL_DB_PORT` accordingly.

3. Install pip:

    - Navigate back to `Anylog-Edgelake-CSE115D/` and run the following:

   ```
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python3.12 get-pip.py
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Set up and start a Postgres instance

    - [Postgres Mac instructions](https://www.sqlshack.com/setting-up-a-postgresql-database-on-mac/)

6. Set up and load data into Postgres

    - In `Anylog-Edgelake-CSE115D/`, navigate to `edgefl/data/winniio-rooms/linode-setup/`
    - Locate `winniio_db_script.py` and ensure that it is configured to get ENV variables from
        `winniio.env`.
        - On PyCharm, you can edit the configuration of the script and specify the
            .env file in _Path(s) to ".env" files_
    - Run the script to load the winniio dataset:
      ```bash
      ./winniio_db_script
      ```

8. Start the Servers:

    - The following script has not yet been updated (eventually):

   ```bash
   ./start_servers.sh
   ```
   
    - Instead, start each node server and the one aggregator server manually.
    
        - The aggregator server file and node server file are located in
            `edgefl/platform_components/aggregator/` and
            `edgefl/platform_components/node/`, respectively.
        - Run `/aggregator/aggregator_server.py` to start the aggregator server.
        - Run `/aggregator/node_server.py` to start a node server.

    - **Notes**:

        - For each server file ran, ensure that you've load the ENV variables
            from the env files.
        - For each node server, make sure you've specified their ports (8081,
            8082, etc.).
            - If you're using PyCharm, you can edit the configuration
                of `node_server.py` and add this to _script parameters_:
                `--p [next available port]`. You can also create a new
                configuration for each node server you want to start for convenience.

## System Initialization

After starting the servers, you need to initialize the nodes. Use the following curl command:

To train on 1 node
```bash
curl -X POST http://localhost:8080/init \          
-H "Content-Type: application/json" \
-d '{
  "nodeUrls": [    
    "http://localhost:8081"
  ],
  "model_def": 1
}'
```
To train on 2 nodes
```bash
curl -X POST http://localhost:8080/init \
-H "Content-Type: application/json" \
-d '{
  "nodeUrls": [    
    "http://localhost:8081", 
    "http://localhost:8082"
  ],
  "model_def": 1
}'
```


## Parameters Explained

### Initialization Parameters
- `nodeUrls`: Array of URLs for the participating nodes

### Training Parameters
- `totalRounds`: Number of training rounds to perform
- `minParams`: Minimum number of parameters required (default: 1)

## Troubleshooting

If you encounter any issues:
1. Ensure all servers are running properly
2. Check that the `.env` file is configured correctly
3. Verify all ports (8080, 8081, 8082) are available
4. Make sure all the software requirements are properly installed \
~~5. If servers aren't responding, try killing them with `./kill_servers.sh` and restart~~

## MongoDB (for file handling) [Mongo is currently deprecated, we will need to re-add this feature]

### Install / setup (Mac)

```bash
brew tap mongodb/brew  
brew install mongodb-community # requires Xcode 16.0+
brew install mongosh # (Optional) allows access through mongo-cli
sudo mkdir -p /usr/local/bin/mongodb/var/mongodb
sudo mkdir -p /usr/local/bin/mongodb/log/mongodb
sudo chown $USER /usr/local/bin/mongodb/
sudo chown $USER /usr/local/bin/mongodb/log/mongodb
```

[Mongo information](https://www.prisma.io/dataguide/mongodb/connecting-to-mongodb)
[Mongo Install](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/)
[Mongo Shell](https://www.mongodb.com/docs/mongodb-shell/)

### Install / setup (for Ubuntu machines)

```bash
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -  
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
sudo apt-get update
sudo apt-get install -y mongodb-mongosh
mongosh --version
```

[Mongo Ubuntu Setup](https://www.slingacademy.com/article/how-to-install-mongodb-shell-mongosh-on-windows-mac-and-ubuntu/)

### Start MongoDB Mac
```bash
brew services start mongodb-community
mongod --dbpath /usr/local/bin/mongodb/var/mongodb --logpath /usr/local/bin/mongodb/log/mongodb/mongo.log
```
