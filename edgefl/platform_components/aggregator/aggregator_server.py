"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""


import argparse
from dotenv import load_dotenv

from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse


from platform_components.aggregator.aggregator import Aggregator
import asyncio
import logging
import pickle
import requests
import os
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, status

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from platform_components.EdgeLake_functions.blockchain_EL_functions import get_local_ip
import warnings

from platform_components.lib.logger.logger_config import configure_logging
from platform_components.lib.modules.exceptions import NodeInitializationError

warnings.filterwarnings("ignore")

app = FastAPI()
load_dotenv()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


configure_logging("aggregator_server")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Excludes WARNING, ERROR, CRITICAL

# Initialize the Aggregator instance
ip = get_local_ip()
port = os.getenv("SERVER_PORT", "8080")
aggregator = Aggregator(ip, port, logger)

# Track the training process of each index so that they can join once they're done
training_processes = {}


#######  FASTAPI IMPLEMENTATION  #######

class InitRequest(BaseModel):
    nodeUrls: list[str]
    index: str

class TrainingRequest(BaseModel):
    totalRounds: int
    minParams: int
    index: str

class UpdatedMinParamsRequest(BaseModel):
    updatedMinParams: int
    index: str

class ContinueTrainingRequest(BaseModel):
    additionalRounds: int
    minParams: int
    index: str

class InferenceRequest(BaseModel):
    input: list # each element in here is one data value to test
    labels: list # check element type within direct_inference


# @app.route('/init', methods=['POST'])

@app.post("/init", response_class=PlainTextResponse)
def init(request: InitRequest):
    """Deploy the smart contract with predefined nodes."""
    try:
        # Initialize the nodes on specified index and send the contract address
        node_urls, index = request.nodeUrls, request.index

        module_name = os.getenv("MODULE_NAME")
        module_file = os.getenv("MODULE_FILE")

        # module_name, module_file = request.module, request.module_file
        # db_name = request.db_name
        db_name = os.getenv("LOGICAL_DATABASE")

        # Verify filepath exists
        module_path = os.path.join(aggregator.training_app_dir, module_file)
        module_path = os.path.join(os.getenv("GITHUB_DIR"), module_path)
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Module '{module_file}' does not exist within the given path: '{module_path}'.")

        # Set up index and specific data
        aggregator.indexes.add(index)
        if index not in aggregator.databases:
            aggregator.databases[index] = db_name
        if not index in aggregator.round_number:
            aggregator.round_number[index] = 1

        initialize_nodes(node_urls, index)

        aggregator.set_module_at_index(index, module_name, module_path)
        aggregator.initialize_index_on_blockchain(index, module_name, module_path, db_name)
        aggregator.initialize_training_app_on_index(index)
        aggregator.initialize_file_write_paths_on_index(index)

        initialized_nodes = [url for url in node_urls if url in aggregator.node_urls[index]]
        failed_nodes = [url for url in node_urls if url not in aggregator.node_urls[index]]

        logger.info(f"Initialized nodes with index ({index}): {aggregator.node_urls[index]}")

        return JSONResponse(content={
            'status': 'success',
            'message': 'Initialization request finished.',
            'initialized nodes': f'{initialized_nodes}',
            'failed nodes': f'{failed_nodes}'
        })




    except FileNotFoundError as e:
        logger.error(f"{str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={str(e)}
        )
    except Exception as e:
        logger.error(f"Failed to initialize nodes with index ({index}): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def is_node_online(node_url: str):
    try:
        response = requests.get(node_url, timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False

def initialize_nodes(node_urls: list[str], index):
    """Send the deployed contract address to multiple node servers."""
    def init_node(node_url: str):
        try:
            ip_port = node_url.split('/')[-1].split(':')
            logger.info(f"Initializing model at {node_url}")

            # Check that node is online; if it's not, then remove it from node_urls and decrement
            # node_count

            if not is_node_online(node_url):
                with aggregator.lock:
                    if node_url in aggregator.node_urls[index]:
                        aggregator.node_urls[index].remove(node_url)
                        aggregator.node_count[index] -= 1
                logger.warning(f"Node {node_url} is offline; skipping initialization.")
                return

            with aggregator.lock:
                if node_url in aggregator.node_urls[index]:  # skip already initialized nodes
                    logger.info(f"Model at {url} already exists for index {index}.")
                    return

                # Reserve a replica number
                replica_number = aggregator.node_count[index] + 1
                replica_name = f"node{replica_number}"
                aggregator.node_count[index] = replica_number

            response = requests.post(f'{node_url}/init-node', json={
                'replica_ip': ip_port[0],
                'replica_port': ip_port[1],
                'replica_name': replica_name,
                'replica_index': index,
                'round_number': aggregator.round_number[index]
            })

            # init end_round



            with aggregator.lock:
                if response.status_code == 200:
                    aggregator.node_urls[index].add(node_url)
                    logger.info(f"Node at {node_url} initialized successfully.")
                else:
                    # Rollback node count if request fails
                    aggregator.node_count[index] -= 1
                    raise NodeInitializationError(
                        status_code=response.status_code,
                        detail=f"Failed to initialize node at {node_url}."
                    )
        except Exception as e:
            with aggregator.lock:
                aggregator.node_count[index] -= 1 # Rollback on exception
            logger.critical(f"{str(e)}")
            if isinstance(e, NodeInitializationError):
                raise e
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )

    if index not in aggregator.node_count:
        aggregator.node_count[index] = 0

    if index not in aggregator.node_urls:
        aggregator.node_urls[index] = set()

    # TODO: if a node gets re-init'ed because the node server re-opened, shut down the corresponding thread and start again
    threads = []
    for url in node_urls:
        thread = threading.Thread(name=f"agg/init--{url}", target=init_node, args=(url,), daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(0.1)

    for i, thread in enumerate(threads):
        thread.join(timeout=180) # Adjust timeout as necessary
        if thread.is_alive():
            logger.warning(f"Node {i} thread timed out. Failed to initialize a node.")


@app.post('/start-training')
async def init_training(request: TrainingRequest):
    """Start the training process by setting the number of rounds."""
    try:
        index = request.index
        if index not in aggregator.indexes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index {index} not found (not yet initialized)."
            )

        node_count = aggregator.node_count[index]
        num_rounds = request.totalRounds
        aggregator.minParams[index] = request.minParams

        if aggregator.minParams[index] > node_count: # prevents stalling when minParams > # of active nodes; warns user
            logger.info(
                f"[{index}] minParams ({aggregator.minParams[index]}) is greater than number of active nodes ({node_count}). Using active nodes as minParams."
            )
            aggregator.minParams[index] = node_count

        if num_rounds <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of rounds must be positive"
            )

        # TODO: if a training process is in-progress, do not allow another call to /start-training

        # TODO: add a manual way to stop training (if needed)

        starting_round = 1
        end_round = num_rounds
        initial_params = ''
        logger.info(f"[{index}] {num_rounds} {'round' if num_rounds == 1 else 'rounds'} of training started.")
        # Allow for independent training processes
        training_thread = threading.Thread(
            name=f"agg/start-training--{index}",
            target=start_training,
            args=(aggregator, initial_params, starting_round, end_round, index),
            daemon=True
        )
        training_thread.start()

        return {
            "status": "success",
            "message": f"Started training at index: {index}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def start_training(aggregator, initial_params, starting_round, end_round, index):
    try:
        aggregator.end_round[index] = end_round
        # for r in range(starting_round, end_round + 1):
        while starting_round <= aggregator.end_round[index]:
            r = starting_round
            aggregator.round_number[index] = r
            logger.info(f"[{index}] Starting training round {r}")
            aggregator.start_round(initial_params, r, index)
            logger.debug(f"[{index}] Sent initial parameters to nodes")

            # Listen for updates from nodes
            new_aggregator_params = asyncio.run(
                listen_for_update_agg(aggregator.minParams[index], r, index)
            )
            logger.debug(f"[{index}] Received aggregated parameters")

            # Set initial params to newly aggregated params for the next round
            initial_params = new_aggregator_params # docker: /app/file_write/agg/{index}/1-agg_update.json
            # print(initial_params) # debugging
            logger.info(f"[{index}][Round {r}] Step 4 Complete: model parameters aggregated")

            starting_round += 1

            # Track the last agg model file because it's not stored in a policy after the last round
            # aggregator.store_most_recent_agg_params(initial_params, index, starting_round)

            # Then, update aggregator's model at 'index'
            local_path_of_initial_params = f"{aggregator.file_write_destination}/{index}/{r}-{aggregator.agg_name}_update.json"
            with open(local_path_of_initial_params, "rb") as f:
                data = pickle.load(f)

            if data and 'newUpdates' in data:
                weights = aggregator.decode_params(data['newUpdates'])
            else:
                aggregator.logger.error(f"[{index}] Invalid data or 'newUpdates' missing in Firestore response: {data}")
                raise ValueError(f"[{index}] Invalid data or 'newUpdates' missing in Firestore response: {data}")

            aggregator.training_apps[index].update_model(weights)



        logger.info(f"[{index}] Training completed successfully")
        return {
            "status": "success",
            "message": "Training completed successfully"
        }
    except Exception as e:
        logger.error(f"[{index}] Training stopped: {e}")
        if isinstance(e, ValueError):
            raise ValueError(f"[{index}] Invalid data or 'newUpdates' missing in Firestore response: {data}")
        else:
            raise RuntimeError(f"An error occurred during training: {str(e)}")


@app.post('/update-minParams')
async def update_minParams(request: UpdatedMinParamsRequest):
    """Update minParams at an existing index. Note that indices are specified on node initialization."""
    url = f'http://{os.getenv("EXTERNAL_IP")}'
    # TODO: Rare bug, when training two different models and both are in-progress, one of them may stop when this endpoint is called...or if a node is added mid-way...not sure
    try:
        index = request.index
        if index not in aggregator.indexes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index {index} not found (not yet initialized)."
            )

        check_index_response = requests.get(url, headers={
            'User-Agent': 'AnyLog/1.23',
            "command": f"blockchain get index where name = {index}"
        })

        if check_index_response.status_code != 200:
            raise HTTPException(
                status_code=check_index_response.status_code,
                detail=check_index_response.text
            )

        index_data = check_index_response.json()
        if not index_data:
            raise HTTPException(
                status_code=404,
                detail=f"Index {index} not found in the blockchain."
            )

        node_count = aggregator.node_count[index]
        aggregator.minParams[index] = request.updatedMinParams

        if aggregator.minParams[index] > node_count: # prevents stalling when minParams > # of active nodes; warns user
            logger.info(
                f"[{index}] minParams ({aggregator.minParams[index]}) is greater than number of active nodes ({node_count}). Using active nodes as minParams."
            )
            aggregator.minParams[index] = node_count
        return {
            "status": "success",
            "message": f"minParams successfully updated to {aggregator.minParams[index]}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to set minParams at index {index}. Have the nodes and index been initialized?"
        )


async def listen_for_update_agg(min_params, round_number, index):
    """Asynchronously poll for aggregated parameters from the blockchain."""
    logger.info(f"[{index}] listening for updates...")
    url = f'http://{os.getenv("EXTERNAL_IP")}'

    # TODO: update min_params here with aggregator.min_params since the update_minParams request doesn't affect here
    #  as of now
    decoded_params = {} # { 'node_params_link': 'decoded_param' }
    check_chances = 5 # Once this reaches <= 0, we will ignore min_params and handle accordingly
    while True:
        try:
            # Fetch policies containing the node models at index and round number
            params_response = requests.get(url, headers={
                'User-Agent': 'AnyLog/1.23',
                # "command": f"blockchain get {index}-a{round_number}"
                "command": f"blockchain get {index} where round_number={round_number} and node_type=training"
            })
            params_response.raise_for_status()  # != 200

            result = params_response.json()
            if result:
                # Extract all trained_params into a list
                node_params_links = [
                    item.get(index).get('trained_params_local_path')
                    for item in result
                    if index in item
                ]
                ip_ports = [
                    item.get(index).get('ip_port')
                    for item in result
                    if index in item
                ]

                rest_ip_ports = [
                    item.get(index).get('rest_ip_port')
                    for item in result
                    if index in item
                ]

                # Updates decoded_params with newly fetched decoded params (with node link as key)
                aggregator.fetch_decoded_params(
                    decoded_params_dict=decoded_params,
                    node_param_download_links=node_params_links,
                    ip_ports=ip_ports,
                    rest_ip_ports=rest_ip_ports,
                    index=index
                )

            # If enough parameters or not getting ALL parameters in time, get the URL
            if len(decoded_params) >= min_params or (decoded_params and not check_chances):

                aggregated_params_link = aggregator.aggregate_model_params(
                    decoded_params=list(decoded_params.values()),
                    round_number=round_number,
                    index=index
                )
                return aggregated_params_link

            # TODO: Adjust this to decrement with >=0 decoded params, but based on nodes' training process
            # Only decrement the counter when there is at least 1 decoded params
            if decoded_params and check_chances:
                check_chances -= 1

            # Use most recent aggregated model link if failed to pull any node models
            if not decoded_params and not check_chances:
                aggregated_params_link = get_last_aggregated_params(index)
                if aggregated_params_link: # but only fetch if there exists one
                    return aggregated_params_link
                check_chances = 5 # If none, then reset and try to fetch node model links again

        except Exception as e:
            logger.error(f"[{index}] Aggregator_server.py --> Waiting for file: {e}")

        # TODO: see there's an alternative to this sleep e.g. sleeping for less time or using another function
        await asyncio.sleep(2)


@app.post('/continue-training')
async def continue_training(request: ContinueTrainingRequest):
    """Continue training from the last completed round."""
    try:
        index = request.index
        if index not in aggregator.indexes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index {index} not found (not yet initialized)."
            )

        node_count = aggregator.node_count[index]
        additional_rounds = request.additionalRounds
        aggregator.minParams[index] = request.minParams

        if aggregator.minParams[index] > node_count: # prevents stalling when minParams > # of active nodes; warns user
            logger.info(
                f"[{index}] minParams ({aggregator.minParams[index]}) is greater than number of active nodes ({node_count}). Using active nodes as minParams."
            )
            aggregator.minParams[index] = node_count

        if additional_rounds <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"[{index}] Invalid number of additional rounds"
            )

        # Get the last round number from the blockchain layer
        last_round = get_last_round_number(index)
        if last_round is None:
            raise HTTPException(
                status_code=400,
                detail=f"[{index}] No previous training found"
            )

        # if mid training, we don't need to do anything but update the end_round value
        if aggregator.round_number[index] < aggregator.end_round[index]:
            aggregator.end_round[index] = aggregator.end_round[index] + additional_rounds
            return {
                "status": "success",
                "message": f"Extended training at index {index} to round {aggregator.end_round[index]}: current round is {aggregator.round_number[index]}"
            }


        # Fetch the most recent aggregated model parameters
        initial_params = get_last_aggregated_params(index)
        if not initial_params:
            raise HTTPException(
                status_code=500,
                detail=f"[{index}] Failed to fetch aggregated parameters from round {last_round}"
            )

        # TODO: if a training process is in-progress, do not allow another call to /continue-training

        # TODO: add a manual way to stop training (if needed)

        starting_round = last_round + 1
        end_round = last_round + additional_rounds
        logger.info(f"[{index}] Continuing training from round {last_round}, adding {additional_rounds} more {'round' if additional_rounds == 1 else 'rounds'}.")
        # Allow for independent training processes
        training_thread = threading.Thread(
            name=f"agg/continue-training--{index}",
            target=start_training,
            args=(aggregator, initial_params, starting_round, end_round, index),
            daemon=True
        )
        training_thread.start()

        return {
            "status": "success",
            "message": f"Continuing training at index from round {starting_round} to {end_round}: {index}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


def get_last_round_number(index):
    """Get the last completed round number from the blockchain."""
    url = f'http://{os.getenv("EXTERNAL_IP")}'

    try:
        # Query the blockchain for all 'r' prefixed keys to find the highest round number
        response = requests.get(url, headers={
            'User-Agent': 'AnyLog/1.23',
            # "command": f"blockchain get * where [index] = {index} and [node_type] = aggregator"
            "command": f"blockchain get {index} where node_type = aggregator"
        })

        if response.status_code == 200:
            policies = response.json()
            if not policies or not isinstance(policies, list):
                return None

            # Extract round numbers from keys like '{index}-r1', '{index}-r2', etc.
            highest_round_number = 0
            for policy in policies:
                # if key.startswith('a') and key[1:].isdigit():
                #     round_numbers.append(int(key[1:]))
                key = next(iter(policy)) # it is dict form at first
                if key[-1] == 'r':
                    break

                _, number = key.rsplit("-r", 1)
                highest_round_number = max(highest_round_number, int(number))

            if highest_round_number == 0:
                return None

            return highest_round_number
        else:
            logger.error(f"[{index}] Error fetching keys: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"[{index}] Error fetching last round number: {str(e)}")
        return None


def get_last_aggregated_params(index):
    """Get the aggregated parameters from the specified round."""
    url = f'http://{os.getenv("EXTERNAL_IP")}'
    try:
        # Get the aggregated parameters from index-r
        response = requests.get(url, headers={
            'User-Agent': 'AnyLog/1.23',
            "command": f"blockchain get {index}-r"
        })

        if response.status_code == 200:
            result = response.json()
            if result and isinstance(result, list) and len(result) > 0:
                for item in result:
                    if f'{index}-r' in item and 'initParams' in item[f'{index}-r']:
                        return item[f'{index}-r']['initParams']

            logger.info(f"[{index}] No aggregated parameters found in policy {index}-r")
            return None

        else:
            logger.error(f"[{index}] Error fetching aggregated parameters: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"[{index}] Error fetching aggregated parameters: {str(e)}")
        return None


# TODO: make labels optional (...maybe user doesn't feel like getting the accuracy?)
@app.post("/direct-inference/{index}", response_class=PlainTextResponse)
async def direct_inference(index, request: InferenceRequest):
    try:
        results = aggregator.direct_inference(index, request.input, request.labels)
        response = (f"{{"
                    f"'index': '{index}',"
                    f" 'status': 'success',"
                    f" 'message': 'Inference completed successfully',"
                    f" 'accuracy': '{str(results)}'"
                    f"}}\n")
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


if __name__ == '__main__':
    # Add argument parsing to make the port configurable
    parser = argparse.ArgumentParser(description="Run the Aggregator Server.")
    parser.add_argument('--port', type=int, default=8080, help="Port to run the server on.")
    args = parser.parse_args()

    uvicorn.run(
        "aggregator_server:app",
        host="0.0.0.0",
        port=args.port,
        reload=False  # Enable auto-reload on code changes (optional)
    )