"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/
"""


import os
import subprocess
import time
import docker
# Add files to MongoDB through EdgeLake

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import sys
import tarfile

def write_file(edgelake_node_url, dbms, table, filename):
    lst = filename.split('/')
    # lst[-1] = f'{dbms}.{table}.{lst[-1]}'
    filename = ('/').join(lst)

    headers = {
        'User-Agent': 'AnyLog/1.23',
        'Content-Type': 'application/octet-stream',  # Specify binary content
        'command': f'file store where dbms = {dbms} and table = {table} and dest = {filename.split("/")[-1]}'
        # 'command': f'file store where dbms = {dbms} and table = {table} and dest = {filename}'
    }

    with open(filename, 'rb') as f:
        binary_data = f.read()
        response = requests.post(edgelake_node_url, headers=headers, data=binary_data, verify=False)
    return response


def create_directory_in_container(edgelake_url, container_name, directory_path):
    """
    Create a directory inside the EdgeLake container.

    AnyLog's `system mkdir` replies with an error body that is not valid
    chunked HTTP, so requests raises ChunkedEncodingError (AnyLog err_code 61).
    Create the directory with docker exec against the named container instead.
    """
    directory_path = directory_path.replace("\\", "/")
    if container_name and _mkdir_with_docker(container_name, directory_path):
        return

    _mkdir_with_anylog(edgelake_url, directory_path)


def _mkdir_with_docker(container_name, directory_path):
    try:
        completed = subprocess.run(
            ["docker", "exec", container_name, "mkdir", "-p", directory_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Could not run docker exec on {container_name}: {error}")
        return False

    if completed.returncode == 0:
        return True

    detail = (completed.stderr or completed.stdout or "").strip()
    print(
        f"docker exec {container_name} mkdir -p {directory_path} failed: {detail}. "
        "EDGELAKE_DOCKER_CONTAINER_NAME must match `docker ps`."
    )
    return False


def _mkdir_with_anylog(edgelake_url, directory_path):
    headers = {
        "command": f"system mkdir -p {directory_path}",
        "User-Agent": "AnyLog/1.23",
        "Content-Type": "text/plain",
    }
    try:
        response = requests.post(edgelake_url, data="", headers=headers, timeout=30)
    except requests.exceptions.RequestException as error:
        print(f"AnyLog could not create {directory_path}: {error}")
        return

    if response.status_code != 200 or "err_code" in response.text:
        print(f"AnyLog could not create {directory_path}: {response.status_code} {response.text}")


def copy_file_to_container(tmp_dir, container_name, edgelake_url, src_path, dest_path):
    """
    Copies a file from the host to a container.

    :param container_name: Name or ID of the container
    :param src_path: Absolute path of the source file on the host
    :param dest_path: Destination path in the container (directory or full path)
    """

    with open(src_path, "rb") as f:
        m = MultipartEncoder(
            fields={
                # force the filename to match exactly curl behavior
                "file": (os.path.basename(src_path), f, "text/plain")
            },
            # boundary="------------------------LyLnkEG8W8JYQL8HJouctc"  # mimic curl’s boundary
        )

        headers = {
            "command": f"file to {dest_path}", # TODO: Make sure dest_path includes filename
            "Content-Type": m.content_type,  # includes the boundary
            "User-Agent": "curl/8.7.1",  # some picky servers care!
            "Accept": "*/*",
        }

        try:
            resp = requests.post(edgelake_url, data=m, headers=headers)
        except:
            errno, value = sys.exc_info()[:2]
            print(f'Error: {errno}: {value}')


    # ######### This is using docker api given the container name #############
    # client = docker.from_env()
    # container = client.containers.get(container_name)
    #
    # # Create a tar archive for the file
    # with tarfile.open(f"{tmp_dir}/temp.tar", mode="w") as tar:
    #     tar.add(src_path, arcname=os.path.basename(src_path))
    #
    # # Open the tar file and send it to the container
    # with open(f"{tmp_dir}/temp.tar", "rb") as tar_file:
    #     container.put_archive(os.path.dirname(dest_path), tar_file)
    #
    # # Clean up the temporary tar file
    # os.remove(f"{tmp_dir}/temp.tar")
    # ###########################################################################


def copy_file_from_container(tmp_dir, container_name, edgelake_data_host_url, src_path, dest_path, ip_port_file_loc):
    """
    Copies a file from a container to the host machine.

    :param container_name: Name or ID of the container
    :param src_path: Path of the source file inside the container
    :param dest_path: Destination path on the host (directory or full path)
    """

    headers = {
        "User-Agent": "AnyLog/1.23",
        "command": f"file from {src_path}",
    }

    # Send the request
    try:
        resp = requests.post(edgelake_data_host_url, headers=headers, stream=True)
        raw = resp.content
        cleaned = raw.decode("utf-8").encode("latin1")
        # Save response content to a local file
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(cleaned)
                # for chunk in resp.iter_content(chunk_size=8192):
                #     if chunk:
                #         f.write(chunk)

        else:
            print(f"Status: {resp.status_code}")
            print("Response:", resp.text)
        return resp
    except:
        errno, value = sys.exc_info()[:2]
        print(f'Error: {errno}: {value}')

    # ####### Using Docker API #######
    # client = docker.from_env()
    # container = client.containers.get(container_name)
    # try:
    #     # Step 1: Get file size inside the container for verification
    #     exec_result = container.exec_run(f"stat -c %s {src_path}")
    #     if exec_result.exit_code != 0:
    #         print(f"❌ Error: Could not get file size for {src_path} inside the container.")
    #         return False
    #     container_file_size = int(exec_result.output.decode().strip())
    #
    #     # Step 2: Get the file from the container as a tar stream
    #     tar_stream, _ = container.get_archive(src_path)
    #
    #     # Step 3: Extract the tar stream to the host
    #     with open(f"{tmp_dir}/temp.tar", "wb") as temp_tar:
    #         for chunk in tar_stream:
    #             temp_tar.write(chunk)
    #         temp_tar.flush()
    #         os.fsync(temp_tar.fileno()) # Ensure data is physically written to disk
    #
    #     # Step 4: Extract tar file
    #     with tarfile.open(f"{tmp_dir}/temp.tar", mode="r") as tar:
    #         # Extract the file to the desired host location
    #         tar.extractall(path=os.path.dirname(dest_path), filter='fully_trusted')
    #
    #     # Step 5: Move extracted file to final destination
    #     extracted_file = os.path.join(os.path.dirname(dest_path), os.path.basename(src_path))
    #     os.rename(extracted_file, dest_path) # rename file
    #
    #     # Step 6: Verify file integrity
    #     for _ in range(5):
    #         if os.path.exists(dest_path) and os.path.getsize(dest_path) == container_file_size:
    #             print(f"✅ Successfully copied {src_path} from {container_name} to {dest_path}")
    #             os.remove(f"{tmp_dir}/temp.tar")  # Cleanup temporary tar file
    #             return True
    #         time.sleep(0.5)
    #     print(
    #         f"❌ Verification failed: Expected {container_file_size} bytes, Got {os.path.getsize(dest_path) if os.path.exists(dest_path) else 'Missing'}")
    # except Exception as e:
    #     print(f"❌ Error: {e}")
    # finally:
    #     # Clean up the temporary tar file
    #     if os.path.exists(f"{tmp_dir}/temp.tar"):
    #         os.remove(f"{tmp_dir}/temp.tar")
    #
    # ####### End Using Docker API ###########


def read_file(edgelake_node_url, file_path, dest, ip_port):
    """Pull a file from an EdgeLake node over REST and save it at `dest`.

    `file get` only works behind `run client` and returns err_code 2 when posted
    directly. `file from` returns the file bytes to the REST caller.
    """
    command = f"file from {file_path}"
    headers = {
        "User-Agent": "AnyLog/1.23",
        "Content-Type": "text/plain",
        "command": command,
    }
    try:
        response = requests.post(edgelake_node_url, headers=headers, data=b"", timeout=60)
    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"AnyLog could not return {file_path} from {edgelake_node_url}: {error}"
        ) from None
    if response.status_code != 200 or not response.content:
        detail = response.text[:300] if response.text else "empty body"
        raise RuntimeError(
            f"AnyLog could not return {file_path} from {edgelake_node_url}: "
            f"HTTP {response.status_code} {detail}"
        )
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(dest, "wb") as handle:
        handle.write(response.content)
    return response



def read_file_mongo(edgelake_node_url, dbms, table, filename, dest, ip_port):

    headers = {
        'User-Agent': 'AnyLog/1.23',
        'Content-Type': 'text/plain',
        # 'command': f'file get (dbms={dbms} and table={table} and id={filename.split("/")[-1]}) {dest}',
        'command': f'run client {ip_port} file get (dbms={dbms} and table={table} and id={filename.split("/")[-1]}) {dest}'
        # 'destination': ip_port
    }

    print(f"FILE GET COMMAND: headers: {headers['command']}")
    try:
        response = requests.post(edgelake_node_url, headers=headers, data='')
        return response
    except:
        errno, value = sys.exc_info()[:2]
        print(f'Error: {errno}: {value}')


if __name__ == '__main__':

    url = "http://192.168.1.125:32249"
    dest_path = "/Users/roy/test2/node3-model.txt"

    headers = {
        "User-Agent": "AnyLog/1.23",
        "command": "file from /app/file_write/node2/fl7/1-replica-node2.pkl"
    }

    resp = requests.post(url, headers=headers, stream=True)

    if resp.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ File saved to {dest_path}")
    else:
        print(f"❌ Error: {resp.status_code}")
        print("Response:", resp.text)

    # file_write_destination = os.getenv("FILE_WRITE_DESTINATION")
    # response = write_file(f"http://192.168.1.125:32049", "blobs_mydb", "admin",
    #            f"/Users/roy/test2.txt")
    # print(response.status_code)
    # the last ip_port variable is the node in which you want to get the file from. edgelake_node_url is the node that acts as your gateway to the anylog network
    # after the below command, test2.txt should be in /Users/roy/new_dir
    # response = read_file_mongo(f"http://192.168.1.125:32049", "blobs_mydb", "admin", "test2.txt", f"/Users/roy/my_file2.txt", "192.168.1.125:32049")
