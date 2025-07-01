### get computa resources:

import requests

## Get available / general

def sessionImport():

    cookies = {
        '_mlx': 'session%3Aadambeedell%3A7bc2c975-e888-441a-91ff-37e9762343a3',
    }

    headers = {
        'accept': 'application/json',
        'accept-language': 'en-GB,en;q=0.9,ja-JP;q=0.8,ja;q=0.7,en-US;q=0.6',
        'content-type': 'application/json',
        'origin': 'https://computa.mlx.institute',
        'priority': 'u=1, i',
        'referer': 'https://computa.mlx.institute/',
        'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"', # test deleting me
        'sec-ch-ua-mobile': '?0', # test deleting me
        'sec-ch-ua-platform': '"macOS"', # test deleting me
        'sec-fetch-dest': 'empty', # test deleting me
        'sec-fetch-mode': 'cors', # test deleting me
        'sec-fetch-site': 'same-site', # test deleting me
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36', # test deleting me
        # 'cookie': '_mlx=session%3Aadambeedell%3A7bc2c975-e888-441a-91ff-37e9762343a3',
    }

    session = {
        'cookies': cookies,
        'headers': headers
    }

    return session


def getMachinesAvailable(session):

    json_data = {
        'source': '\n  query ComputaGetServers {\n    computa_get_servers\n  }\n',
        'variableValues': {},
    }

    response = requests.post('https://api.mlx.institute/g', cookies=session.cookies, headers=session.headers, json=json_data)

    return response

## get a 3090


def get3090(session):

    json_data = {
        'source': '\n  mutation ComputaCreateGpu(\n    $cloudType: String,\n    $templateId: String,\n    $containerDiskInGb: Int,\n    $gpuTypeId: String,\n  ) {\n    computa_create_gpu(\n      cloudType: $cloudType,\n      templateId: $templateId,\n      containerDiskInGb: $containerDiskInGb,\n      gpuTypeId: $gpuTypeId\n    )\n  }\n',
        'variableValues': {
            'cloudType': 'COMMUNITY',
            'templateId': 'runpod-torch-v21',
            'containerDiskInGb': 160,
            'gpuTypeId': 'NVIDIA GeForce RTX 3090',
        },
    }

    response = requests.post('https://api.mlx.institute/g', cookies=session.cookies, headers=session.headers, json=json_data)

    return response


  ## get a CPU (cx32 in hellsinki community)

def getCPU(session):

    json_data = {
        'source': '\n  mutation ComputaCreateCpu(\n    $server_type: String,\n    $location: String\n  ) {\n    computa_create_cpu(\n      server_type: $server_type,\n      location: $location\n    )\n  }\n',
        'variableValues': {
            'server_type': 'cx32',
            'location': 'hel1',
        },
    }

    response = requests.post('https://api.mlx.institute/g', cookies=session.cookies, headers=session.headers, json=json_data)

    return response

    


## delete by ID (CPU)

def deleteInstance(session, id):

    json_data = {
        'source': '\n  mutation ComputaTerminate($id: ID!, $type: String!) {\n    computa_terminate(id: $id, type: $type)\n  }\n',
        'variableValues': {
            'type': 'cpu',
            'id': id,
        },
    }

    response = requests.post('https://api.mlx.institute/g', cookies=session.cookies, headers=session.headers, json=json_data)

    return response