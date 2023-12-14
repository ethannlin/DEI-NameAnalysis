import requests

while True:

    name = input('Enter a name: ')
    resp = requests.post('http://127.0.0.1:5000/predict/{}'.format(name))
    print(resp.text)