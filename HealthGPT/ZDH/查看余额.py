import requests

url = "https://api.deepseek.com/user/balance"

payload={}
headers = {
  'Accept': 'application/json',
  'Authorization': 'Bearer sk-3d990190200d42f2b500fe6d8117aa76'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)