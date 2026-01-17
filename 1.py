import http.client
import json

conn = http.client.HTTPSConnection("ark.cn-beijing.volces.com")
payload = json.dumps({
   "model": "doubao-seed-1-6-lite-251015",
   "messages": [
      {
         "role": "system",
         "content": "You are a helpful assistant."
      },
      {
         "role": "user",
         "content": "端口映射是什么意思？"
      }
   ]
})
headers = {
   'Authorization': 'Bearer dc2bd008-7c45-4744-ba12-ec6754d8c1a1',
   'Content-Type': 'application/json'
}
conn.request("POST", "/api/v3/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))