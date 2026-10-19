import urllib.request

soap_req = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope" xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
  <soap:Body>
    <tds:GetCapabilities>
      <tds:Category>All</tds:Category>
    </tds:GetCapabilities>
  </soap:Body>
</soap:Envelope>"""

req = urllib.request.Request(
    'http://192.168.1.13:888/onvif/device_service',
    data=soap_req.encode(),
    headers={'Content-Type': 'application/soap+xml; charset=utf-8'}
)
try:
    with urllib.request.urlopen(req, timeout=3) as resp:
        content = resp.read().decode()
        print("ONVIF Response:")
        print(content[:1500])
except Exception as e:
    print("ONVIF Error:", e)
