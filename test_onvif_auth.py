import urllib.request
import base64
import hashlib
import time

created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
nonce_bytes = b"1234567890123456"
nonce_b64 = base64.b64encode(nonce_bytes).decode()
# PasswordDigest = Base64( SHA-1( nonce + created + password ) )
pwd = b""
digest_raw = hashlib.sha1(nonce_bytes + created.encode() + pwd).digest()
digest_b64 = base64.b64encode(digest_raw).decode()

soap_req = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope"
               xmlns:tds="http://www.onvif.org/ver10/device/wsdl"
               xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
               xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">
  <soap:Header>
    <wsse:Security>
      <wsse:UsernameToken>
        <wsse:Username>admin</wsse:Username>
        <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest_b64}</wsse:Password>
        <wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce_b64}</wsse:Nonce>
        <wsu:Created>{created}</wsu:Created>
      </wsse:UsernameToken>
    </wsse:Security>
  </soap:Header>
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
        print(content[:2000])
except Exception as e:
    print("ONVIF Error:", e)
