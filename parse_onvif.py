import urllib.request
import base64
import hashlib
import time
import os
import xml.etree.ElementTree as ET

def get_onvif_capabilities():
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nonce_bytes = os.urandom(16)
    nonce_b64 = base64.b64encode(nonce_bytes).decode()
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
    with urllib.request.urlopen(req, timeout=3) as resp:
        return resp.read().decode()

xml_str = get_onvif_capabilities()
# Print all services and XAddrs
import re
print("=== ONVIF CAPABILITIES SUMMARY ===")
for match in re.finditer(r'<tt:([a-zA-Z]+) xsi:type="[^"]+">.*?<tt:XAddr>([^<]+)</tt:XAddr>', xml_str, re.DOTALL):
    print(f"Service: {match.group(1)} -> {match.group(2)}")

# Check audio in media
for line in xml_str.split("><"):
    if any(k in line.lower() for k in ["audio", "talk", "backchannel", "output"]):
        print("MATCH:", line[:120])
