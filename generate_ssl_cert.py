#!/usr/bin/env python3
"""
SSL 인증서 자동 재생성 스크립트
현재 IP 주소로 SSL 인증서를 자동 생성합니다.
"""

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import ipaddress
import os
import shutil

# 현재 IP 주소 설정
CURRENT_IP = "222.106.254.193"

print("=" * 60)
print("🔐 SSL 인증서 자동 재생성")
print("=" * 60)
print()

# 인증서 저장 경로
cert_dir = "shelter_chatbot/cert"
cert_file = f"{cert_dir}/cert.pem"
key_file = f"{cert_dir}/key.pem"

# 디렉토리 생성
os.makedirs(cert_dir, exist_ok=True)

# 기존 인증서 백업
if os.path.exists(cert_file):
    backup_file = f"{cert_dir}/cert_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pem"
    shutil.copy2(cert_file, backup_file)
    print(f"📦 기존 인증서 백업: {backup_file}")

if os.path.exists(key_file):
    backup_file = f"{cert_dir}/key_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pem"
    shutil.copy2(key_file, backup_file)
    print(f"📦 기존 키 백업: {backup_file}")

print()
print("🔨 새 인증서 생성 중...")
print()

# 개인키 생성
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=4096,
)
print("✅ 개인키 생성 완료 (4096-bit RSA)")

# 인증서 주체 정보
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "KR"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Seoul"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "Seoul"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Shelter Chatbot"),
    x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
])

# Subject Alternative Name (SAN) - 모든 접속 가능한 주소 포함
san = x509.SubjectAlternativeName([
    x509.DNSName("localhost"),
    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    x509.IPAddress(ipaddress.IPv4Address(CURRENT_IP)),
])

# 인증서 생성
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(private_key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.now(datetime.UTC))
    .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
    .add_extension(san, critical=False)
    .sign(private_key, hashes.SHA256())
)
print("✅ 인증서 생성 완료 (유효기간: 365일)")

# 개인키 저장
with open(key_file, "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
print(f"✅ 개인키 저장: {key_file}")

# 인증서 저장
with open(cert_file, "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))
print(f"✅ 인증서 저장: {cert_file}")

print()
print("=" * 60)
print("✅ SSL 인증서 재생성 완료!")
print("=" * 60)
print()
print("📋 인증서 정보:")
print(f"   📁 위치: {cert_dir}/")
print(f"   📄 파일: cert.pem, key.pem")
print(f"   ⏰ 유효기간: 365일 ({datetime.datetime.now().strftime('%Y-%m-%d')} ~ {(datetime.datetime.now() + datetime.timedelta(days=365)).strftime('%Y-%m-%d')})")
print()
print("🔐 포함된 호스트:")
print("   - localhost")
print("   - 127.0.0.1")
print(f"   - {CURRENT_IP}")
print()
print("🌐 접속 가능한 주소:")
print("   - https://127.0.0.1:8443/")
print("   - https://localhost:8443/")
print(f"   - https://{CURRENT_IP}:8443/")
print()
print("⚠️  서버를 재시작해야 새 인증서가 적용됩니다!")
print("   실행: python main.py")
print()
